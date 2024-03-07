import argparse
import logging
import os, sys
from datetime import datetime

import random
import numpy as np
import cv2
import torch
from backbones import get_model
from dataset import get_dataloader
from losses import CombinedMarginLoss, ChamferLoss
# from lr_scheduler import PolynomialLRWarmup
from partial_fc_v2 import PartialFC_V2
from torch import distributed
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils.utils_callbacks import CallBackLogging, CallBackEpochLogging, CallBackVerification, EvaluatorLogging
from utils.utils_config import get_config
from utils.utils_distributed_sampler import setup_seed
from utils.utils_logging import AverageMeter, init_logging
from utils.utils_pointcloud import read_obj, write_obj
from torch.distributed.algorithms.ddp_comm_hooks.default_hooks import fp16_compress_hook

rank = 0
world_size = 1
distributed.init_process_group(
    backend="nccl",
    # init_method="tcp://127.0.0.1:12584",    # original
    init_method="tcp://127.0.0.1:" + str(int(random.random() * 10000 + 12000)),    # Bernardo
    rank=rank,
    world_size=world_size,
)

def main(args):

    # get config
    cfg = get_config(args.config)
    # global control random seed
    setup_seed(seed=cfg.seed, cuda_deterministic=False)

    local_rank = 0
    torch.cuda.set_device(local_rank)

    '''
    os.makedirs(cfg.output, exist_ok=True)
    init_logging(rank, cfg.output)
    '''

    summary_writer = (
        SummaryWriter(log_dir=os.path.join(cfg.output, "tensorboard"))
        if rank == 0
        else None
    )
    
    wandb_logger = None
    run_name = datetime.now().strftime("%y%m%d_%H%M") + f"_GPU{rank}"
    if cfg.using_wandb:
        import wandb
        # Sign in to wandb
        try:
            wandb.login(key=cfg.wandb_key)
        except Exception as e:
            print("WandB Key must be provided in config file (base.py).")
            print(f"Config Error: {e}")
        # Initialize wandb
        # run_name = datetime.now().strftime("%y%m%d_%H%M") + f"_GPU{rank}"
        run_name = run_name if cfg.suffix_run_name is None else run_name + f"_{cfg.suffix_run_name}"
        try:
            wandb_logger = wandb.init(
                entity = cfg.wandb_entity, 
                project = cfg.wandb_project, 
                sync_tensorboard = True,
                resume=cfg.wandb_resume,
                name = run_name, 
                notes = cfg.notes) if rank == 0 or cfg.wandb_log_all else None
            if wandb_logger:
                wandb_logger.config.update(cfg)
        except Exception as e:
            print("WandB Data (Entity and Project name) must be provided in config file (base.py).")
            print(f"Config Error: {e}")


    print(f'\nBuilding model \'{cfg.network}\'...')
    backbone = get_model(
        '3dpcnet', img_size=cfg.img_size, dropout=0.0, fp16=cfg.fp16, num_features=cfg.embedding_size).cuda()

    backbone = torch.nn.parallel.DistributedDataParallel(
        module=backbone, broadcast_buffers=False, device_ids=[local_rank], bucket_cap_mb=16,
        find_unused_parameters=True)
    backbone.register_comm_hook(None, fp16_compress_hook)
    backbone.eval().cuda()

    margin_loss = CombinedMarginLoss(
        64,
        cfg.margin_list[0],
        cfg.margin_list[1],
        cfg.margin_list[2],
        cfg.interclass_filtering_threshold
    )

    module_partial_fc = PartialFC_V2(
        # margin_loss, cfg.embedding_size, cfg.num_classes,
        margin_loss,   2,                  cfg.num_classes,
        cfg.sample_rate, False)
    module_partial_fc.eval().cuda()

    checkpoint = '/home/bjgbiesseck/GitHub/BOVIFOCR_3DPC-Net/face_anti_spoofing/work_dirs/oulu-npu_frames_3d_hrn_r18_prot=1_imgsize=224_maxepoch=300_batch=32_lr=0.1_wd=0.0005_embedd=256/best_model.pt'
    print(f'Loading trained model \'{checkpoint}\'...')
    dict_checkpoint = torch.load(checkpoint)
    start_epoch = dict_checkpoint["epoch"]
    global_step = dict_checkpoint["global_step"]
    backbone.module.load_state_dict(dict_checkpoint["state_dict_backbone"])
    module_partial_fc.load_state_dict(dict_checkpoint["state_dict_softmax_fc"])
    # opt.load_state_dict(dict_checkpoint["state_optimizer"])
    # lr_scheduler.load_state_dict(dict_checkpoint["state_lr_scheduler"])
    del dict_checkpoint



    print(f'\nLoading validation paths (dataset: \'{cfg.train_dataset}\')...')
    val_loader = get_dataloader(
        # cfg.rec,          # original
        cfg.train_dataset,  # Bernardo
        cfg.protocol_id,    # Bernardo
        cfg.dataset_path,   # Bernardo
        cfg.frames_path,    # Bernardo
        cfg.img_size,       # Bernardo
        'val',
        local_rank,
        cfg.batch_size,
        cfg.dali,
        cfg.dali_aug,
        cfg.seed,
        cfg.num_workers
    )

    val_evaluator = EvaluatorLogging(num_samples=len(val_loader.dataset),
                                      batch_size=cfg.batch_size,
                                      num_batches=len(val_loader))

    print(f'    val samples: {len(val_loader.dataset)}')



    print(f'\nLoading test paths (dataset: \'{cfg.train_dataset}\')...')
    test_loader = get_dataloader(
        # cfg.rec,          # original
        cfg.train_dataset,  # Bernardo
        cfg.protocol_id,    # Bernardo
        cfg.dataset_path,   # Bernardo
        cfg.frames_path,    # Bernardo
        cfg.img_size,       # Bernardo
        'test',
        local_rank,
        cfg.batch_size,
        cfg.dali,
        cfg.dali_aug,
        cfg.seed,
        cfg.num_workers
    )

    test_evaluator = EvaluatorLogging(num_samples=len(test_loader.dataset),
                                      batch_size=cfg.batch_size,
                                      num_batches=len(test_loader))

    print(f'    test samples: {len(test_loader.dataset)}')


    print('\nValidating model...')
    validate(module_partial_fc, backbone, val_loader, val_evaluator)

    print('\nTesting model...')
    test(module_partial_fc, backbone, test_loader, test_evaluator)



# Bernardo
def validate(module_partial_fc, backbone, val_loader, val_evaluator):
    with torch.no_grad():
        module_partial_fc.eval()
        backbone.eval()
        val_evaluator.reset()

        all_confidences = -torch.ones((len(val_loader.dataset)), dtype=torch.float32)
        idx_conf = 0

        for val_batch_idx, (val_img, val_pointcloud, val_labels) in enumerate(val_loader):
            print(f'batch {val_batch_idx}/{len(val_loader)}', end='\r')
            val_pred_pointcloud, val_pred_logits = backbone(val_img)
            val_loss_class, val_probs, _ = module_partial_fc(val_pred_logits, val_labels)
            
            confidences = val_probs[:,1]
            all_confidences[idx_conf:idx_conf+val_img.shape[0]] = confidences
            # val_pred_labels = torch.ge(confidences, 0.5).type(torch.int)
            # print('confidences:', confidences)
            # print('val_pred_labels:', val_pred_labels)
            # print('all_confidences:', all_confidences)
            # sys.exit(0)
            idx_conf += val_img.shape[0]

            # val_evaluator.update(val_pred_labels, val_labels)

        # metrics = val_evaluator.evaluate()
        # print('\nValidation:    val_acc: %.4f%%    val_apcer: %.4f%%    val_bpcer: %.4f%%    val_acer: %.4f%%' %
        #     (metrics['acc'], metrics['apcer'], metrics['bpcer'], metrics['acer']))

        print('all_confidences:', all_confidences)
        print('all_confidences.shape:', all_confidences.shape)
        sys.exit(0)

        thresholds = np.arange(0, 1, 0.01)
        for thresh_idx, thresh in enumerate(thresholds):
            pass

def test(module_partial_fc, backbone, test_loader, test_evaluator):
    with torch.no_grad():
        module_partial_fc.eval()
        backbone.eval()
        test_evaluator.reset()

        for test_batch_idx, (test_img, test_pointcloud, test_labels) in enumerate(test_loader):
            print(f'batch {test_batch_idx}/{len(test_loader)}', end='\r')
            test_pred_pointcloud, val_pred_logits = backbone(test_img)
            test_loss_class, val_pred_labels = module_partial_fc(val_pred_logits, test_labels)

            test_evaluator.update(val_pred_labels, test_labels)

        metrics = test_evaluator.evaluate()

        print('\nTest:    test_acc: %.4f%%    test_apcer: %.4f%%    test_bpcer: %.4f%%    test_acer: %.4f%%' %
              (metrics['acc'], metrics['apcer'], metrics['bpcer'], metrics['acer']))


'''
def save_sample(path_dir_samples, img, true_pointcloud, local_labels, pred_pointcloud, pred_labels):
    for i in range(img.size(0)):
        sample_dir = f'sample={i}_true-label={local_labels[i]}_pred-label={pred_labels[i]}'
        path_sample = os.path.join(path_dir_samples, sample_dir)
        os.makedirs(path_sample, exist_ok=True)

        img_rgb = np.transpose(img[i].cpu().numpy(), (1, 2, 0))  # from (3,224,224) to (224,224,3)
        # img = ((img/255.)-0.5)/0.5
        img_rgb = (((img_rgb*0.5)+0.5)*255).astype(np.uint8)
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        path_img = os.path.join(path_sample, f'img.png')
        cv2.imwrite(path_img, img_bgr)

        path_true_pc = os.path.join(path_sample, f'true_pointcloud.obj')
        write_obj(path_true_pc, true_pointcloud[i])

        path_pred_pc = os.path.join(path_sample, f'pred_pointcloud.obj')
        write_obj(path_pred_pc, pred_pointcloud[i])


def save_model(checkpoint, path_save_model, cfg, wandb_logger, run_name, epoch):
    print(f'Saving model \'{path_save_model}\'...')
    torch.save(checkpoint, path_save_model)

    if wandb_logger and cfg.save_artifacts:
        import wandb
        artifact_name = f"{run_name}_E{epoch}"
        model = wandb.Artifact(artifact_name, type='model')
        model.add_file(path_save_model)
        wandb_logger.log_artifact(model)
'''


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    parser = argparse.ArgumentParser(
        description="Distributed Arcface Training in Pytorch")
    parser.add_argument("--config", type=str, default='configs/oulu-npu_frames_3d_hrn_r18.py', help="Ex: --config configs/oulu-npu_frames_3d_hrn_r18.py")
    main(parser.parse_args())
