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
from lr_scheduler import PolynomialLRWarmup
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

# from ignite.engine import Engine, Events
# from ignite.handlers import EarlyStopping
from utils.pytorchtools import EarlyStopping


# Commented by Bernardo in order to use torch==1.10.1
# assert torch.__version__ >= "1.12.0", "In order to enjoy the features of the new torch, \
# we have upgraded the torch to 1.12.0. torch before than 1.12.0 may not work in the future."

try:
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    distributed.init_process_group("nccl")
except KeyError:
    rank = 0
    local_rank = 0
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

    torch.cuda.set_device(local_rank)

    print(f'Loading test paths (dataset: \'{cfg.train_dataset}\')...')
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
        cfg.frames_per_video if hasattr(cfg, 'frames_per_video') else 1,
        cfg.dali,
        cfg.dali_aug,
        cfg.seed,
        cfg.num_workers,
        role='test',
        percent=1.0,
        ignore_pointcloud_files=True
    )
    print(f'    test samples: {len(test_loader.dataset)}')

    print(f'\nBuilding model \'{cfg.network}\'...')
    backbone = get_model(
        '3dpcnet', img_size=cfg.img_size, dropout=0.0, fp16=cfg.fp16, num_features=cfg.embedding_size).cuda()

    backbone = torch.nn.parallel.DistributedDataParallel(
        module=backbone, broadcast_buffers=False, device_ids=[local_rank], bucket_cap_mb=16,
        find_unused_parameters=True)
    backbone.register_comm_hook(None, fp16_compress_hook)

    margin_loss = CombinedMarginLoss(
        64,
        cfg.margin_list[0],
        cfg.margin_list[1],
        cfg.margin_list[2],
        cfg.interclass_filtering_threshold
    )
    chamfer_loss = ChamferLoss()

    module_partial_fc = PartialFC_V2(
        margin_loss, 128, cfg.num_classes,
        cfg.sample_rate, False)
    module_partial_fc.cuda()

    print(f'\nLoading weights \'{args.weights}\'...')
    dict_checkpoint = torch.load(args.weights)
    backbone.module.load_state_dict(dict_checkpoint["state_dict_backbone"])
    module_partial_fc.load_state_dict(dict_checkpoint["state_dict_softmax_fc"])
    # del dict_checkpoint

    test_evaluator = EvaluatorLogging(num_samples=len(test_loader.dataset),
                                      batch_size=cfg.batch_size,
                                      num_batches=len(test_loader))

    print(f'\nStarting test...')
    with torch.no_grad():
        test(chamfer_loss, module_partial_fc, backbone, test_loader, test_evaluator, cfg, args)   # Bernardo



# Bernardo
def test(chamfer_loss, module_partial_fc, backbone, test_loader, test_evaluator, cfg, args):
    with torch.no_grad():
        # module_partial_fc.eval()
        # backbone.eval()
        test_evaluator.reset()

        # test_reconst_loss_am = AverageMeter()
        test_class_loss_am = AverageMeter()
        # test_total_loss_am = AverageMeter()

        print(f'\nDoing test...')
        for test_batch_idx, (test_img, test_pointcloud, test_labels) in enumerate(test_loader):
            print(f'batch: {test_batch_idx}/{len(test_loader)}', end='\r')
            test_pred_pointcloud, test_pred_logits = backbone(test_img)
            # test_loss_reconst = chamfer_loss(test_pointcloud, test_pred_pointcloud)
            test_loss_class, test_probabilities, test_pred_labels = module_partial_fc(test_pred_logits, test_labels)
            # test_total_loss = test_loss_reconst + test_loss_class
            
            # test_reconst_loss_am.update(test_loss_reconst.item(), 1)
            test_class_loss_am.update(test_loss_class.item(), 1)
            # test_total_loss_am.update(test_total_loss.item(), 1)

            test_evaluator.update(test_pred_labels, test_labels)

            if args.save_samples and test_batch_idx == 0:
                path_dir_samples = os.path.join('/'.join(args.weights.split('/')[:-1]), f'samples/batch={test_batch_idx}/test')
                print(f'Saving test samples at \'{path_dir_samples}\'...')
                save_sample(path_dir_samples, test_img, test_pointcloud, test_labels,
                            test_pred_pointcloud, test_pred_labels)
        print('')

        metrics = test_evaluator.evaluate()

        # print('Test:    test_ReconstLoss: %.4f    test_ClassLoss: %.4f    test_TotalLoss: %.4f    test_acc: %.4f%%    test_apcer: %.4f%%    test_bpcer: %.4f%%    test_acer: %.4f%%' %
        #       (test_reconst_loss_am.avg, test_class_loss_am.avg, test_total_loss_am.avg, metrics['acc'], metrics['apcer'], metrics['bpcer'], metrics['acer']))
        print('Test:    test_ClassLoss: %.4f    test_acc: %.4f%%    test_apcer: %.4f%%    test_bpcer: %.4f%%    test_acer: %.4f%%' %
              (test_class_loss_am.avg, metrics['acc'], metrics['apcer'], metrics['bpcer'], metrics['acer']))

        # test_reconst_loss_am.reset()
        test_class_loss_am.reset()
        # test_total_loss_am.reset()



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

        if len(true_pointcloud.size()) > 0:
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


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    parser = argparse.ArgumentParser(
        description="Distributed Arcface Training in Pytorch")
    parser.add_argument("--config", type=str, default='configs/oulu-npu_frames_3d_hrn_r18.py', help="Ex: --config configs/oulu-npu_frames_3d_hrn_r18.py")
    parser.add_argument("--weights", type=str, default='work_dirs/oulu-npu_frames_3d_hrn_r18_prot=1_imgsize=224_maxepoch=300_batch=32_lr=0.1_wd=0.0005_embedd=256_20231207_184909/best_model.pt', help="Ex: --config configs/oulu-npu_frames_3d_hrn_r18.py")
    parser.add_argument("--save-samples", action='store_true')
    main(parser.parse_args())
