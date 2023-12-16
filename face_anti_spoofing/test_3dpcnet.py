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

    # os.makedirs(cfg.output, exist_ok=True)
    # init_logging(rank, cfg.output)

    # summary_writer = (
    #     SummaryWriter(log_dir=os.path.join(cfg.output, "tensorboard"))
    #     if rank == 0
    #     else None
    # )
    
    # wandb_logger = None
    # run_name = datetime.now().strftime("%y%m%d_%H%M") + f"_GPU{rank}"
    # if cfg.using_wandb:
    #     import wandb
    #     # Sign in to wandb
    #     try:
    #         wandb.login(key=cfg.wandb_key)
    #     except Exception as e:
    #         print("WandB Key must be provided in config file (base.py).")
    #         print(f"Config Error: {e}")
    #     # Initialize wandb
    #     # run_name = datetime.now().strftime("%y%m%d_%H%M") + f"_GPU{rank}"
    #     run_name = run_name if cfg.suffix_run_name is None else run_name + f"_{cfg.suffix_run_name}"
    #     try:
    #         wandb_logger = wandb.init(
    #             entity = cfg.wandb_entity, 
    #             project = cfg.wandb_project, 
    #             sync_tensorboard = True,
    #             resume=cfg.wandb_resume,
    #             name = run_name, 
    #             notes = cfg.notes) if rank == 0 or cfg.wandb_log_all else None
    #         if wandb_logger:
    #             wandb_logger.config.update(cfg)
    #     except Exception as e:
    #         print("WandB Data (Entity and Project name) must be provided in config file (base.py).")
    #         print(f"Config Error: {e}")

    print(f'Loading test paths (dataset: \'{cfg.train_dataset}\')...')
    test_loader = get_dataloader(
        # cfg.rec,          # original
        cfg.train_dataset,  # Bernardo
        cfg.protocol_id,    # Bernardo
        cfg.dataset_path,   # Bernardo
        cfg.frames_path,    # Bernardo
        cfg.img_size,       # Bernardo
        # 'test',
        'val',
        local_rank,
        cfg.batch_size,
        cfg.frames_per_video if hasattr(cfg, 'frames_per_video') else 1,
        cfg.dali,
        cfg.dali_aug,
        cfg.seed,
        cfg.num_workers,
        role='test',
        percent=1.0
    )
    print(f'    test samples: {len(test_loader.dataset)}')

    # print(f'Loading val paths (dataset: \'{cfg.train_dataset}\')...')
    # val_loader = get_dataloader(
    #     # cfg.rec,          # original
    #     cfg.train_dataset,  # Bernardo
    #     cfg.protocol_id,    # Bernardo
    #     cfg.dataset_path,   # Bernardo
    #     cfg.frames_path,    # Bernardo
    #     cfg.img_size,       # Bernardo
    #     'val',
    #     local_rank,
    #     cfg.batch_size,
    #     cfg.dali,
    #     cfg.dali_aug,
    #     cfg.seed,
    #     cfg.num_workers
    # )
    # print(f'    val samples: {len(val_loader.dataset)}')

    print(f'\nBuilding model \'{cfg.network}\'...')
    backbone = get_model(
        '3dpcnet', img_size=cfg.img_size, dropout=0.0, fp16=cfg.fp16, num_features=cfg.embedding_size).cuda()

    backbone = torch.nn.parallel.DistributedDataParallel(
        module=backbone, broadcast_buffers=False, device_ids=[local_rank], bucket_cap_mb=16,
        find_unused_parameters=True)
    backbone.register_comm_hook(None, fp16_compress_hook)

    # backbone.train()
    # FIXME using gradient checkpoint if there are some unused parameters will cause error
    # backbone._set_static_graph()

    # print(f'\nSetting loss function...')
    # margin_loss = CombinedMarginLoss(
    #     64,
    #     cfg.margin_list[0],
    #     cfg.margin_list[1],
    #     cfg.margin_list[2],
    #     cfg.interclass_filtering_threshold
    # )
    # chamfer_loss = ChamferLoss()

    # print(f'\nSetting optimizer...')
    # if cfg.optimizer == "sgd":
    #     module_partial_fc = PartialFC_V2(
    #         # margin_loss, cfg.embedding_size, cfg.num_classes,
    #         margin_loss,   2,                  cfg.num_classes,
    #         cfg.sample_rate, False)
    #     module_partial_fc.train().cuda()
    #     # TODO the params of partial fc must be last in the params list
    #     opt = torch.optim.SGD(
    #         params=[{"params": backbone.parameters()}, {"params": module_partial_fc.parameters()}],
    #         lr=cfg.lr, momentum=0.9, weight_decay=cfg.weight_decay)
    #
    # elif cfg.optimizer == "adamw":
    #     module_partial_fc = PartialFC_V2(
    #         # margin_loss, cfg.embedding_size, cfg.num_classes,
    #         margin_loss,   2,                  cfg.num_classes,
    #         cfg.sample_rate, False)
    #     module_partial_fc.train().cuda()
    #     opt = torch.optim.AdamW(
    #         params=[{"params": backbone.parameters()}, {"params": module_partial_fc.parameters()}],
    #         lr=cfg.lr, weight_decay=cfg.weight_decay)
    # else:
    #     raise

    # cfg.total_batch_size = cfg.batch_size * world_size
    # cfg.warmup_step = cfg.num_image // cfg.total_batch_size * cfg.warmup_epoch
    # cfg.total_step = cfg.num_image // cfg.total_batch_size * cfg.max_epoch

    # lr_scheduler = PolynomialLRWarmup(
    #     optimizer=opt,
    #     warmup_iters=cfg.warmup_step,
    #     total_iters=cfg.total_step)

    # start_epoch = 0
    # global_step = 0
    # if cfg.resume:
    #     dict_checkpoint = torch.load(os.path.join(cfg.output, f"checkpoint_gpu_{rank}.pt"))
    #     start_epoch = dict_checkpoint["epoch"]
    #     global_step = dict_checkpoint["global_step"]
    #     backbone.module.load_state_dict(dict_checkpoint["state_dict_backbone"])
    #     module_partial_fc.load_state_dict(dict_checkpoint["state_dict_softmax_fc"])
    #     opt.load_state_dict(dict_checkpoint["state_optimizer"])
    #     lr_scheduler.load_state_dict(dict_checkpoint["state_lr_scheduler"])
    #     del dict_checkpoint

    margin_loss = CombinedMarginLoss(
        64,
        cfg.margin_list[0],
        cfg.margin_list[1],
        cfg.margin_list[2],
        cfg.interclass_filtering_threshold
    )
    chamfer_loss = ChamferLoss()

    module_partial_fc = PartialFC_V2(
        margin_loss,   2,                  cfg.num_classes,
        cfg.sample_rate, False)
    module_partial_fc.cuda()

    print(f'\nLoading weights \'{args.weights}\'...')
    dict_checkpoint = torch.load(args.weights)
    backbone.module.load_state_dict(dict_checkpoint["state_dict_backbone"])
    module_partial_fc.load_state_dict(dict_checkpoint["state_dict_softmax_fc"])
    # del dict_checkpoint

    # for key, value in cfg.items():
    #     num_space = 25 - len(key)
    #     logging.info(": " + key + " " * num_space + str(value))

    # # Bernardo
    # callback_logging = CallBackEpochLogging(
    #     frequent=cfg.frequent,
    #     total_step=cfg.total_step,
    #     batch_size=len(test_loader),
    #     num_batches=cfg.batch_size,
    #     start_step = global_step,
    #     writer=summary_writer
    # )

    test_evaluator = EvaluatorLogging(num_samples=len(test_loader.dataset),
                                       batch_size=cfg.batch_size,
                                       num_batches=len(test_loader))
    
    # val_evaluator = EvaluatorLogging(num_samples=len(val_loader.dataset),
    #                                  batch_size=cfg.batch_size,
    #                                  num_batches=len(val_loader))

    # reconst_loss_am = AverageMeter()
    # class_loss_am = AverageMeter()
    # total_loss_am = AverageMeter()
    # amp = torch.cuda.amp.grad_scaler.GradScaler(growth_interval=100)

    # patience = 30
    # early_stopping = EarlyStopping(patience=patience, verbose=True, delta=0.01, max_epochs=cfg.max_epoch)

    print(f'\nStarting testing...')
    with torch.no_grad():
        test(chamfer_loss, module_partial_fc, backbone, test_loader, test_evaluator, cfg, args)   # Bernardo



# Bernardo
def test(chamfer_loss, module_partial_fc, backbone, val_loader, val_evaluator, cfg, args):
    with torch.no_grad():
        # module_partial_fc.eval()
        # backbone.eval()
        val_evaluator.reset()

        val_reconst_loss_am = AverageMeter()
        val_class_loss_am = AverageMeter()
        val_total_loss_am = AverageMeter()
        for val_batch_idx, (val_img, val_pointcloud, val_labels) in enumerate(val_loader):
            val_pred_pointcloud, val_pred_logits = backbone(val_img)
            val_loss_reconst = chamfer_loss(val_pointcloud, val_pred_pointcloud)
            val_loss_class, val_probabilities, val_pred_labels = module_partial_fc(val_pred_logits, val_labels)
            val_total_loss = val_loss_reconst + val_loss_class
            
            val_reconst_loss_am.update(val_loss_reconst.item(), 1)
            val_class_loss_am.update(val_loss_class.item(), 1)
            val_total_loss_am.update(val_total_loss.item(), 1)

            val_evaluator.update(val_pred_labels, val_labels)

            if val_batch_idx == 0:
                path_dir_samples = os.path.join('/'.join(args.weights.split('/')[:-1]), f'samples/batch={val_batch_idx}/test')
                print(f'Saving test samples at \'{path_dir_samples}\'...')
                save_sample(path_dir_samples, val_img, val_pointcloud, val_labels,
                            val_pred_pointcloud, val_pred_labels)

        metrics = val_evaluator.evaluate()

        print('Test:    test_ReconstLoss: %.4f    test_ClassLoss: %.4f    test_TotalLoss: %.4f    test_acc: %.4f%%    test_apcer: %.4f%%    test_bpcer: %.4f%%    test_acer: %.4f%%' %
              (val_reconst_loss_am.avg, val_class_loss_am.avg, val_total_loss_am.avg, metrics['acc'], metrics['apcer'], metrics['bpcer'], metrics['acer']))

        val_reconst_loss_am.reset()
        val_class_loss_am.reset()
        val_total_loss_am.reset()



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


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    parser = argparse.ArgumentParser(
        description="Distributed Arcface Training in Pytorch")
    parser.add_argument("--config", type=str, default='configs/oulu-npu_frames_3d_hrn_r18.py', help="Ex: --config configs/oulu-npu_frames_3d_hrn_r18.py")
    parser.add_argument("--weights", type=str, default='work_dirs/oulu-npu_frames_3d_hrn_r18_prot=1_imgsize=224_maxepoch=300_batch=32_lr=0.1_wd=0.0005_embedd=256_20231207_184909/best_model.pt', help="Ex: --config configs/oulu-npu_frames_3d_hrn_r18.py")
    main(parser.parse_args())
