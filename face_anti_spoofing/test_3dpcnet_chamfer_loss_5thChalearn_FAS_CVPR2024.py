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


def normalize_img(img):
    img = np.transpose(img, (2, 0, 1))  # from (224,224,3) to (3,224,224)
    img = ((img/255.)-0.5)/0.5
    return img


def load_img(img_path):
    img_bgr = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return img_rgb.astype(np.float32)








def main(args):
    # get config
    # cfg = get_config(args.config)
    cfg = get_config(args)
    # global control random seed
    setup_seed(seed=cfg.seed, cuda_deterministic=False)

    torch.cuda.set_device(local_rank)

    # print(f'\nLoading train paths (dataset: \'{cfg.train_dataset}\')...')
    # train_loader = get_dataloader(
    #     # cfg.rec,                    # original
    #     cfg.train_dataset,            # Bernardo
    #     args.protocol,                # Bernardo
    #     cfg.dataset_path,             # Bernardo
    #     [cfg.rgb_path, cfg.pc_path],  # Bernardo
    #     cfg.img_size,                 # Bernardo
    #     args.part,
    #     local_rank,
    #     # cfg.batch_size,
    #     args.batch,
    #     cfg.frames_per_video if hasattr(cfg, 'frames_per_video') else 1,
    #     cfg.dali,
    #     cfg.dali_aug,
    #     cfg.seed,
    #     cfg.num_workers,
    #     role='test',
    #     percent=1.0,
    #     protocol_data=None,
    #     filter_valid_samples=False if args.part=='dev' or args.part=='test' else True,
    #     shuffle_samples=False
    # )
    # print(f'    train samples: {len(train_loader.dataset)}')

    print(f'\nBuilding model \'{cfg.network}\'...')
    backbone = get_model(
        '3dpcnet_reconst_classifMLP', encoder_name=cfg.network, img_size=cfg.img_size, dropout=0.0, fp16=cfg.fp16, num_features=cfg.embedding_size).cuda()

    backbone = torch.nn.parallel.DistributedDataParallel(
        module=backbone, broadcast_buffers=False, device_ids=[local_rank], bucket_cap_mb=16,
        find_unused_parameters=True)
    backbone.register_comm_hook(None, fp16_compress_hook)

    # backbone.train()

    print(f'\nSetting loss function...')
    margin_loss = CombinedMarginLoss(
        # 64,
        4,
        cfg.margin_list[0],
        cfg.margin_list[1],
        cfg.margin_list[2],
        cfg.interclass_filtering_threshold
    )
    chamfer_loss = ChamferLoss()

    module_partial_fc = PartialFC_V2(
        # margin_loss, cfg.embedding_size, cfg.num_classes,
        margin_loss, 128,                cfg.num_classes,
        cfg.sample_rate, False)
    module_partial_fc.train().cuda()

    cfg.total_batch_size = cfg.batch_size * world_size
    cfg.warmup_step = cfg.num_image // cfg.total_batch_size * cfg.warmup_epoch
    cfg.total_step = cfg.num_image // cfg.total_batch_size * cfg.max_epoch

    print(f'Loading weights \'{args.weights}\'')
    dict_checkpoint = torch.load(args.weights)
    backbone.module.load_state_dict(dict_checkpoint["state_dict_backbone"])
    module_partial_fc.load_state_dict(dict_checkpoint["state_dict_softmax_fc"])
    print('Done!')


    if args.test_one_sample:
        print(f'\nTesting one sample...')
        with torch.no_grad():
            backbone.eval()            # Bernardo
            module_partial_fc.eval()   # Bernardo

            img = load_img(args.input_img)
            img = normalize_img(img)
            img = torch.from_numpy(img).cuda()
            img = torch.unsqueeze(img, 0)
            print('img.size():', img.size(), '    img.device:', img.device)

            local_labels = torch.tensor([args.true_label]).cuda()
            print('local_labels.size():', local_labels.size(), '    local_labels.device:', local_labels.device)

            loaded_pred_pointcloud = read_obj(args.pred_pc)['vertices']
            loaded_pred_pointcloud = torch.from_numpy(loaded_pred_pointcloud).cuda()
            loaded_pred_pointcloud = torch.unsqueeze(loaded_pred_pointcloud, 0)
            print('loaded_pred_pointcloud.size():', loaded_pred_pointcloud.size(), '    loaded_pred_pointcloud.device:', loaded_pred_pointcloud.device)

            true_pointcloud = read_obj(args.true_pc)['vertices']
            true_pointcloud = torch.from_numpy(true_pointcloud).cuda()
            true_pointcloud = torch.unsqueeze(true_pointcloud, 0)
            print('true_pointcloud.size():', true_pointcloud.size(), '    true_pointcloud.device:', true_pointcloud.device)

            print('')
            face_embedd, curr_pred_pointcloud, pred_logits = backbone(img)
            print('face_embedd:', face_embedd)
            print('face_embedd.min():', face_embedd.min(), '    face_embedd.max():', face_embedd.max())
            print('pred_logits:', pred_logits)
            print('pred_logits.min():', pred_logits.min(), '    pred_logits.max():', pred_logits.max())
            print('curr_pred_pointcloud:', curr_pred_pointcloud)
            print('curr_pred_pointcloud.size():', curr_pred_pointcloud.size())
            print('curr_pred_pointcloud.min():', curr_pred_pointcloud.min(), '    curr_pred_pointcloud.max():', curr_pred_pointcloud.max())

            reconst_loss_loaded_pred_pointcloud = chamfer_loss(true_pointcloud, loaded_pred_pointcloud)
            reconst_loss_curr_pred_pointcloud = chamfer_loss(true_pointcloud, curr_pred_pointcloud)
            print('')
            print('reconst_loss_loaded_pred_pointcloud:', reconst_loss_loaded_pred_pointcloud)
            print('reconst_loss_curr_pred_pointcloud:', reconst_loss_curr_pred_pointcloud)

            class_loss, probabilities, pred_labels = module_partial_fc(pred_logits, local_labels)
            print('')
            print('class_loss:', class_loss)
            print('probabilities:', probabilities)
            print('pred_labels:', pred_labels)
            print('true_labels:', local_labels)

            curr_pred_pointcloud_path = '/home/bjgbiesseck/curr_pred_pointcloud.obj'
            print(f'\nSaving curr_pred_pointcloud: {curr_pred_pointcloud_path}...')
            write_obj(curr_pred_pointcloud_path, curr_pred_pointcloud[0].cpu().numpy())



    else:
        print(f'\nLoading train paths (dataset: \'{cfg.train_dataset}\')...')
        train_loader = get_dataloader(
            # cfg.rec,                    # original
            cfg.train_dataset,            # Bernardo
            args.protocol,                # Bernardo
            cfg.dataset_path,             # Bernardo
            [args.img_path, cfg.pc_path], # Bernardo
            cfg.img_size,                 # Bernardo
            args.part,
            local_rank,
            # cfg.batch_size,
            args.batch,
            cfg.frames_per_video if hasattr(cfg, 'frames_per_video') else 1,
            cfg.dali,
            cfg.dali_aug,
            cfg.seed,
            cfg.num_workers,
            role='test',
            percent=1.0,
            protocol_data=None,
            filter_valid_samples=False if args.part=='dev' or args.part=='test' else True,
            shuffle_samples=False
        )
        print(f'    train samples: {len(train_loader.dataset)}')

        epoch = 0
        global_step = 0
        summary_writer = None
        callback_logging = CallBackEpochLogging(
            frequent=cfg.frequent,
            total_step=cfg.total_step,
            batch_size=len(train_loader),
            num_batches=cfg.batch_size,
            start_step = global_step,
            writer=summary_writer
        )
        
        train_evaluator = EvaluatorLogging(num_samples=len(train_loader.dataset),
                                        batch_size=cfg.batch_size,
                                        num_batches=len(train_loader))

        reconst_loss_am = AverageMeter()
        class_loss_am = AverageMeter()
        total_loss_am = AverageMeter()
        amp = torch.cuda.amp.grad_scaler.GradScaler(growth_interval=100)

        all_input_imgs = np.zeros((len(train_loader.dataset),3,224,224), dtype=np.float32)
        all_true_pointcloud = np.zeros((len(train_loader.dataset),2500,3), dtype=np.float32)
        all_pred_pointcloud = np.zeros((len(train_loader.dataset),2500,3), dtype=np.float32)
        all_true_labels = np.zeros((len(train_loader.dataset),), dtype=np.float32)
        all_pred_labels = np.zeros((len(train_loader.dataset),), dtype=np.float32)
        all_probs_real_face = np.zeros((len(train_loader.dataset),), dtype=np.float32)

        print(f'\nStarting test...')
        with torch.no_grad():
            for batch_idx, (img, true_pointcloud, local_labels) in enumerate(train_loader):   # Bernardo
                print(f'batch_idx: {batch_idx+1}/{len(train_loader)}', end='\r')
                backbone.eval()            # Bernardo
                module_partial_fc.eval()   # Bernardo

                # loss: torch.Tensor = module_partial_fc(local_embeddings, local_labels)   # original
                pred_face_embedd, pred_pointcloud, pred_logits = backbone(img)
                # reconst_loss = chamfer_loss(true_pointcloud, pred_pointcloud)                           # Bernardo
                class_loss, probabilities, pred_labels = module_partial_fc(pred_logits, local_labels)     # Bernardo
                # total_loss = reconst_loss + class_loss

                train_evaluator.update(pred_labels, local_labels, probabilities[:,1])


                # batch_probs_real_face = probabilities[:,0].cpu().numpy()  # probability of real
                batch_probs_real_face = probabilities[:,1].cpu().numpy()    # probability of spoof

                sample_start_batch = batch_idx*cfg.batch_size
                sample_end_batch = sample_start_batch+img.size(0)

                batch_img_paths = train_loader.dataset.protocol_data[sample_start_batch:sample_end_batch]
                all_input_imgs[sample_start_batch:sample_end_batch] = img.cpu().numpy()
                all_true_pointcloud[sample_start_batch:sample_end_batch] = true_pointcloud.cpu().numpy()
                all_pred_pointcloud[sample_start_batch:sample_end_batch] = pred_pointcloud.cpu().numpy()
                all_true_labels[sample_start_batch:sample_end_batch] = local_labels.cpu().numpy()
                all_pred_labels[sample_start_batch:sample_end_batch] = pred_labels.cpu().numpy()
                all_probs_real_face[sample_start_batch:sample_end_batch] = batch_probs_real_face

                print('\nbatch_img_paths:', batch_img_paths)
                # print('probabilities:', probabilities)
                print('local_labels:', local_labels)
                print('pred_labels:', pred_labels)
                print('batch_probs_real_face:', batch_probs_real_face)
                # print('all_probs_real_face:', all_probs_real_face)
                print('-----------------')
                # sys.exit(0)

            print('')

            # callback_logging(global_step, reconst_loss_am, class_loss_am, total_loss_am, train_evaluator,
            #                  epoch, cfg.fp16, 0.0, amp)

            output_path_dir = os.path.join(os.path.dirname(args.weights), args.output_dir_name, args.part)
            os.makedirs(output_path_dir, exist_ok=True)
            output_file_name = '_'.join(args.protocol.split('/')[-3:])
            output_path_scores_file = os.path.join(output_path_dir, output_file_name)
            # print('output_path_scores_file:', output_path_scores_file)
            print(f'Saving scores file: \'{output_path_scores_file}\'')
            save_scores_text_file(train_loader.dataset.protocol_data, all_probs_real_face, file_path=output_path_scores_file)

            if args.save_pred_samples:
                path_save_test_samples = os.path.join(output_path_dir, 'preds_samples')
                # save_samples(path_save_test_samples, batch_img_paths, local_labels.cpu().numpy(), pred_labels.cpu().numpy(), pred_pointcloud.cpu().numpy())
                print('')
                save_samples(path_save_test_samples,
                            train_loader.dataset.protocol_data,
                            all_input_imgs,
                            all_true_pointcloud,
                            all_pred_pointcloud,
                            all_true_labels,
                            all_pred_labels,
                            all_probs_real_face)
            
            metrics = train_evaluator.evaluate()
            print('\nTest:    test_acc: %.4f%%    test_auc: %.4f%%    val_apcer: %.4f%%    val_bpcer: %.4f%%    val_acer: %.4f%%' %
                (metrics['acc'], metrics['auc_roc'], metrics['apcer'], metrics['bpcer'], metrics['acer']))

    print('Finished')


def save_scores_text_file(protocol_data, all_probs_real_face, file_path):
    assert len(protocol_data) == len(all_probs_real_face), f'Error, len(protocol_data) ({len(protocol_data)}) != len(all_probs_real_face) ({len(all_probs_real_face)})'
    with open(file_path, 'w') as file:
        for idx, (img_path, prob_real_face) in enumerate(zip(protocol_data, all_probs_real_face)):
            if type(img_path) is list:
                img_path = img_path[0]
            # if idx > 0:
            #     file.write('\n')
            file.write(img_path + ' ' + str(prob_real_face) + '\n')
            file.flush()    

def get_pred_group(true_label, pred_label):
    pred_group = ''    # For test samples pred_label == -1
    if int(true_label) == 0 and int(pred_label) == 0:
        pred_group = 'TP'
    elif int(true_label) == 0 and int(pred_label) == 1:
        pred_group = 'FN'
    elif int(true_label) == 1 and int(pred_label) == 0:
        pred_group = 'FP'
    elif int(true_label) == 1 and int(pred_label) == 1:
        pred_group = 'TN'
    return pred_group
    

def save_samples(path_dir_samples, protocol_data, input_imgs, true_pointcloud, pred_pointcloud, true_labels, pred_labels, probs_real_face):
    for i in range(len(protocol_data)):
        subdir = '/'.join(protocol_data[i][0].split('/')[:-1])
        sample_name = protocol_data[i][0].replace('/', '_')
        sample_dir = f'{sample_name}_truelabel={int(true_labels[i])}_predlabel={int(pred_labels[i])}_predprob={probs_real_face[i]}'
        pred_group = get_pred_group(int(true_labels[i]), int(pred_labels[i]))
        path_dir_sample = os.path.join(path_dir_samples, subdir, pred_group, sample_dir).replace('//','/')
        print(f'\npath_dir_sample: \'{path_dir_sample}\'')
        os.makedirs(path_dir_sample, exist_ok=True)

        img_rgb = np.transpose(input_imgs[i], (1, 2, 0))  # from (3,224,224) to (224,224,3)
        img_rgb = (((img_rgb*0.5)+0.5)*255).astype(np.uint8)
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        path_img = os.path.join(path_dir_sample, f'img_{sample_name}.png')
        print(f'path_img: \'{path_img}\'')
        cv2.imwrite(path_img, img_bgr)

        path_pred_pc = os.path.join(path_dir_sample, f'pred_pointcloud_{sample_name}.obj')
        print(f'path_pred_pc: \'{path_pred_pc}\'')
        write_obj(path_pred_pc, pred_pointcloud[i])

        if int(true_labels[i]) > -1:
            path_true_pc = os.path.join(path_dir_sample, f'true_pointcloud_{sample_name}.obj')
            print(f'path_true_pc: \'{path_true_pc}\'')
            write_obj(path_true_pc, true_pointcloud[i])

        print('-----------------')


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
    parser.add_argument("--config", type=str, default='configs/UniAttackData_3d_hrn_r50.py', help="Ex: --config configs/UniAttackData_3d_hrn_r50.py")
    parser.add_argument("--weights", type=str, default='work_dirs/UniAttackData_3d_hrn_r50_prot=[\'p1\',\'p2.1\',\'p2.2\']_fpv=-1_imgsize=224_maxepoch=300_batch=8_lr=0.0025_wd=5e-06_embedd=256_20240214_220712/best_model.pt', help="Ex: --weights work_dirs/UniAttackData_3d_hrn_r50_prot=[\'p1\',\'p2.1\',\'p2.2\']_fpv=-1_imgsize=224_maxepoch=300_batch=8_lr=0.0025_wd=5e-06_embedd=256_20240214_220712/best_model.pt")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--part", type=str, default='dev', help="Ex: --part dev")
    parser.add_argument("--protocol", type=str, default='/datasets1/bjgbiesseck/liveness/fas_cvpr2024/UniAttackData/phase1/p1/dev.txt', help="Ex: --protocol /datasets1/bjgbiesseck/liveness/fas_cvpr2024/UniAttackData/phase1/p1/dev.txt")
    parser.add_argument("--img-path", type=str, default='/datasets1/bjgbiesseck/liveness/fas_cvpr2024/UniAttackData_bbox_crop/phase1', help="Ex: --img_path /datasets1/bjgbiesseck/liveness/fas_cvpr2024/UniAttackData_bbox_crop/phase1")
    parser.add_argument("--output-dir-name", type=str, default='scores_5thChalearn_FAS_CVPR2024', help="Ex: --img_path /datasets1/bjgbiesseck/liveness/fas_cvpr2024/UniAttackData_bbox_crop/phase1")
    parser.add_argument("--save-pred-samples", action='store_true')

    parser.add_argument("--test-one-sample", action='store_true')
    parser.add_argument("--input_img", type=str, default='/home/bjgbiesseck/GitHub/BOVIFOCR_3DPC-Net/face_anti_spoofing/work_dirs/UniAttackData_3d_hrn_r50_prot=[p1,p2.1,p2.2]_fpv=-1_imgsize=224_maxepoch=300_batch=8_lr=0.0025_wd=5e-06_embedd=256_20240218_175713_WHOLE_TRAIN_NO_VAL/scores_5thChalearn_FAS_CVPR2024/train/preds_samples/p1/train/p1_train_000001.jpg_truelabel=1_predlabel=1_predprob=0.7305814623832703/img_p1_train_000001.jpg.png')
    parser.add_argument("--pred_pc", type=str, default='/home/bjgbiesseck/GitHub/BOVIFOCR_3DPC-Net/face_anti_spoofing/work_dirs/UniAttackData_3d_hrn_r50_prot=[p1,p2.1,p2.2]_fpv=-1_imgsize=224_maxepoch=300_batch=8_lr=0.0025_wd=5e-06_embedd=256_20240218_175713_WHOLE_TRAIN_NO_VAL/scores_5thChalearn_FAS_CVPR2024/train/preds_samples/p1/train/p1_train_000001.jpg_truelabel=1_predlabel=1_predprob=0.7305814623832703/pred_pointcloud_p1_train_000001.jpg.obj')
    parser.add_argument("--true_pc", type=str, default='/home/bjgbiesseck/GitHub/BOVIFOCR_3DPC-Net/face_anti_spoofing/work_dirs/UniAttackData_3d_hrn_r50_prot=[p1,p2.1,p2.2]_fpv=-1_imgsize=224_maxepoch=300_batch=8_lr=0.0025_wd=5e-06_embedd=256_20240218_175713_WHOLE_TRAIN_NO_VAL/scores_5thChalearn_FAS_CVPR2024/train/preds_samples/p1/train/p1_train_000001.jpg_truelabel=1_predlabel=1_predprob=0.7305814623832703/true_pointcloud_p1_train_000001.jpg.obj')
    parser.add_argument("--true_label", type=int, default=1)


    # print(parser.parse_args())
    # sys.exit(0)

    main(parser.parse_args())
