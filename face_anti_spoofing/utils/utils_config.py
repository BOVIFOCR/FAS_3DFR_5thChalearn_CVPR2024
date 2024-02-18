import importlib
import os.path as osp
from datetime import datetime


def get_config(args):
    config_file = args.config
    assert config_file.startswith('configs/'), 'config file setting must start with configs/'
    temp_config_name = osp.basename(config_file)
    temp_module_name = osp.splitext(temp_config_name)[0]
    config = importlib.import_module("configs.base")
    cfg = config.config
    config = importlib.import_module("configs.%s" % temp_module_name)
    job_cfg = config.config
    cfg.update(job_cfg)
    date_time = datetime.today().strftime('%Y%m%d_%H%M%S')
    if cfg.output is None:
        # cfg.output = osp.join('work_dirs', temp_module_name)
        cfg.output = osp.join('work_dirs', temp_module_name
                              + (f'_prot={cfg.protocol_id}' if hasattr(cfg, 'protocol_id') else '')
                              + (f'_fpv={cfg.frames_per_video}' if hasattr(cfg, 'frames_per_video') else '_fpv=1')
                              +  f'_imgsize={cfg.img_size}'
                              +  f'_maxepoch={cfg.max_epoch}'
                              +  f'_batch={cfg.batch_size}'
                              +  f'_lr={cfg.lr}'
                              +  f'_wd={cfg.weight_decay}'
                              +  f'_embedd={cfg.embedding_size}'
                              +  f'_{date_time}'
                              + (f'_{args.exp_suffix}' if hasattr(args, 'exp_suffix') and args.exp_suffix!='' else '')
                              )
        cfg.output = cfg.output.replace(' ', '').replace('\'', '')
    return cfg