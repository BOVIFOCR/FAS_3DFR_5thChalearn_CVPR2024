import importlib
import os.path as osp


def get_config(config_file):
    assert config_file.startswith('configs/'), 'config file setting must start with configs/'
    temp_config_name = osp.basename(config_file)
    temp_module_name = osp.splitext(temp_config_name)[0]
    config = importlib.import_module("configs.base")
    cfg = config.config
    config = importlib.import_module("configs.%s" % temp_module_name)
    job_cfg = config.config
    cfg.update(job_cfg)
    if cfg.output is None:
        # cfg.output = osp.join('work_dirs', temp_module_name)
        cfg.output = osp.join('work_dirs', temp_module_name
                              + (f'_prot={cfg.protocol_id}' if hasattr(cfg, 'protocol_id') else '')
                              + f'_imgsize={cfg.img_size}'
                              + f'_epoch={cfg.num_epoch}'
                              + f'_batch={cfg.batch_size}'
                              + f'_lr={cfg.lr}'
                              + f'_wd={cfg.weight_decay}'
                              + f'_embedd={cfg.embedding_size}'
                              )
    return cfg