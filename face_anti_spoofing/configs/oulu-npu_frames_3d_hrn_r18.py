from easydict import EasyDict as edict

# make training faster
# our RAM is 256G
# mount -t tmpfs -o size=140G  tmpfs /train_tmp

config = edict()
config.margin_list = (1.0, 0.5, 0.0)
# config.network = "r50"   # original
config.network = "r18"     # Bernardo
config.resume = False
config.output = None
config.embedding_size = 512
config.sample_rate = 1.0
config.fp16 = True
config.momentum = 0.9
config.weight_decay = 5e-4
config.batch_size = 128
config.lr = 0.1
config.verbose = 2000
config.dali = False

# config.rec = "/train_tmp/ms1m-retinaface-t1"   # original
config.train_dataset = 'oulu-npu_frames_3d_hrn'                                               # Bernardo
config.protocol_id = 1                                                                        # Bernardo
config.dataset_path = '/experiments/BOVIFOCR_project/datasets/bjgbiesseck/liveness/oulu-npu'  # Bernardo
config.frames_path = '/datasets1/bjgbiesseck/liveness/HRN_3D_reconstruction/oulu-npu_frames'  # Bernardo

# config.num_classes = 93431   # original
config.num_classes = 2         # Bernardo

# config.num_image = 5179510   # original
config.num_image = 1800        # original

config.num_epoch = 20
config.warmup_epoch = 0
# config.val_targets = ['lfw', 'cfp_fp', "agedb_30"]
