import os
nodename = os.uname().nodename
from easydict import EasyDict as edict

# make training faster
# our RAM is 256G
# mount -t tmpfs -o size=140G  tmpfs /train_tmp

config = edict()
config.margin_list = (1.0, 0.5, 0.0)

config.network = "r50"   # original

config.resume = False
config.output = None

# config.embedding_size = 512  # original
# config.embedding_size = 256  # Bernardo
config.embedding_size = 2048  # Bernardo

config.sample_rate = 1.0
# config.fp16 = True       # original
config.fp16 = False        # Bernardo
config.momentum = 0.9

# config.weight_decay = 5e-4
config.weight_decay = 5e-6

# config.batch_size = 128  # original
# config.batch_size = 64   # Bernardo
config.batch_size = 32     # Bernardo
# config.batch_size = 16   # Bernardo

# config.lr = 0.1
# config.lr = 0.05
# config.lr = 0.005
config.lr = 0.0025
# config.lr = 0.0015
# config.lr = 0.001
# config.lr = 0.0005
# config.lr = 0.00005

# config.verbose = 2000  # original for 5.1M images
config.verbose = 100     # Bernardo
# config.verbose = 10    # Bernardo

config.dali = False

# config.rec = "/train_tmp/ms1m-retinaface-t1"                                                # original
# config.train_dataset = 'oulu-npu_frames_3d_hrn'                                             # Bernardo
config.train_dataset = 'oulu-npu_frames'                                                      # Bernardo
config.protocol_id = 1                                                                        # Bernardo
# config.protocol_id = 2
# config.protocol_id = 3
# config.protocol_id = 4

if nodename == 'duo':
    config.dataset_path = '/experiments/BOVIFOCR_project/datasets/bjgbiesseck/liveness/oulu-npu'  # Bernardo
    config.frames_path = '/datasets1/bjgbiesseck/liveness/HRN_3D_reconstruction/oulu-npu_frames'  # Bernardo

elif nodename == 'diolkos':
    config.dataset_path = '/nobackup/unico/datasets/liveness/oulu-npu'                                   # Bernardo
    config.frames_path = '/nobackup/unico/datasets/liveness/3D_face_reconstruction/HRN/oulu-npu_frames'  # Bernardo

elif nodename == 'peixoto':
    config.dataset_path = '/nobackup1/bjgbiesseck/datasets/liveness/oulu-npu'                          # Bernardo
    # config.frames_path = '/nobackup1/bjgbiesseck/datasets/3D_face_reconstruction/HRN/oulu-npu_frames'  # Bernardo
    config.frames_path = '/nobackup1/bjgbiesseck/datasets/liveness/oulu-npu_frames_cropped_align'  # Bernardo

# config.img_size = 112        # Bernardo
config.img_size = 224          # Bernardo

# config.num_classes = 93431   # original
# config.num_classes = 2         # (live or spoof) Bernardo
config.num_classes = 5       # (0=real; 1=print1; 2=print2; 3=video-replay1; 4=video-replay2) Bernardo

# config.num_image = 5179510   # original
config.num_image = 1800        # Bernardo

# config.max_epoch = 20
# config.max_epoch = 50
# config.max_epoch = 100
config.max_epoch = 300

config.warmup_epoch = 0
# config.val_targets = ['lfw', 'cfp_fp', "agedb_30"]
