## TRAIN MODEL (BOVIFOCR)

#### 1. Clone this repo:
```
git clone https://github.com/BOVIFOCR/FAS_3DFR_5thChalearn_CVPR2024.git
cd FAS_3DFR_5thChalearn_CVPR2024
```

#### 2. Create conda env and install python libs:
```
export CONDA_ENV=fas_3dfr_5thchalearn_cvpr2024_py39
conda create -n $CONDA_ENV python=3.9
conda activate $CONDA_ENV
conda env config vars set CUDA_HOME="/usr/local/cuda-11.6"; conda deactivate; conda activate $CONDA_ENV
conda env config vars set LD_LIBRARY_PATH="$CUDA_HOME/lib64"; conda deactivate; conda activate $CONDA_ENV
conda env config vars set PATH="$CUDA_HOME:$CUDA_HOME/bin:$LD_LIBRARY_PATH:$PATH"; conda deactivate; conda activate $CONDA_ENV

conda install pytorch=1.13.0 torchvision pytorch-cuda=11.6 -c pytorch -c nvidia
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
conda install -c bottler nvidiacub
conda install pytorch3d -c pytorch3d
pip3 install -r requirements.txt
```

#### 3. Reconstruct faces with the HRN method:
- https://github.com/BOVIFOCR/HRN_3D_face_reconstruction

#### 4. Train model:
```
export CUDA_VISIBLE_DEVICES=0; python train_model_5thChalearn_FAS_CVPR2024.py --config configs/UniAttackData_3d_hrn_r50.py
```
The trained model will be saved in the folder `work_dirs`.


#### 5 Test model:
```
export CUDA_VISIBLE_DEVICES=0; python test_model_5thChalearn_FAS_CVPR2024.py --config configs/UniAttackData_3d_hrn_r50.py --weights /work_dirs/experiment_folder/best_model.pt --part dev --protocol /path/to/UniAttackData/phase1/p1/dev.txt --img-path /path/to/UniAttackData_align_crop/phase1
```

A file `phase1_p1_dev.txt` will be generated with the image scores:
```
p1/dev/000001.jpg 0.73088217
p1/dev/000002.jpg 0.7308881
p1/dev/000003.jpg 0.73088604
p1/dev/000004.jpg 0.73089004
p1/dev/000005.jpg 0.7308881
p1/dev/000006.jpg 0.7308893
...
p1/dev/005998.jpg 0.7295462
p1/dev/005999.jpg 0.729513
p1/dev/006000.jpg 0.7308867
```


#### 6 Scores files:
Here are the face scores files of our best model submitted in phase1 and phase2 of the 5th Chalearn Face Anti-spoofing Workshop and Challenge@CVPR2024:
- [phase1_p1_p2.1_p2.2_dev.zip](https://github.com/BOVIFOCR/FAS_3DFR_5thChalearn_CVPR2024/blob/master/scores_bovifocr_team/phase1_p1_p2.1_p2.2_dev.zip)
- [phase2_p1_p2.1_p2.2_dev_test.zip](https://github.com/BOVIFOCR/FAS_3DFR_5thChalearn_CVPR2024/blob/master/scores_bovifocr_team/phase2_p1_p2.1_p2.2_dev_test.zip)
