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

#### 3. Train model:
```
export CUDA_VISIBLE_DEVICES=0; python train_model_5thChalearn_FAS_CVPR2024.py --config configs/UniAttackData_3d_hrn_r50.py
```
