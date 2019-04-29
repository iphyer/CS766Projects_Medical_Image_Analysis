#!/usr/bin/env bash

# set up queue
#SBATCH -p slurm_sbel_cmg
#SBATCH --account=cmg --qos=cmg_owner

## Request one CPU core from the scheduler
#SBATCH -c 1

## Request a GPU from the scheduler, we don't care what kind
#SBATCH --gres=gpu:gtx1080:1
#SBATCH -t 14-2:00 # time (D-HH:MM)

## Create a unique output file for the job
#SBATCH -o cuda_Training-%j.log

source activate yolo 
## Load CUDA into your environment
#module load cuda/9.0
## Load CUDA into your environment
module load cuda/9.0

source activate Python3.6
# install cudatoolkit and cudnn
conda install -c anaconda cudatoolkit --yes
conda install -c anaconda cudnn --yes

## Run the installe
pip install numpy
pip install tensorflow-gpu==1.8
pip install numpy scipy scikit-learn pandas matplotlib seaborn
pip install Pillow
pip uninstall cupy
pip install keras
pip install cupy-cuda90
pip install opencv-python

export CUDA_HOME=/usr/local/cuda
export PATH=$PATH:$CUDA_HOME/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDA_HOME/lib64
# this installs the right pip and dependencies for the fresh python

# maskrcnn_benchmark and coco api dependencies

#export INSTALL_DIR=$PWD
# install pycocotools
#cd $INSTALL_DIR
#git clone https://github.com/cocodataset/cocoapi.git
#cd cocoapi/PythonAPI
#python setup.py build_ext install

# install PyTorch Detection
#cd $INSTALL_DIR
#git clone https://github.com/facebookresearch/maskrcnn-benchmark.git
#cd maskrcnn-benchmark

# follow PyTorch installation in https://pytorch.org/get-started/locally/
# we give the instructions for CUDA 9.0
# conda install -c pytorch pytorch-nightly torchvision cudatoolkit=9.0

#python setup.py build develop
#unset INSTALL_DIR
#cd ..
#pwd
# running scripts
python train.py
#python tools/train_net.py --config-file "configs/defect_detection.yaml" SOLVER.IMS_PER_BATCH 2 SOLVER.BASE_LR 0.0005 SOLVER.MAX_ITER 60000 SOLVER.STEPS "(30000, 40000)" TEST.IMS_PER_BATCH 1
