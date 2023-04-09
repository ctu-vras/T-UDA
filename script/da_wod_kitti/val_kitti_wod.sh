#!/bin/bash

#SBATCH --nodes=1
#SBATCH --cpus-per-task=1               # 1 node
#SBATCH --ntasks-per-node=1        # 36 tasks per node
#SBATCH --time=4:00:00               # time limits: 500 hour
#SBATCH --partition=amdgpufast	  # gpufast
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --output=/home/gebreawe/Code/Segmentation/T-UDA/logs/val_uda_kitti_wod_T1_1_S0_0_%j.log     # file name for stdout/stderr
# module

cd ../..

ml torchsparse/1.4.0-foss-2021a-CUDA-11.3.1

python evaluate_uda.py configs/data_config/da_wod_kitti/uda_kitti_wod.yaml --network student
