#!/bin/bash

#SBATCH --nodes=1
#SBATCH --cpus-per-task=24                  # 1 node
#SBATCH --ntasks-per-node=1         # 36 tasks per node
#SBATCH --time=10-00:00:0              # time limits: 500 hour
#SBATCH --partition=amdgpuextralong	  # gpufast
#SBATCH --gres=gpu:1
#SBATCH --mem=60G
#SBATCH --output=/home/gebreawe/Code/Segmentation/T-UDA/logs/train_uda_nuscenes_kitti_T1_1_S0_0_time_%j.log
# module

cd ../..

ml torchsparse/1.4.0-foss-2021a-CUDA-11.3.1

python train_uda.py configs/data_config/da_kitti_nuscenes/uda_nuscenes_kitti.yaml --distributed False --ssl False


