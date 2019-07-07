#!/bin/bash
#SBATCH -J segmentation
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --output=out_%j.log
#SBATCH --error=err_%j.log
#SBATCH -t 100:00:00
#SBATCH --gres=gpu:2
module load anaconda3/5.3.0
python train_seg2019.py --train-root HaN_OAR --save models/ --batchSz 2 --ngpu 1

