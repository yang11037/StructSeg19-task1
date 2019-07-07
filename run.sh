#!/bin/bash
#SBATCH -J segmentation
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --output=out.log
#SBATCH --error=err.log
#SBATCH -t 100:00:00
#SBATCH --gres=gpu:4
module load anaconda3/5.3.0
source activate mm
python train_seg2019.py --train-root ../data/HaN_OAR --save ../mode_saved --batchSz 1 --ngpu 1

