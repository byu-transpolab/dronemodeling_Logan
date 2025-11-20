#!/bin/bash

#SBATCH --time=02:0:00   # walltime
#SBATCH --ntasks=10   # number of processor cores (i.e. tasks)
#SBATCH --gpus=2
#SBATCH --mem-per-cpu=102400M   # memory per CPU core
#SBATCH -J "aerial3"   # job name
#SBATCH --mail-user=willicon@byu.edu   # email address
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END

source ~/miniconda3/bin/activate yolomodel_v_1
python /home/willicon/training/ultralytics_model/train.py