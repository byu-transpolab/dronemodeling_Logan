#!/bin/bash
#SBATCH --job-name=aerialAImodelTEST
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --partition=cs
#SBATCH --qos=test            
#SBATCH --mail-user=willicon@byu.edu
#SBATCH --mail-type=END,FAIL

export OMP_NUM_THREADS=$SLURM_CPUS_ON_NODE
source ~/miniconda3/bin/activate yolomodel_v_1
python /home/willicon/training/ultralytics_model/train.py

