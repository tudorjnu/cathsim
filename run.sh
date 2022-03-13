#!/bin/bash -l

# Insert your own username to get e-mail notifications
#SBATCH --mail-user=sgtjianu@liverpool.ac.uk
#SBATCH --mail-type=ALL
#SBATCH -t 72:00:00
# Request the number of nodes
#SBATCH -p nodes -N 2 -n 80
#SBATCH --export-ALL

ulimit -s unlimited

export OMP_NUM_THREADS=$SLURM_NTASKS

# Load Modules
module purge
module load apps/anaconda3/2021.05
module load libs/cudnn/8.1.0_cuda11.2
module load compilers/gcc/7.4.0

conda activate cathsim

echo =========================================================   
echo SLURM job: submitted  date = `date`      
date_start=`date +%s`

hostname

echo "CUDA_VISIBLE_DEVICES : $CUDA_VISIBLE_DEVICES"
echo "GPU_DEVICE_ORDINAL   : $GPU_DEVICE_ORDINAL"

echo "Running GPU jobs:"
python train.py --algo ppo --env-name 1 --scene 1 --target bca -o internal 

python train.py --algo ppo --env-name 1 --scene 1 --target lcca -o internal 

python train.py --algo ppo --env-name 1 --scene 2 --target bca -o internal 

python train.py --algo ppo --env-name 1 --scene 2 --target lcca -o internal 

python train.py --algo a2c --env-name 1 --scene 1 --target bca -o internal 

python train.py --algo a2c --env-name 1 --scene 1 --target lcca -o internal 

python train.py --algo  --env-name 1 --scene 2 --target bca -o internal 

python train.py --algo  --env-name 1 --scene 2 --target lcca -o internal 
#deactivate the gpu virtual environment
conda deactivate

date_end=`date +%s`
seconds=$((date_end-date_start))
minutes=$((seconds/60))
seconds=$((seconds-60*minutes))
hours=$((minutes/60))
minutes=$((minutes-60*hours))
echo =========================================================   
echo SLURM job: finished   date = `date`   
echo Total run time : $hours Hours $minutes Minutes $seconds Seconds
echo =========================================================   

conda deactivate
