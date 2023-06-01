#!/bin/bash -l
# Use the current working directory
#SBATCH -D ./
# Use the current environment for this job
#SBATCH --export=ALL
# Define job name
#SBATCH -J cathsim
# Define a standard output file. When the job is running, %N will be replaced by the name of 
# the first node where the job runs, %j will be replaced by job id number.
#SBATCH -o ./logs/cathsim_%j.out
# Define a standard error file
#SBATCH -e ./logs/cathsim_%j.err
# Request the GPU partition (gpu or gpuc). We don't recommend requesting multiple partitions, as the specifications of the nodes in these partitions are different.
#SBATCH -p gpu
# Request the number of nodes
#SBATCH -N 1
# Request the number of GPUs per node to be used (if more than 1 GPU per node is required, change 1 into Ngpu, where Ngpu=2,3,4)
#SBATCH --gres=gpu:1
# Request the number of CPU cores. (There are 24 CPU cores and 4 GPUs on each GPU node,
# so please request 6*Ngpu CPU cores, i.e., 6 CPU cores for 1 GPU, 12 CPU cores for 2 GPUs, and so on.
# User may request more CPU cores for jobs that need very large CPU memory occasionally, 
# but the number of CPU cores should not be greater than 6*Ngpu+5. Please email hpc-support if you need to do so.)
#SBATCH -n 6
# Set time limit in format a-bb:cc:dd, where a is days, b is hours, c is minutes, and d is seconds.
#SBATCH -t 3-00:00:00
# PLEASE don't set the memory option as we should use the default memory which is based on the number of cores 
#SBATCH --mail-user=sgtjianu@liverpool.ac.uk
#SBATCH --mail-type=ALL

first_arg=$1

# Set your maximum stack size to unlimited
ulimit -s unlimited

export OMP_NUM_THREADS=$SLURM_NTASKS


# Load tensorflow and relevant modules
# module load apps/anaconda3/2021.05
# module load libs/cudnn/8.1.0_cuda11.2
# module load compilers/gcc/7.4.0
# activate environment
conda activate ray

# List all modules
# module list

echo =========================================================   
echo SLURM job: submitted  date = `date`      
date_start=`date +%s`

hostname

echo "CUDA_VISIBLE_DEVICES : $CUDA_VISIBLE_DEVICES"
echo "GPU_DEVICE_ORDINAL   : $GPU_DEVICE_ORDINAL"

echo "Running GPU jobs:"

echo "Running GPU job $first_arg"
python $first_arg
# deactivate the gpu virtual environment
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


