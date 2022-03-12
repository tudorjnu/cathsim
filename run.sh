#!/bin/bash -l

#SBATCH -J cathsim
# Request the number of nodes
#SBATCH -p nodes -N 1 -n 40

# Insert your own username to get e-mail notifications
#SBATCH --mail-user=sgtjianu@liverpool.ac.uk
#SBATCH --mail-type=ALL



echo =========================================================
echo SLURM job: submitted date = `date`
date_start=`date +%s`
echo =========================================================
echo Job output begins
echo -----------------
echo
hostname
# $SLURM_NTASKS is defined automatically as the number of processes in the
# parallel environment.
python main.py
echo
echo ---------------
echo Job output ends
date_end=`date +%s`
seconds=$((date_end-date_start))
minutes=$((seconds/60))
seconds=$((seconds-60*minutes))
hours=$((minutes/60))
minutes=$((minutes-60*hours))
echo =========================================================
echo SLURM job: finished
date = `date`
echo Total run time : $hours Hours $minutes Minutes $seconds Seconds
echo =========================================================
