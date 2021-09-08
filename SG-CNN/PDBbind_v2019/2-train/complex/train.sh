#!/bin/bash
#$ -S /bin/bash
#$ -N Po_cpx_shuff_epoch_500_train
#$ -q cuda
#$ -o xxxxxxxxxxx
#$ -j y
#$ -r y
#$ -pe cuda 4
#$ -R y
#$ -notify
#$ -wd xxxxxxxxxxx

# source /usr/bin/startcuda.sh
printf "%s Start on %s.\n" "[$(date +"%Y-%m-%d %H:%M:%S")]" $(hostname)
conda activate fast
export CUDA_VISIBLE_DEVICES=0
python 2-train/train_reduce_valid.py \
    --train-data dataset/complex/PDBbind_v19_original_train_true_rec.hdf \
    --val-data dataset/complex/PDBbind_v19_original_valid_true_rec.hdf \
    --checkpoint-dir train_result/complex/ \
    --dataset-name PDBbind_v2019_original \
    --rec true_rec \
    --shuffle \
    --epochs 500
    # --batch-size 16 \
    # --checkpoint-iter 100 \
printf "%s End on %s.\n" "[$(date +"%Y-%m-%d %H:%M:%S")]" $(hostname)
# source /usr/bin/startcuda.sh
