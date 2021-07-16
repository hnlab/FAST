#!/bin/bash
#$ -S /bin/bash
#$ -N fast_train
#$ -q cuda
#$ -o /pubhome/xli02/project/PLIM/deep_learning/FAST/fast_plim/train.log
#$ -j y
#$ -r y
#$ -l gpu=1
#$ -notify
#$ -wd /pubhome/xli02/project/PLIM/deep_learning/FAST/fast_plim
##$ -now y
##$ -l h="k214.hn.org"

conda activate fast; 
python train.py \
    --train-data dataset/PLIM_train.hdf \
    --val-data dataset/PLIM_valid.hdf