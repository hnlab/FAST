#!/bin/bash
#$ -S /bin/bash
#$ -N v2016_lig_train_refined
#$ -q ampere
#$ -o /pubhome/xli02/project/PLIM/deep_learning/FAST/fast_plim/PDBbind/pdbbind_v2016/2-train/log/pdbbindv2016_refine_lig_alone_pybel.log
#$ -j y
#$ -r y
#$ -l gpu=1
#$ -notify
#$ -wd /pubhome/xli02/project/PLIM/deep_learning/FAST/fast_plim/PDBbind/pdbbind_v2016

source /usr/bin/startcuda.sh
conda activate fast; 
python 2-train/train_pdbbind.py \
    --train-data datasets/refined_set/lig_alone/PDBbind_train_false_rec.hdf \
    --val-data datasets/refined_set/lig_alone/PDBbind_valid_false_rec.hdf \
    --checkpoint-dir train_results/refine_set/lig_alone \
    --dataset-name PDBbind_v2016
source /usr/bin/end_cuda.sh
