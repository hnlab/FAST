#!/bin/bash
#$ -S /bin/bash
#$ -N v16_cpx_train_refine
#$ -q ampere
#$ -o /pubhome/xli02/project/PLIM/deep_learning/FAST/fast_plim/PDBbind/pdbbind_v2016/2-train/log/pdbbindv2016_refine_complex_pybel.log
#$ -j y
#$ -r y
#$ -l gpu=1
#$ -notify
#$ -wd /pubhome/xli02/project/PLIM/deep_learning/FAST/fast_plim/PDBbind/pdbbind_v2016

source /usr/bin/startcuda.sh
conda activate fast; 
python 2-train/train_pdbbind.py \
    --train-data datasets/refined_set/complex/PDBbind_train_true_rec.hdf \
    --val-data datasets/refined_set/complex/PDBbind_valid_true_rec.hdf \
    --checkpoint-dir train_results/refine_set/complex \
    --dataset-name PDBbind_v2016 \
    # --batch-size 16 \
    # --checkpoint-iter 100 \
source /usr/bin/end_cuda.sh
