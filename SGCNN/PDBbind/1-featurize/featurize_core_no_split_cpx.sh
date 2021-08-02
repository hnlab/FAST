#!/bin/bash
#$ -S /bin/bash
#$ -N core_cpx_2016_featurize
#$ -q cuda
#$ -o /pubhome/xli02/project/PLIM/deep_learning/FAST/fast_plim/PDBbind/pdbbind_v2016/1-featurize/log/pdbbindv2016_core_complex_pybel.log
#$ -j y
#$ -r y
#$ -l gpu=1
#$ -notify
#$ -wd /pubhome/xli02/project/PLIM/deep_learning/FAST/fast_plim/PDBbind/pdbbind_v2016/

conda activate fast; 
python 1-featurize/featurize_core_no_split.py \
    --dataset-name PDBbind_v2016 \
    --input /pubhome/xli02/Downloads/dataset/PDBbind/CASF-2016/coreset \
    --output datasets/core/complex \
    --metadata /pubhome/xli02/Downloads/dataset/PDBbind/CASF-2016/core_data.2016 \
    --rec true_rec