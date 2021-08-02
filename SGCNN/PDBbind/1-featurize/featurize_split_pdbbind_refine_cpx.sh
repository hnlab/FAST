#!/bin/bash
#$ -S /bin/bash
#$ -N v16_refine_cpx_feature
#$ -q cuda
#$ -o /pubhome/xli02/project/PLIM/deep_learning/FAST/fast_plim/PDBbind/pdbbind_v2016/1-featurize/log/pdbbindv2016_refine_complex_pybel.log
#$ -j y
#$ -r y
#$ -l gpu=1
#$ -notify
#$ -wd /pubhome/xli02/project/PLIM/deep_learning/FAST/fast_plim/PDBbind/pdbbind_v2016

conda activate fast; 
python 1-featurize/featurize_split_pdbbind.py \
    --dataset-name PDBbind_v2016 \
    --input /pubhome/xli02/Downloads/dataset/PDBbind/pdbbind_v2016_refined-set \
    --output datasets/refined_set/complex \
    --metadata /pubhome/xli02/Downloads/dataset/PDBbind/pdbbind_v2016_refined-set/index/INDEX_refined_data_grepped.2016 \
    --rec true_rec