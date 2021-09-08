# PDBbind_v2019 for example
## 0. add charge for receptor and extract pocket 6A
```bash
cd 0-rec_add_charge_extract_pocket_6A
pdbbind_v19_general_all_dir=/pubhome/xli02/Downloads/dataset/PDBbind/PDBbind_v2019/general_structure_only
ls $pdbbind_v19_general_all_dir/*/*_protein.pdb | \
    parallel -k \
    'bash add_charge_AND_extract_pocket_6A.sh {} >> add_charge.log'
cd ..
```
## 1. featurize
- complex for example
```bash
mkdir dataset/complex
cd 1-featurize
conda activate fast
python featurize_PDBbind_v19_original.py \
    --dataset-name PDBbind_v2019_original \
    --input /pubhome/xli02/Downloads/dataset/PDBbind/PDBbind_v2019/general_structure_only \
    --output dataset/complex/ \
    --train_data train.csv \
    --valid_data valid.csv \
    --test_data test.csv \
    --rec true_rec
cd ..
```
## 2. train
- complex for example
```bash
mkdir train_result
#modify output log and current workdir in 2-train/complex/train.sh
qsub 2-train/complex/train.sh
cd ..
```
- **attention: `best_checkpoint.pth` is not the best model, why?**
## 3. test
- complex for example
```bash
mkdir test_result
cd 3-test
mkdir log
#valid
python test_density.py \
    --checkpoint train_result/complex/model-epoch-xx-step-xx.pth \
    --preprocessing-type raw \
    --feature-type pybel \
    --dataset-name PDBbind_v2019_original \
    --output ../test_result/complex/ \
    --subset-name valid \
    --test-data ../dataset/complex/PDBbind_v19_original_valid_true_rec.hdf \
    --title "PDBbind_v2019_cpx_validation" \
    &> log/PDBbind_v2019_valid_complex.log

#train
python test_density.py \
    --checkpoint train_result/complex/model-epoch-xx-step-xx.pth \
    --preprocessing-type raw \
    --feature-type pybel \
    --dataset-name PDBbind_v2019_original \
    --output ../test_result/complex/ \
    --subset-name train \
    --test-data ../dataset/complex/PDBbind_v19_original_train_true_rec.hdf \
    --title "PDBbind_v2019_cpx_training" \
    &> log/PDBbind_v2019_train_complex.log

#test
python test_density.py \
    --checkpoint train_result/complex/model-epoch-xx-step-xx.pth \
    --preprocessing-type raw \
    --feature-type pybel \
    --dataset-name PDBbind_v2019_original \
    --output ../test_result/complex/ \
    --subset-name test \
    --test-data ../dataset/complex/PDBbind_v19_original_test_true_rec.hdf \
    --title "PDBbind_v2019_cpx_testing" \
    &> log/PDBbind_v2019_test_complex.log
```