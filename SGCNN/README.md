# SGCNN
## 1. For ligand alone in PLIM
- Use benzene as `false_rec.pdb`
### 1.1 featurize and split data into train/validate/test(8:1:1)
- modify `tfbio/data.py`
    - line 3: `import pybel` -> `from openbabel import pybel`
    - line 147: `self.NAMED_PROPS = ['hyb', 'heavyvalence', 'heterovalence', 'partialcharge']` -> `self.NAMED_PROPS = ['hyb', 'heavydegree', 'heterodegree', 'partialcharge']`
```bash
cd sgcnn_lig_alone/1-featureize
export HDF5_USE_FILE_LOCKING='FALSE'
python featurize_split_data.py &> featurize_split_data.log
```
### 1.2 train & validate
- GPU is required
```bash
cd sgcnn_lig_alone/2-train
qsub train.sh
```
