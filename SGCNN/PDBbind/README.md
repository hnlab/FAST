# 1. PDBbind_v2019
## 1.1 general set 随机分8:1:1
- `grep -v '#' INDEX_general_PL_data.2019|awk '{print $1,$4}' > INDEX_general_PL_data_grepped.2019`
- ligand alone
    - `python featurize_split_pdbbind.py &> featurize_split_pdbbind.log`
        - `dataset_name`: `'PDBbind_v2019'`
        - cry_lig from mol2 报错比from sdf少
            - `Open Babel Warning`: 3899
                - `Open Babel Warning  in ReadMolecule`: 3891
                    - `Failed to kekulize aromatic bonds in MOL2 file`: 3887
            - failure.csv: 0
        - ~~**考虑rdkit?**~~
            - pybel 的Warning可能影响不大，先默认用pybel
    - `qsub train_pdbbind.sh`
        - best r2: 0.3673034062806845 pearsonr: (0.6064896870365258, 2.6869626685392856e-177) spearmanr: SpearmanrResult(correlation=0.6037814418474441, pvalue=2.543181634695957e-175)
        - loss变化仍然较大 —— **不收敛，加大epoch？**
- complex
    - `python featurize_split_pdbbind.py --rec true_rec &> featurize_split_pdbbind_true_rec.log`
        - `Open Babel Warning`: 17218
            - `Open Babel Warning  in ReadMolecule`: 3891
                - `Failed to kekulize aromatic bonds in MOL2 file`: 3887
            - `Open Babel Warning  in PerceiveBondOrders`: 12194
        - failure.csv: 538
            - `invalid charges for the pocket`
    - `qsub train_pdbbind_complex.sh`
        - `--checkpoint-dir xx`
        - ~~`--batch-size 16`~~ -> **`--batch-size 8`**
        - ~~`--checkpoint-iter 100`~~
        - ~~`--epochs 200`~~
## 2. PDBbind_v2016
- 分别下载refined set & core set
### 2.1 refined set
#### 2.1.1 featurize
- ligand alone
    - `grep -v '#' INDEX_refined_data.2016|awk '{print $1,$4}' > INDEX_refined_data_grepped.2016`
    - Featurize: split into train/validate/test
        - **`--dataset-name PDBbind_v2016`**
        - 先随机分train/valid/test(`np.random.permutation`)，再在每个子集内按活性排序
- complex
    - **pocket.pdb用Chimera加氢加charge**？
        - `ls /pubhome/xli02/Downloads/dataset/PDBbind/pdbbind_v2016_refined-set/*/*_pocket.pdb|parallel -k --joblog add_charge_job.log 'bash add_charge.sh {} >> add_charge.log'`
    - Featurize: split into train/validate/test
        - 先随机分train/valid/test(`np.random.permutation`)，再在每个子集内按活性排序
        - `pocket.pdb` -> **`pocket.mol2`**
        - **`--rec true_rec`**
        - **`--dataset-name PDBbind_v2016`**
#### 2.1.2 train
- `DataListLoader`: 先shuffle，再drop_last；因此，要复现结果，应设置`drop_last=False`
    - `generator`可设置shuffle的random seed
    - `worker_init_fn(worker_id)`可设置`num_workers!=0`时各workers不同的random seed
    - [补充信息：pytorch+numpy](https://tanelp.github.io/posts/a-bug-that-plagues-thousands-of-open-source-ml-projects/)
For trainning set, `shuffle=True/False`,(`drop_last=True`)
For validation set, `drop_last = False`, (`shuffle=False`)
- ligand alone
    - `--dataset-name PDBbind_v2016`
    - `shuffle = False/True`
- complex
    - `--dataset-name PDBbind_v2016`
    - `shuffle = False/True`
### 2.2 core set: as test set?
- `ls * -d > ../core_pdbid.csv`
- `grep -f core_pdbid.csv ../pdbbind_v2016_refined-set/index/INDEX_refined_data_grepped.2016 > core_data.2016`
#### 2.2.1 featurize: no split?
- ligand alone: no split
- complex
    - pocket.pdb用Chimera加氢加charge -> `pocket.mol2`
        - `ls ~/Downloads/dataset/PDBbind/CASF-2016/coreset/*/*_pocket.pdb|parallel -k --joblog add_charge_core_job.log 'bash add_charge.sh {} >> add_charge_core.log'`
    - featurize: no split
        - `pocket.pdb` -> **`pocket.mol2`**
### 2.3 test on core set
```bash
python test.py \
    --checkpoint ../train_results/refine_set/complex/best_checkpoint.pth \
    --preprocessing-type raw \
    --feature-type pybel \
    --dataset-name PDBbind_v2016 \
    --output ../test_results/refined_set/complex \
    --test-data ../datasets/core/complex/PDBbind_v2016_core_true_rec.hdf &> test_complex.log
```

```bash
python test.py \
    --checkpoint ../train_results/refine_set/lig_alone/best_checkpoint.pth \
    --preprocessing-type raw \
    --feature-type pybel \
    --dataset-name PDBbind_v2016 \
    --output ../test_results/refined_set/lig_alone \
    --test-data ../datasets/core/lig_alone/PDBbind_v2016_core_false_rec.hdf &> test_lig_alone.log
```
- `best_checkpoint.pth` 不是最好的model？
- `RuntimeError: CUDA out of memory.`
    - https://segmentfault.com/a/1190000022589080
    - **`with torch.no_grad():`**
    - 版本一致
    - 调小batch_size
    - `nvidia-smi` 查看显存 / 关掉jupyter
## 3. 结果

||r2_valid|mae_valid|mse_valid|pearsonr_valid|spearmanr_valid|r2_test|mae_test|mse_test|pearsonr_test|spearmanr_test|loss|
|--|--|--|--|--|--|--|--|--|--|--|--|
|PDBbind_v2016_refined(ligand alone)|0.392|1.154|2.092|0.626|0.619|0.374|1.384|2.948|0.632|0.626|![total_loss](https://user-images.githubusercontent.com/54713559/126934369-b4582390-93ca-4950-98ba-aa840caf24c4.png) ![loss_y_10](https://user-images.githubusercontent.com/54713559/126934403-361e6c1f-32f1-4f7e-b052-02c81a648da9.png) ![loss_y_10_x_tail1000](https://user-images.githubusercontent.com/54713559/126934432-b37e74cb-ea43-423a-a86c-891093aec6df.png)|
|PDBbind_v2016_refined(complex)|0.487|1.214|2.268|0.698|0.698|0.523|1.199|2.246|0.742|0.751|![loss_total](https://user-images.githubusercontent.com/54713559/126934149-300e3de8-98a5-4bbf-a963-19d0664d2b49.png) ![loss_y_10](https://user-images.githubusercontent.com/54713559/126934310-65230241-3fac-4142-b4e0-f0f99cf293b8.png) ![loss_y_10_x_tail1000](https://user-images.githubusercontent.com/54713559/126934259-aa422e5a-a798-4712-9d21-820e16e054e6.png)|
|PDBbind_v2019_general(ligand alone)|0.367|1.160|2.245|0.606|0.604|||||||
|PDBbind_v2019_general(complex)|0.391|1.149|2.158|0.642|0.649|||||||
|PLIM(ligand alone)|0.185|1.014|1.662|0.434|0.404|||||||
- loss没有收敛，波动较大
    - 该模型的特征仅根据简单加和（不准确）：'The resulting features of the noncovalent propagation are then “gathered” across the ligand nodes in the graph to produce
a flattened vector representation by taking a node-wise summation of the features.'   
    ![Screenshot from 2021-07-26 12-49-02](https://user-images.githubusercontent.com/54713559/126934592-76d03903-25d4-49f9-a002-56682a503b7f.png)
        - 这篇文章的GNN只用sum来gather各节点表征hv来获得图表征hg，模型拟合能力差，训练也不稳定
        - 可以参考这篇GHDDI文章使用多种gather函数（比如max，mean等），虽然水平不高，不过方法应该有用。https://arxiv.org/abs/2102.04064
        - 或者使用更自然的分层pooling（池化），逐渐从多个节点聚合成一个。https://papers.nips.cc/paper/2018/hash/e77dbaf6759253c7c6d0efc5690369c7-Abstract.html
