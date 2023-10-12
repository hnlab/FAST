- Environment
    - see `README.md` in `env/`
- Train PLANet(BindingNet) dataset
    - complex for example, need to change the path of input files and output files.
    - input file
        - `bindingnet_database/from_chembl_client/index/For_ML/BindingNet_Uw_final_median.csv`
        - structure files of PLANet (2fvd_target_CHEMBL301_compound_CHEMBL213713 for example)
            - `bindingnet_database/from_chembl_client/2fvd/target_CHEMBL301/CHEMBL213713/2fvd_CHEMBL301_CHEMBL213713.sdf`
            - `bindingnet_database/from_chembl_client/2fvd/target_CHEMBL301/CHEMBL213713/rec_addcharge_pocket_6A.mol2`
    1. featurize
        - `bash 1-featurize/PLANet/complex_6A/featurize_Uw_cpx.sh`
            - need to change the `target_pdb_dir` in `1-featurize/featurize_split_data_PLANet_true_lig_alone_diff_split.py`
    2. training
        - Note that this script is for SGE queueing system, you need to change it for slurm system.
        - `qsub 2-train/PLANet/complex_6A/train_Uw_cpx_1.sh`
    3. test
        - `bash 3-test/PLANet/complex_6A/Uw_cpx_test.sh`
