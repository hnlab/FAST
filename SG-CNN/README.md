- Environment
    - see `README.md` in `env/`
- Train PLANet(BindingNet) dataset
    - complex for example, need to change the path of input files and output files.
    - input file
        - `PLANet_Uw_final_median.csv`
        - structure files of PLANet
            - `compound.sdf`, `rec_addcharge_pocket_6A.mol2`
    1. featurize
        - `bash 1-featurize/PLANet/complex_6A/featurize_Uw_cpx.sh`
    2. training
        - Note that this script is for SGE queueing system, you need to change it for slurm system.
        - `qsub 2-train/PLANet/complex_6A/train_Uw_cpx_1.sh`
    3. test
        - `bash 3-test/PLANet/complex_6A/Uw_cpx_test.sh`
