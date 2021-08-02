python test.py \
    --checkpoint ../train_results/refine_set/lig_alone/run2/best_checkpoint.pth \
    --preprocessing-type raw \
    --feature-type pybel \
    --dataset-name PDBbind_v2016 \
    --output ../test_results/refined_set/lig_alone \
    --test-data ../datasets/core/lig_alone/PDBbind_v2016_core_false_rec.hdf \
    --output-file-name test_results_2 &> lig_alone/test_lig_alone_2.log

python test.py \
    --checkpoint ../train_results/refine_set/lig_alone/run3/best_checkpoint.pth \
    --preprocessing-type raw \
    --feature-type pybel \
    --dataset-name PDBbind_v2016 \
    --output ../test_results/refined_set/lig_alone \
    --test-data ../datasets/core/lig_alone/PDBbind_v2016_core_false_rec.hdf \
    --output-file-name test_results_3 &> lig_alone/test_lig_alone_3.log