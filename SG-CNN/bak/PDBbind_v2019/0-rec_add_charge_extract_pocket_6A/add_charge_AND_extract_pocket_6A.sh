#!/bin/bash
rec=$1
echo ${rec}
pdb_dir=${rec%/*}
pdbid=`echo ${rec##*/}|awk -F '_' '{print $1}'`
cry_lig_sdf=${pdb_dir}/${pdbid}_ligand.sdf
mol2file=${pdb_dir}/${pdbid}_pocket_6A.mol2
echo -e "open $rec \n addh \n addcharge \n open $cry_lig_sdf \n sel #1 z<6 \n write format mol2 selected 0 ${mol2file%mol2}tmp.mol2 \n stop" | chimera --nogui
sed 's/H\.t3p/H    /' ${mol2file%mol2}tmp.mol2 | sed 's/O\.t3p/O\.3  /' > $mol2file
# rm ${mol2file%mol2}tmp.mol2
