#!/bin/bash
rec=$1
echo ${rec}
mol2file=${rec%pdb}mol2
echo -e "open $rec \n addh \n addcharge \n write format mol2 0 ${rec%pdb}tmp.mol2 \n stop" | chimera --nogui
sed 's/H\.t3p/H    /' ${rec%pdb}tmp.mol2 | sed 's/O\.t3p/O\.3  /' > $mol2file
rm ${rec%pdb}tmp.mol2
