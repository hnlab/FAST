import os
from tfbio.data import Featurizer
import numpy as np
import h5py
import argparse
# import pybel
from openbabel import pybel
import warnings
#from data_generator.atomfeat_util import read_pdb, rdkit_atom_features, rdkit_atom_coords
#from data_generator.chem_info import g_atom_vdw_ligand, g_atom_vdw_protein
import xml.etree.ElementTree as ET
from rdkit.Chem.rdmolfiles import MolFromMol2File
import rdkit
from rdkit import Chem
from rdkit.Chem import rdchem
# from pybel import Atom
import pandas as pd
from tqdm import tqdm

ob_log_handler = pybel.ob.OBMessageHandler()
ob_log_handler.SetOutputLevel(0)

# TODO: compute rdkit features and store them in the output hdf5 file
# TODO: instead of making a file for each split, squash into one?

parser = argparse.ArgumentParser()
parser.add_argument("--input", default="/pubhome/xli02/Downloads/PDBbind_v2019/general_structure_only")
parser.add_argument("--output", default="/pubhome/xli02/project/PLIM/deep_learning/FAST/fast_plim/PDBbind/pdbbind_dataset")
parser.add_argument("--metadata", default="/pubhome/xli02/Downloads/PDBbind_v2019/INDEX_general_PL_data_grepped.2019")
parser.add_argument("--dataset-name", default="PDBbind_v2019")
parser.add_argument("--rec", default="false_rec", choices=['false_rec', 'true_rec'])
args = parser.parse_args()


def parse_element_description(desc_file):
    element_info_dict = {}
    element_info_xml = ET.parse(desc_file)
    for element in element_info_xml.getiterator():
        if "comment" in element.attrib.keys():
            continue
        else:
            element_info_dict[int(element.attrib["number"])] = element.attrib

    return element_info_dict


def parse_mol_vdw(mol, element_dict):
    vdw_list = []

    if isinstance(mol, pybel.Molecule):
        for atom in mol.atoms:
            # NOTE: to be consistent between featurization methods, throw out the hydrogens
            if int(atom.atomicnum) == 1:
                continue
            if int(atom.atomicnum) == 0:
                continue
            else:
                vdw_list.append(float(element_dict[atom.atomicnum]["vdWRadius"]))

    elif isinstance(mol, rdkit.Chem.rdchem.Mol):
        for atom in mol.GetAtoms():
            # NOTE: to be consistent between featurization methods, throw out the hydrogens
            if int(atom.GetAtomicNum()) == 1:
                continue
            else:
                vdw_list.append(float(element_dict[atom.GetAtomicNum()]["vdWRadius"]))
    else:
        raise RuntimeError("must provide a pybel mol or an RDKIT mol")

    return np.asarray(vdw_list)


def featurize_pybel_complex(ligand_mol, pocket_mol, name, dataset_name):

    featurizer = Featurizer()
    charge_idx = featurizer.FEATURE_NAMES.index('partialcharge') 

    # get ligand features
    ligand_coords, ligand_features = featurizer.get_features(ligand_mol, molcode=1)

    if not (ligand_features[:, charge_idx] != 0).any():  # ensures that partial charge on all atoms is non-zero?
        raise RuntimeError("invalid charges for the ligand {} ({} set)".format(name, dataset_name))  

    # get processed pocket features
    pocket_coords, pocket_features = featurizer.get_features(pocket_mol, molcode=-1)
    if not (pocket_features[:, charge_idx] != 0).any():
        raise RuntimeError("invalid charges for the pocket {} ({} set)".format(name, dataset_name))   

    # center the coordinates on the ligand coordinates
    centroid_ligand = ligand_coords.mean(axis=0)
    ligand_coords -= centroid_ligand

    pocket_coords -= centroid_ligand
    data = np.concatenate((np.concatenate((ligand_coords, pocket_coords)), 
                                np.concatenate((ligand_features, pocket_features))), axis=1) 

    return data

def split_data(affinity_df, test_ratio=0.1, valid_ratio=0.1):
    rand_df = affinity_df.sample(frac=1)
    test_idx = int(test_ratio * rand_df.shape[0])
    valid_idx = int((test_ratio + valid_ratio) * rand_df.shape[0])
    # cut_idx = int(valid_ratio * rand_df.shape[0])
    test_df, valid_df, train_df = rand_df.iloc[:test_idx], rand_df.iloc[test_idx:valid_idx], rand_df.iloc[valid_idx:]
    return test_df, valid_df, train_df

def write_hdf(input_df, set_name, failure_dict, element_dict, process_type, dataset_name):
    print("\nfound {} complexes in {} dataset".format(len(input_df), set_name))
    with h5py.File(f'{args.output}/{set_name}_{args.rec}.hdf', 'w') as f:
        for _, row in tqdm(input_df.iterrows(), total=input_df.shape[0]):
            name = row['pdb_id']
            affinity = row['-logKd/Ki']
            candi_lig = f'{args.input}/{name}/{name}_ligand.mol2'
            # candi_lig = f'{args.input}/{name}/{name}_ligand.sdf'
            # rec_pdb = f'{args.input}/{name}/{name}_pocket.pdb'
            rec_mol2 = f'{args.input}/{name}/{name}_pocket.mol2'

            grp = f.create_group(str(name))
            grp.attrs['affinity'] = affinity
            pybel_grp = grp.create_group("pybel")
            processed_grp = pybel_grp.create_group(process_type)
            
            try:
                crystal_ligand = next(pybel.readfile('mol2', f'{candi_lig}')) 
                # crystal_ligand = next(pybel.readfile('sdf', f'{candi_lig}')) 


            # do not add the hydrogens! they were already added in chimera and it would reset the charges
            except:
                error ="no ligand for {} ({} dataset)".format(name, set_name)
                warnings.warn(error)
                failure_dict["name"].append(name), failure_dict["partition"].append("crystal") , failure_dict["set"].append(set_name), failure_dict["error"].append(error) 
                continue

            try:
                if args.rec == 'false_rec':
                    crystal_pocket = next(pybel.readfile('pdb', '/pubhome/xli02/project/PLIM/deep_learning/FAST/fast_plim/complex/false_rec.pdb')) 
                elif args.rec == 'true_rec':
                    # crystal_pocket = next(pybel.readfile('pdb', rec_pdb))
                    crystal_pocket = next(pybel.readfile('mol2', rec_mol2))  

            except:
                error = "no pocket for {} ({} dataset)".format(name, set_name)
                warnings.warn(error)
                failure_dict["name"].append(name), failure_dict["partition"].append("crystal") , failure_dict["set"].append(set_name), failure_dict["error"].append(error)
                continue

            # extract the van der waals radii for the ligand/pocket
            crystal_ligand_vdw = parse_mol_vdw(mol=crystal_ligand, element_dict=element_dict) 
            
            # in some, albeit strange, cases the pocket consists purely of hydrogen, skip over these if that is the case
            if len(crystal_ligand_vdw) < 1:
                error = "{} ligand consists purely of hydrogen, no heavy atoms to featurize".format(name)
                warnings.warn(error) 
                failure_dict["name"].append(name), failure_dict["partition"].append("crystal") , failure_dict["set"].append(set_name), failure_dict["error"].append(error)
                continue

            crystal_pocket_vdw = parse_mol_vdw(mol=crystal_pocket, element_dict=element_dict)
            # in some, albeit strange, cases the pocket consists purely of hydrogen, skip over these if that is the case
            if len(crystal_pocket_vdw) < 1:
                error = "{} pocket consists purely of hydrogen, no heavy atoms to featurize".format(name)
                warnings.warn(error) 
                failure_dict["name"].append(name), failure_dict["partition"].append("crystal") , failure_dict["set"].append(set_name), failure_dict["error"].append(error)
                continue

            crystal_ligand_pocket_vdw = np.concatenate([crystal_ligand_vdw.reshape(-1), crystal_pocket_vdw.reshape(-1)], axis=0)
            try:
                crystal_data = featurize_pybel_complex(ligand_mol=crystal_ligand, pocket_mol=crystal_pocket, name=name, dataset_name=set_name)
            except RuntimeError as error:
                failure_dict["name"].append(name), failure_dict["partition"].append("crystal") , failure_dict["set"].append(set_name), failure_dict["error"].append(error)
                continue
            
            # enforce a constraint that the number of atoms for which we have features is equal to number for which we have VDW radii 
            assert crystal_ligand_pocket_vdw.shape[0] == crystal_data.shape[0]

            # END QUALITY CONTROL: made it past the try/except blocks....now featurize the data and store into the .hdf file 
            crystal_grp = processed_grp.create_group(dataset_name)
            crystal_grp.attrs["van_der_waals"] = crystal_ligand_pocket_vdw 
            crystal_dataset = crystal_grp.create_dataset("data", data=crystal_data, 
                                                shape=crystal_data.shape, dtype='float32', compression='lzf') 
    return failure_dict


def main():
    process_type = 'raw'
    # dataset_name = 'PDBbind_v2019'
    print(f'dataset_name is {args.dataset_name}; process_type is {process_type}; receptor is {args.rec}')
    affinity_data = pd.read_csv(args.metadata, sep = ' ', header=None, names=['pdb_id', '-logKd/Ki'])

    affi_df = affinity_data.copy()
    for i, pdbid in enumerate(affinity_data['pdb_id']):
        lig_dir = f'{args.input}/{pdbid}'
        if not os.path.exists(lig_dir):
            print(f'Structure file of {pdbid} do not exsit, drop it.')
            affi_df = affi_df.drop(i)
    print(f'Finally found {len(affi_df)} pdbids with structures.')

    element_dict = parse_element_description("/pubhome/xli02/project/PLIM/deep_learning/FAST/fast_plim/PDBbind/pdbbind_v2016/1-featurize/elements.xml")
    failure_dict = {"name": [], "partition": [], "set": [], "error": []}
    test_df, valid_df, train_df = split_data(affi_df)

    if not os.path.exists(args.output):
        os.makedirs(args.output) 

    train_name = 'PDBbind_train'
    failure_dict = write_hdf(train_df, train_name, failure_dict, element_dict, process_type, args.dataset_name)
    valid_name = 'PDBbind_valid'
    failure_dict = write_hdf(valid_df, valid_name, failure_dict, element_dict, process_type, args.dataset_name)
    test_name = 'PDBbind_test'
    failure_dict = write_hdf(test_df, test_name, failure_dict, element_dict, process_type, args.dataset_name)


    failure_df = pd.DataFrame(failure_dict)
    failure_df.to_csv("{}/failure_summary.csv".format(args.output), index=False)

if __name__ == "__main__":
    main()
