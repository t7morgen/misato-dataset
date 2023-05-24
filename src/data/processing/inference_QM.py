
import numpy as np
import h5py
import argparse
import pickle
import sys
import os
import urllib.request
import re

atomic_numbers_Map = {1:'H',4:'Be', 5:'B', 6:'C', 7:'N', 8:'O',9:'F', 11:'Na',12:'Mg',14:'Si',15:'P',16:'S',17:'Cl',19:'K',20:'Ca',23:'V',26:'Fe',27:'Co', 29:'Cu',30:'Zn',33:'As',34:'Se', 35:'Br',44:'Ru', 45:'Rh',51:'Sb',52:'Te',53:'I', 75:'Re', 76:'Os', 77:'Ir', 78:'Pt'}
inv_atomic_numbers_Map = {v:k for k,v in atomic_numbers_Map.items()}


def download_pdbfile(pdb):
    if not os.path.isfile(pdb+'.pdb'):
        urllib.request.urlretrieve('https://files.rcsb.org/ligands/download/'+pdb.upper()+'_model.sdf', pdb+'.sdf')

def read_sdf_file(pdbName):
    print('reading', pdbName+'.sdf')
    file = open(pdbName+'.sdf', mode="r")
    content = file.read()
    file.close()
    match = re.search(r"^ {3,}-?\d.*(?:\r?\n {3,}-?\d.*)*", content, re.M)
    datasplit = []
    if match:
        for line in match.group().splitlines():
            datasplit.append([part for part in line.split()][:4])
    return datasplit

def write_h5_info(struct, atoms_type, values, outName):
    with h5py.File(outName, 'w') as oF:
        structgroup = oF.create_group(struct)     
        atomprop_group = structgroup.create_group('atom_properties')
        molprop_group = structgroup.create_group('mol_properties')    
        atomprop_group.create_dataset('atoms_names', data= atoms_type, compression = "gzip", dtype = "i8")
        atomprop_group.create_dataset('atom_properties_values', data= values, compression = "gzip", dtype='f8')
        molprop_group.create_dataset('Electron_Affinity', data= [0], compression = "gzip", dtype = "f8")
        molprop_group.create_dataset('Hardness', data= [0], compression = "gzip", dtype = "f8")


def process_content(content):
    x, y, z, atom_type = [], [], [], []    
    for x_i, y_i, z_i, atom_type_i in content:
        x.append(float(x_i))
        y.append(float(y_i))
        z.append(float(z_i))
        atom_type.append(inv_atomic_numbers_Map[atom_type_i])
    values = np.array([x,y,z]).T
    padded_values = np.pad(values, ((0,0),(0,25)), mode='constant', constant_values=(0))
    return padded_values, atom_type

def setup(args):
    if args.pdbid is None and args.fileName is None:
        sys.exit('Please provide pdb-id or pdb file name')
    if args.fileName == None:
        pdbName = args.pdbid
        if not os.path.isdir(pdbName):
            os.mkdir(pdbName)
        download_pdbfile(pdbName)
    else:
        pdbName = args.fileName.split('.sdf')[0]
        if not os.path.isdir(pdbName):
            os.mkdir(pdbName)
    return pdbName    

def main(args): 
    pdbName = setup(args)
    content = read_sdf_file(pdbName)
    values, atom_types = process_content(content)
    write_h5_info(pdbName, atom_types, values, args.datasetOutName)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-O", "-datasetOutName", required=False, help="Name of dataset in hdf5 format that will be created", default='inference_for_qm.hdf5', type=str)
    parser.add_argument("-p", "--pdbid", required=False, help="ID that will be downloaded from the PDB Ligand", type=str)
    parser.add_argument("-f", "--fileName", required=False, help="Name of the sdf file, e.g. vww.sdf", type=str)
    args = parser.parse_args()
    main(args)








