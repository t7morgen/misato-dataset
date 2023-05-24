"""
Alignment for adapatability calculation was adapted from https://github.com/HHChuang/align2mole
- thanks

Combination of multiple preprocessing steps at once or a combination of Masks is not supported yet, but should be easy to add. 
"""

import pickle
import h5py
import numpy as np
import argparse
import os
import sys


def get_maps(current_dir):
    """
    Loading the mapping of amber AT to index.
    """
    elementMap = pickle.load(open(current_dir+'/Maps/atoms_type_map_generate.pickle', 'rb'))
    return elementMap

def get_entries(struct, f, h5_properties):
    h5_entries = {}
    for h5_property in h5_properties:
        h5_entries[h5_property] = f.get(struct+'/'+h5_property)
    return h5_entries

def get_strip_indices(args, h5_entries, strip_value, strip_feature):
    """
    We generate the indices for stripping for different cases. We also have to adjust the molecule_begin_atom_index.
    We strip the strip_values from the rest of the values. 
    """
    stripping_available = False
    if strip_feature == 'atoms_element':
        stripped_indices = np.where(h5_entries[strip_feature][:]!=strip_value)[0]   
        stripped_molecules_begin_atom_index = []
        for begin_molecule in h5_entries["molecules_begin_atom_index"]:
            mol_subindices = np.where(h5_entries[strip_feature][:begin_molecule]!=strip_value)[0]
            stripped_molecules_begin_atom_index.append(len(mol_subindices))
        stripping_available = True

    if args.Pres_Lat:
        stripped_protein_indices = np.where(h5_entries[strip_feature][:h5_entries["molecules_begin_atom_index"][-1]]!=strip_value)[0]
        ligand_indices = np.where(h5_entries["atoms_residue"][:]==0)[0]
        if len(ligand_indices)==0: # if the ligand is a peptide and not a small molecule
            ligand_indices = np.array([i for i in range(h5_entries["molecules_begin_atom_index"][-1], len(h5_entries["atoms_type"]))])
        stripped_indices = np.append(stripped_protein_indices, ligand_indices)
        stripped_molecules_begin_atom_index = []
        for begin_molecule in h5_entries["molecules_begin_atom_index"]:
            mol_subindices = np.where(h5_entries[strip_feature][:begin_molecule]!=strip_value)[0]
            stripped_molecules_begin_atom_index.append(len(mol_subindices))
        stripping_available = True

    if args.Pocket >0.0:
        cog = centroid(h5_entries["trajectory_coordinates"][0,h5_entries["molecules_begin_atom_index"][-1]:,:])
        stripped_indices, stripped_molecules_begin_atom_index = get_atom_indices_pocket(cog, h5_entries["trajectory_coordinates"], h5_entries["molecules_begin_atom_index"], args.Pocket)
        stripping_available = True    

    if not stripping_available:
        sys.exit("Please give either strip_feature atoms_element or use Pres_Lat or use Pocket. Other features are currently not supported.")
    return stripped_indices, stripped_molecules_begin_atom_index

def get_inverse_strip_indices(h5_entries, strip_value, strip_feature):
    """
    We generate the indices for stripping for different cases. We also have to adjust the molecule_begin_atom_index.
    The same as get_strip_indices, but we keep the strip_values instead of stripping them.
    """
    stripping_available = False
    if strip_feature == 'atoms_element':
        stripped_indices = np.where(h5_entries[strip_feature][:]==strip_value)[0]   
        stripped_molecules_begin_atom_index = []
        for begin_molecule in h5_entries["molecules_begin_atom_index"]:
            mol_subindices = np.where(h5_entries[strip_feature][:begin_molecule]==strip_value)[0]
            stripped_molecules_begin_atom_index.append(len(mol_subindices))
        stripping_available = True

    if args.Pres_Lat:
        stripped_protein_indices = np.where(h5_entries[strip_feature][:h5_entries["molecules_begin_atom_index"][-1]]==strip_value)[0]
        ligand_indices = np.where(h5_entries["atoms_residue"][:]==0)[0]
        if len(ligand_indices)==0: # if the ligand is a peptide and not a small molecule
            ligand_indices = np.array([i for i in range(h5_entries["molecules_begin_atom_index"][-1], len(h5_entries["atoms_type"]))])
        stripped_indices = np.append(stripped_protein_indices, ligand_indices)
        stripped_molecules_begin_atom_index = []
        for begin_molecule in h5_entries["molecules_begin_atom_index"]:
            mol_subindices = np.where(h5_entries[strip_feature][:begin_molecule]==strip_value)[0]
            stripped_molecules_begin_atom_index.append(len(mol_subindices))
        stripping_available = True

    if args.Pocket >0.0:
        sys.exit('Inversion of indices for Pocket currently not supported.')
   
    if not stripping_available:
        sys.exit("Please give either strip_feature atoms_element or use Pres_Lat or use Pocket. Other features are currently not supported.")

    return stripped_indices, stripped_molecules_begin_atom_index

def strip_feature(args, strip_properties, h5_entries, strip_value, strip_feature, inversion=False):
    """
    The different properties are stripped according to the calculated indices.
    """
    stripped_entries = {}
    if not inversion:
        stripped_indices,  stripped_molecules_begin_atom_index = get_strip_indices(args, h5_entries, strip_value, strip_feature)
    if inversion:
        stripped_indices,  stripped_molecules_begin_atom_index = get_inverse_strip_indices(h5_entries, strip_value, strip_feature)
    for strip_property in strip_properties:
        if strip_property.startswith('atoms_'):
            stripped_entries[strip_property] = h5_entries[strip_property][stripped_indices]
        if strip_property.startswith('trajectory_'):
            stripped_entries[strip_property] = h5_entries[strip_property][:,stripped_indices,:]
        if strip_property.startswith('molecules_'):
            stripped_entries[strip_property] = stripped_molecules_begin_atom_index
    return stripped_entries


def write_h5_info(outName, struct, preprocessing_entries, h5_entries):
    """
    Writing features to h5 file. Please beware that the feature name is relevant for correct dtype definiton.
    atoms_ and molecules_ are always i8, frames_, feature_ and trajectory_ f8.
    
    """
    with h5py.File(outName, 'a') as oF:
        subgroup = oF.create_group(struct)
        for preprocessing_property in  preprocessing_entries.keys():
            if preprocessing_property.startswith('atoms_') or preprocessing_property.startswith('molecules_'):
                subgroup.create_dataset(preprocessing_property, data= preprocessing_entries[preprocessing_property], compression = "gzip", dtype='i8')
            if preprocessing_property.startswith('trajectory_') or preprocessing_property.startswith('feature_'):
                subgroup.create_dataset(preprocessing_property, data= preprocessing_entries[preprocessing_property], compression = "gzip", dtype='f8')   
        for h5_property in  h5_entries.keys():
            if h5_property.startswith('frames_'):
                subgroup.create_dataset(h5_property, data= h5_entries[h5_property], compression = "gzip", dtype='f8')

def convert_to_Pres_Lat(stripped_entries, elementMap, strip_value, strip_feature):
    """
    To create a new feature that contains only protein residue specifications (Pres) but all atom types for the ligand (Lat)
    The protein atom elements go until 10, but residue number 1 is ACE which will only appear in peptides, so -1.
    """
    atoms_Pres_Lat_protein = stripped_entries["atoms_residue"][:stripped_entries["molecules_begin_atom_index"][-1]]+9
    atoms_Pres_Lat = []
    for i in range(stripped_entries["molecules_begin_atom_index"][-1],len(stripped_entries["atoms_type"])):
        atoms_Pres_Lat.append(elementMap[stripped_entries["atoms_type"][i]])
    return np.append(atoms_Pres_Lat_protein, np.array(atoms_Pres_Lat))


def get_atom_indices_pocket(cog, trajectory_coordinates, molecules_begin_atoms_index, cutoff):
    """
    Pocket indices are calculated based on a distance criterion.
    """
    protein_coordinates= trajectory_coordinates[0,:molecules_begin_atoms_index[-1], :]
    cog = np.expand_dims(cog, axis=0)
    cog = cog.repeat(np.shape(protein_coordinates)[0], axis=0)
    distance = np.linalg.norm(protein_coordinates-cog, axis=1)
    protein_indices = np.where(distance<cutoff)[0]
    numAtomsMolecule = np.shape(trajectory_coordinates)[1]-molecules_begin_atoms_index[-1]
    molecule_indizes = np.array([i for i in range(len(protein_coordinates), len(protein_coordinates)+numAtomsMolecule)])
    return(np.append(protein_indices, molecule_indizes)), [0,len(protein_indices)]

def align_frame_to_ref(h5_entries, varframe, coord_ref):
    """
    Gets coordinates, translates by centroid and rotates by rotation matrix R
    """
    coord_var = h5_entries["trajectory_coordinates"][varframe]
    trans = centroid(coord_ref)
    coord_var_cen = coord_var - centroid(coord_var)
    coord_ref_cen = coord_ref - centroid(coord_ref)
    R = kabsch(coord_var_cen, coord_ref_cen)
    coord_var_shifted = np.dot(coord_var_cen,R) + trans
    return coord_var_shifted

def rmsd(A, B):
    """
    Not used yet, but might be helpful for some applications.
    """
    Coord = len(A[0])
    NAtom = len(A)
    cum = 0.0
    for i in range(NAtom):
        for j in range(Coord):
            cum += (A[i][j] - B[i][j])**2.0
    return np.sqrt(cum / NAtom)

def centroid(A):
    A = A.mean(axis=0)
    return A

def kabsch(coord_var, coord_ref):
    """
    calculation of Rotation Matrix R
    see SVD  http://en.wikipedia.org/wiki/Kabsch_algorithm
    and  proper/improper rotation, JCC 2004, 25, 1894.
    """
    covar = np.dot(coord_var.T, coord_ref)
    v, s, wt = np.linalg.svd(covar)
    d = (np.linalg.det(v) * np.linalg.det(wt)) < 0.0
    if d: # antialigns of the last singular vector
        s[-1] = -s[-1]
        v[:, -1] = -v[:, -1]
    R = np.dot(v, wt)
    return R

def adaptability(h5_entries):
    ref = h5_entries["trajectory_coordinates"][0]
    NAtom = len(ref)
    dist_to_ref_mat = np.zeros((NAtom,100))
    for ind in range(100):
        aligned = align_frame_to_ref(h5_entries, ind, ref)
        squared_dist = np.sum((ref-aligned)**2, axis=1)
        dist_to_ref_mat[:, ind] = np.sqrt(squared_dist)
    return np.mean(dist_to_ref_mat, axis=1), np.std(dist_to_ref_mat, axis=1), ref 

def main(args):
    h5_properties = ['trajectory_coordinates', 'atoms_type', 'atoms_number','atoms_residue','atoms_element','molecules_begin_atom_index','frames_rmsd_ligand','frames_distance','frames_interaction_energy','frames_bSASA']
    strip_properties = ["atoms_type", "atoms_number", "atoms_residue", "atoms_element", "trajectory_coordinates","molecules_begin_atom_index"]
    current_dir = os.path.dirname(os.path.realpath(__file__))
    if os.path.isfile(args.datasetOut):
        print('Removing existing output file...')
        os.remove(args.datasetOut)
    elementMap = get_maps(current_dir)
    f = h5py.File(args.datasetIn, 'r')
    structs = pickle.load(open(current_dir+'/available_structs.pickle', 'rb'))
    count = 0
    for struct in structs[args.begin:args.end]:
        count += 1
        print(struct, count)
        h5_entries = get_entries(struct, f, h5_properties)
        if not h5_entries["atoms_type"] == None: 
            preprocessing_done = False
            if args.Pres_Lat:
                if count ==1:
                    print('Stripping all but ', args.strip_feature, args.strip_value, ' and adding feature atoms_Pres_Lat')
                preprocessing_entries = strip_feature(args, strip_properties, h5_entries, args.strip_value, args.strip_feature, inversion = True)
                preprocessing_entries['atoms_Pres_Lat'] = convert_to_Pres_Lat(preprocessing_entries, elementMap, args.strip_value, args.strip_feature)
                preprocessing_done = True

            if args.Pocket > 0.0:
                if count ==1:
                    print('Stripping the protein pocket with cutoff', args.Pocket)
                args.strip_feature = 'pocket'
                args.strip_value = args.Pocket
                preprocessing_entries = strip_feature(args, strip_properties, h5_entries, args.strip_value, args.strip_feature) 
                preprocessing_done = True

            if args.Adaptability:
                if count ==1:
                    print('Stripping ', args.strip_feature, args.strip_value, ' and calculating adaptability for the atoms that were not stripped.')                
                preprocessing_entries = strip_feature(args, strip_properties, h5_entries, args.strip_value, args.strip_feature)
                preprocessing_entries["feature_atoms_adaptability"], preprocessing_entries["feature_atoms_adaptability"], preprocessing_entries["atoms_coordinates_ref"] = adaptability(preprocessing_entries)
                preprocessing_done = True

            if not preprocessing_done:
                if count ==1:
                    print('Stripping ', args.strip_feature, args.strip_value)
                preprocessing_entries = strip_feature(strip_properties, h5_entries, args.strip_value, args.strip_feature) 

            write_h5_info(args.datasetOut, struct, preprocessing_entries, h5_entries)


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--datasetIn", required=False, help="MISATO dataset path to read from in hdf5 format.", default='MD_dataset_mapped.hdf5', type=str)
    parser.add_argument("-O", "--datasetOut", required=False, help="Output dataset in hdf5 format. Will be overwritten if it already exists.", default='MD_dataset_mapped_stripped.hdf5', type=str)
    parser.add_argument("-sf", "--strip_feature", required=False, help="Feature that should be stripped, e.g. atoms_element or atoms_type", default='atoms_element', type=str)
    parser.add_argument("-sv", "--strip_value", required=False, help="Value to strip, e.g. if strip_freature= atoms_element; 1 for H. ", default=1, type=int)
    parser.add_argument("-PL", "--Pres_Lat", required=False, help="If set to True this will create a new feature that combines one entry for each protein AA but all ligand entries; e.g. for only ca set strip_feature = atoms_type and strip_value = 14", default=False, type=bool)
    parser.add_argument("-P", "--Pocket", required=False, help="We strip the complex by given distance (in Angstrom) from COG of molecule, use e.g. 15.0. If default value is given (0.0) no pocket stripping will be applied. ", default=0.0, type=float)
    parser.add_argument("-A", "--Adaptability", required=False, help="We calculate the adaptability for each atom. Default behaviour will also strip H atoms, if no stripping should be perfomed set strip_value to -1.", default=False, type=bool)
    parser.add_argument("-b", "--begin", required=False, help="Start index of structures", default=0, type=int)
    parser.add_argument("-e", "--end", required=False, help="End index of structures", default=9999999, type=int)
    args = parser.parse_args()

    main(args)