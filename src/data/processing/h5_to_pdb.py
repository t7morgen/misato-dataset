import pickle
import h5py
import argparse
import numpy as np

atomic_numbers_Map = {1:'H', 5:'B', 6:'C', 7:'N', 8:'O', 9:'F',11:'Na',12:'Mg',13:'Al',14:'Si',15:'P',16:'S',17:'Cl',19:'K',20:'Ca',34:'Se',35:'Br',53:'I'}

def get_maps(mapdir):
    residueMap = pickle.load(open(mapdir+'atoms_residue_map.pickle','rb'))
    typeMap = pickle.load(open(mapdir+'atoms_type_map.pickle','rb'))
    nameMap = pickle.load(open(mapdir+'atoms_name_map_for_pdb.pickle','rb'))
    return residueMap, typeMap, nameMap

def get_entries(struct, f, frame):
    trajectory_coordinates = f.get(struct+'/'+'trajectory_coordinates')[frame]
    atoms_type = f.get(struct+'/'+'atoms_type')    
    atoms_number = f.get(struct+'/'+'atoms_number') 
    atoms_residue = f.get(struct+'/'+'atoms_residue') 
    molecules_begin_atom_index = f.get(struct+'/'+'molecules_begin_atom_index') 
    return trajectory_coordinates,atoms_type,atoms_number,atoms_residue,molecules_begin_atom_index

def get_entries_QM(struct, f):
    x = f.get(struct+'/atom_properties/atom_properties_values/')[:,0]
    y = f.get(struct+'/atom_properties/atom_properties_values/')[:,1]
    z = f.get(struct+'/atom_properties/atom_properties_values/')[:,2]
    xyz = np.array([x,y,z]).T
    atoms_number = f.get(struct+'/'+'/atom_properties/atom_names')[:]  
    return xyz, atoms_number


def get_atom_name(i, atoms_number, residue_atom_index, residue_name, type_string, nameMap):
    if residue_name == 'MOL':
        try:
            atom_name = atomic_numbers_Map[atoms_number[i]]+str(residue_atom_index)
        except KeyError:
            #print('KeyError', (residue_name, residue_atom_index-1, type_string))
            atom_name = atomic_numbers_Map[atoms_number[i]]+str(residue_atom_index)
    else:
        try:
            atom_name = nameMap[(residue_name, residue_atom_index-1, type_string)]
        except KeyError:
            #print('KeyError', (residue_name, residue_atom_index-1, type_string))
            atom_name = atomic_numbers_Map[atoms_number[i]]+str(residue_atom_index)
    return atom_name

def update_residue_indices(residue_number, i, type_string, atoms_type, atoms_residue, residue_name, residue_atom_index,residue_Map, typeMap):
    """
    If the atom sequence has O-N icnrease the residueNumber
    """
    if i < len(atoms_type)-1:
        if type_string == 'O' and typeMap[atoms_type[i+1]] == 'N' or residue_Map[atoms_residue[i+1]]=='MOL':
            # GLN has a O N sequence within the AA
            if not ((residue_name == 'GLN' and residue_atom_index==12) or (residue_name == 'ASN' and residue_atom_index==9)):
                residue_number +=1
                residue_atom_index = 0
    return residue_number, residue_atom_index

def insert_TERS(i, molecules_begin_atom_index, residue_number, residue_atom_index, lines):
    """
    We have to insert TERs for the endings of the molecule
    """
    if i+1 in molecules_begin_atom_index:
        lines.append('TER')
        residue_number +=1
        residue_atom_index = 0
    return residue_number, residue_atom_index, lines

def create_pdb_lines_MD(trajectory_coordinates, atoms_type, atoms_number, atoms_residue, molecules_begin_atom_index, typeMap,residue_Map, nameMap):
    """
    We go through each atom line and bring the inputs in the pdb format
    
    """
    lines = []
    residue_number = 1
    residue_atom_index = 0
    for i in range(len(atoms_type)):
        residue_atom_index +=1
        type_string = typeMap[atoms_type[i]]
        residue_name = residue_Map[atoms_residue[i]]
        atom_name = get_atom_name(i, atoms_number, residue_atom_index, residue_name, type_string, nameMap)
        x,y,z = trajectory_coordinates[i][0],trajectory_coordinates[i][1],trajectory_coordinates[i][2]
        line = 'ATOM{0:7d}  {1:<4}{2:<4}{3:>5}    {4:8.3f}{5:8.3f}{6:8.3f}  1.00  0.00           {7:<5}'.format(i+1,atom_name,residue_name,residue_number,x,y,z,atomic_numbers_Map[atoms_number[i]])
        residue_number, residue_atom_index = update_residue_indices(residue_number, i, type_string, atoms_type, atoms_residue, residue_name, residue_atom_index,residue_Map, typeMap)
        lines.append(line)
        residue_number, residue_atom_index, lines = insert_TERS(i, molecules_begin_atom_index, residue_number, residue_atom_index, lines)
    return lines

def create_pdb_lines_QM(trajectory_coordinates, atoms_number, nameMap):
    """
    We go through each atom line and bring the inputs in the pdb format
    
    """
    lines = []
    residue_number = 1
    residue_atom_index = 0
    for i in range(len(trajectory_coordinates[:])):
        residue_atom_index +=1
        x,y,z = trajectory_coordinates[i][0],trajectory_coordinates[i][1],trajectory_coordinates[i][2]
        line = 'ATOM{0:7d}  {1:<4}{2:<4}{3:>5}    {4:8.3f}{5:8.3f}{6:8.3f}  1.00  0.00           {7:<5}'.format(i+1,atomic_numbers_Map[int(atoms_number[i])]+str(i), 'MOL',residue_number,x,y,z,atomic_numbers_Map[int(atoms_number[i])])
        lines.append(line)
    return lines

def write_pdb(struct, specification, lines):
    with open(struct+specification+'.pdb', 'w') as of:
        for line in lines:
            of.write(line+'\n')



if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--struct", required=True, help="pdb code of struct to convert e.g. 11gs")
    parser.add_argument("-f", "--frame", required=False, help="Frame of trajectory to convert", default=0, type=int)
    parser.add_argument("-dMD", "--datasetMD", required=False, help="MD dataset in hdf5 format, e.g. MD_dataset_mapped.hdf5", type=str)
    parser.add_argument("-dQM", "--datasetQM", required=False, help="QM dataset in hdf5 format",  type=str)    
    parser.add_argument("-mdir", "--mapdir", required=False, help="Path to maps", default='Maps/', type=str)
    args = parser.parse_args()

    struct = args.struct
    residue_Map, typeMap, nameMap = get_maps(args.mapdir)
    if args.datasetMD is not None:
        f = h5py.File(args.datasetMD, 'r')
        frame = args.frame
        trajectory_coordinates, atoms_type, atoms_number, atoms_residue, molecules_begin_atom_index = get_entries(struct, f, frame)
        print('Generating pdb for MD dataset for '+struct+' frame '+str(args.frame))
        lines = create_pdb_lines_MD(trajectory_coordinates, atoms_type, atoms_number, atoms_residue, molecules_begin_atom_index, typeMap,residue_Map, nameMap)
        write_pdb(struct, '_MD_frame'+str(frame), lines)
    if args.datasetQM is not None:
        print('Generating pdb for QM dataset for '+struct)
        f = h5py.File(args.datasetQM, 'r')
        coordinates, atoms_number = get_entries_QM(struct, f)
        print(coordinates, atoms_number)
        lines = create_pdb_lines_QM(coordinates, atoms_number, nameMap)
        write_pdb(struct, '_qm', lines)

    if args.datasetQM is None and args.datasetMD is None:
        print('Please provide either a MD or a QM dataset name!')





