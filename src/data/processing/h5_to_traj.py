'''MISATO, a database for protein-ligand interactions
    Copyright (C) 2023  
                        Till Siebenmorgen  (till.siebenmorgen@helmholtz-munich.de)
                        Sabrina Benassou   (s.benassou@fz-juelich.de)
                        Filipe Menezes     (filipe.menezes@helmholtz-munich.de)
                        Erinç Merdivan     (erinc.merdivan@helmholtz-munich.de)

    This library is free software; you can redistribute it and/or
    modify it under the terms of the GNU Lesser General Public
    License as published by the Free Software Foundation; either
    version 2.1 of the License, or (at your option) any later version.

    This library is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
    Lesser General Public License for more details.

    You should have received a copy of the GNU Lesser General Public
    License along with this library; if not, write to the Free Software 
    Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA'''


# usage: python src/data/processing/h5_to_traj.py -s 11GS -dMD data/MD/h5_files/tiny_md.hdf5 -oF test_traj.nc



import pickle
import h5py
import argparse
import numpy as np
import pytraj as pt
import os
import sys



def get_entries(struct, f):
    """
    Get the trajectory coordinates from the h5 file.
    Args:
    struct (str): PDB code of the structure
    f (h5py file): h5py file containing the dataset
    
    """
    trajectory_coordinates = f.get(os.path.join(struct,'trajectory_coordinates'))
    return trajectory_coordinates

def open_restart_file(struct, rstPath):
    """
    Open the restart file of the structure.
    Args:
    struct (str): PDB code of the structure
    rstPath (str): Path to the directory of the structures (containing a separate structure folder containing production.rst,production.top files)
    """
    traj = pt.iterload(os.path.join(rstPath,struct,"production.rst"), os.path.join(rstPath,struct,"production.top"))
    return traj

def create_new_traj(traj, h5_coordinates):
    """
    Coordinates from h5 file are appended as frames to stripped traj.  

    Args:
    traj (pytraj trajectory): Trajectory from the restart file
    h5_coordinates (np.array): Coordinates
    
    """
    h5_coordinates = np.expand_dims(h5_coordinates, axis=2)
    for frameNum in range(np.shape(h5_coordinates)[0]):
        frame = pt.Frame()
        for i in range(traj.n_atoms):
            frame.append_xyz(h5_coordinates[frameNum][i])
        traj.append(frame)
    return traj[1:]

def create_topology(topNameNew, file_top, mask):
    """
    A stripped toplogy is written to file.
    Args:
    topNameNew (str): Name of the new topology file
    file_top (str): Original topology file
    mask (str): Mask to strip the topology

    """
    top = pt.load_topology(file_top)
    top = pt.strip(top, "!({})".format(mask))
    top.save(topNameNew)


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--struct", required=True, help="pdb code of struct to convert e.g. 11GS, use capital letters")
    parser.add_argument("-dMD", "--datasetMD", required=False, help="MD dataset in hdf5 format, e.g. MD_dataset_mapped.hdf5", type=str)
    parser.add_argument("-oF", "--outFilename", required=False, help="Name of the outfile. The name specifies the format. Use XXX.nc for netcdf, (XXX.pdb for PDB not recommended because pytraj messes up here). ", type=str)
    parser.add_argument("-r", "--rstPath", required=False, help="Path to the directory of the structures (containing a separate structure folder containing production.rst,production.top files), see data/MD/restart/", default = "data/MD/restart/", type=str)

    args = parser.parse_args()
    struct = args.struct
    traj = open_restart_file(struct.lower(), args.rstPath)
    f = h5py.File(args.datasetMD, 'r')
    h5_coordinates = get_entries(struct, f)
    # This mask strips all waters and ions. 
    mask = "!:WAT,Na+,Cl-"
    new_traj = create_new_traj(traj[mask], h5_coordinates)
    pt.write_traj(args.outFilename, new_traj, overwrite=True)
    create_topology(args.outFilename.split(".")[0]+".top",  os.path.join(args.rstPath,struct.lower(),"production.top"), mask)

