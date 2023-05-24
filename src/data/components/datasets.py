from pathlib import Path
import numpy as np
import pandas as pd
import h5py

import torch
from torch.utils.data import Dataset

class ProtDataset(Dataset):
    """
        Load the MD dataset
    """

    def __init__(self, md_data_file, idx_file, transform=None, post_transform=None):
        """

        Args:
            md_data_file (str): H5 file path
            idx_file (str): path of txt file which contains pdb ids for a specific split such as train, val or test.
            transform (obj): class that convert a dict to a PyTorch Geometric graph.
            post_transform (PyTorch Geometric, optional): data augmentation. Defaults to None.
        """

        self.md_data_file = Path(md_data_file).absolute()

        with open(idx_file, 'r') as f: 
            self.ids = f.read().splitlines()

        self.f = h5py.File(self.md_data_file, 'r') 

        self._transform = transform

        self._post_transform = post_transform

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, index: int):
        if not 0 <= (index) < len(self.ids):
            raise IndexError(index)

        item = {}
            
        column_names = ["x", "y", "z", "element"]
        atoms_protein = pd.DataFrame(columns = column_names)
        print(self.ids[index])
        pitem = self.f[self.ids[index]]

        cutoff = pitem["molecules_begin_atom_index"][:][-1]
      
        atoms_protein["x"] = pitem["atoms_coordinates_ref"][:][:cutoff, 0]
        atoms_protein["y"] = pitem["atoms_coordinates_ref"][:][:cutoff, 1]
        atoms_protein["z"] = pitem["atoms_coordinates_ref"][:][:cutoff, 2]

        atoms_protein["element"] = pitem["atoms_element"][:][:cutoff]  

        item["scores"] = pitem["feature_atoms_adaptability"][:][:cutoff]

        item["atoms_protein"] = atoms_protein

        item["id"] = self.ids[index]
        
        if self._transform:
            item = self._transform(item)

        if self._post_transform:
            item = self._post_transform(item)
    
        return item


class MolDataset(Dataset):
    """
        Load the QM dataset.
    """

    def __init__(self, data_file, idx_file, target_norm_file, transform, isTrain=False, post_transform=None):
        """

        Args:
            data_file (str): H5 file path
            idx_file (str): path of txt file which contains pdb ids for a specific split such as train, val or test.
            target_norm_file (str): H5 file path where training mean and std are stored.  
            transform (obj): class that convert a dict to a PyTorch Geometric graph.
            isTrain (bool, optional): Flag to standardize the target values (only used for train set). Defaults to False.
            post_transform (PyTorch Geometric, optional): data augmentation. Defaults to None.
        """
        
        if type(data_file) is list:
            if len(data_file) != 1:
                raise RuntimeError("Need exactly one filepath for h5")
            data_file = data_file[0]

        self.data_file = Path(data_file).absolute()

        with open(idx_file, 'r') as f: 
            self.ids = f.read().splitlines()

        self.f = h5py.File(self.data_file, 'r') 

        self.target_norm = h5py.File(target_norm_file, 'r') 
        
        self.target_dict = {
            "Electron_Affinity" : 1,
            "Hardness" : 3
        }

        self._transform = transform
        self._post_transform = post_transform
        
        self.isTrain = isTrain

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, index: int):
        if not 0 <= index < len(self.ids):
            raise IndexError(index)

        column_names = ["x", "y", "z", "element"]
        atoms = pd.DataFrame(columns = column_names)

        pitem = self.f[self.ids[index]]
        prop = pitem["atom_properties"]["atom_properties_values"]
        
        atoms["x"] = prop[:,0].astype(np.float32)
        atoms["y"] = prop[:,1].astype(np.float32)
        atoms["z"] = prop[:,2].astype(np.float32)
        
        atoms["element"] = np.array([int(element.decode('utf-8')) for element in pitem["atom_properties/atom_names"][:]])
        
        bonds = pitem["atom_properties/bonds"][:]

        elec_aff = torch.tensor(pitem["mol_properties"]["Electron_Affinity"][()])        
        hardness = torch.tensor(pitem["mol_properties"]["Hardness"][()])
        scores = torch.cat([elec_aff.view(1), hardness.view(1)])
        
        if self.isTrain:
            aff_std = torch.tensor(self.target_norm["Electron_Affinity"]["std"][()])
            h_std = torch.tensor(self.target_norm["Hardness"]["std"][()])

            aff_mean = torch.tensor(self.target_norm["Electron_Affinity"]["mean"][()])
            h_mean = torch.tensor(self.target_norm["Hardness"]["mean"][()])

            all_mean = torch.cat([aff_mean.view(1), h_mean.view(1)])
            all_std = torch.cat([aff_std.view(1), h_std.view(1)])
        else:
            all_mean = 0.0
            all_std = 1.0
            

        item = {"atoms" : atoms,
            "labels": ((scores - all_mean) / all_std).float(),
            "bonds": bonds, 
            "id": self.ids[index]
        }

        if self._transform:
            item = self._transform(item)

        if self._post_transform:
            item = self._post_transform(item)
    

        return item