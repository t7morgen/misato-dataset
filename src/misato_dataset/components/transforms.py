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

import torch
from torch_geometric.data import Data
from .graph import prot_df_to_graph, mol_df_to_graph_for_qm

def prot_graph_transform(item, atom_keys, label_key, edge_dist_cutoff):
    """Transform for converting dataframes to Pytorch Geometric graphs, to be applied when defining a :mod:`Dataset <atom3d.datasets.datasets>`.
    Operates on Dataset items, assumes that the item contains all keys specified in ``keys`` and ``labels`` arguments.

    :param item: Dataset item to transform
    :type item: dict
    :param atom_keys: list of keys to transform, where each key contains a dataframe of atoms, defaults to ['atoms']
    :type atom_keys: list, optional
    :param label_key: name of key containing labels, defaults to ['scores']
    :type label_key: str, optional
    :return: Transformed Dataset item
    :rtype: dict
    """    

    for key in atom_keys:
        node_feats, edge_index, edge_feats, pos = prot_df_to_graph(item, item[key], edge_dist_cutoff)
        item[key] = Data(node_feats, edge_index, edge_feats, y=torch.FloatTensor(item[label_key]), pos=pos, ids=item["id"])
        
    return item

def mol_graph_transform_for_qm(item, atom_key, label_key, allowable_atoms, use_bonds, onehot_edges, edge_dist_cutoff):
    """Transform for converting dataframes to Pytorch Geometric graphs, to be applied when defining a :mod:`Dataset <atom3d.datasets.datasets>`.
    Operates on Dataset items, assumes that the item contains all keys specified in ``keys`` and ``labels`` arguments.

    :param item: Dataset item to transform
    :type item: dict
    :param atom_key: name of key containing molecule structure as a dataframe, defaults to 'atoms'
    :type atom_keys: list, optional
    :param label_key: name of key containing labels, defaults to 'scores'
    :type label_key: str, optional
    :param use_bonds: whether to use molecular bond information for edges instead of distance. Assumes bonds are stored under 'bonds' key, defaults to False
    :type use_bonds: bool, optional
    :return: Transformed Dataset item
    :rtype: dict
    """    

    bonds = item['bonds'] if use_bonds else  None
     
    node_feats, edge_index, edge_feats, pos = mol_df_to_graph_for_qm(item[atom_key], bonds=bonds, onehot_edges=onehot_edges, allowable_atoms=allowable_atoms, edge_dist_cutoff=edge_dist_cutoff)
    item[atom_key] = Data(node_feats, edge_index, edge_feats, y=item[label_key], pos=pos)

    return item
