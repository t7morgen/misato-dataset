import numpy as np
import scipy.spatial as ss
import torch
import torch.nn.functional as F
from torch_geometric.utils import to_undirected
from torch_sparse import coalesce

atom_mapping = {0:'H', 1:'C', 2:'N', 3:'O', 4:'F', 5:'P', 6:'S', 7:'CL', 8:'BR', 9:'I', 10: 'UNK'}
residue_mapping = {0:'ALA', 1:'ARG', 2:'ASN', 3:'ASP', 4:'CYS', 5:'CYX', 6:'GLN', 7:'GLU', 8:'GLY', 9:'HIE', 10:'ILE', 11:'LEU', 12:'LYS', 13:'MET', 14:'PHE', 15:'PRO', 16:'SER', 17:'THR', 18:'TRP', 19:'TYR', 20:'VAL', 21:'UNK'}
        
ligand_atoms_mapping = {8: 0, 16: 1, 6: 2, 7: 3, 1: 4, 15: 5, 17: 6, 9: 7, 53: 8, 35: 9, 5: 10, 33: 11, 26: 12, 14: 13, 34: 14, 44: 15, 12: 16, 23: 17, 77: 18, 27: 19, 52: 20, 30: 21, 4: 22, 45: 23}


def prot_df_to_graph(item, df, edge_dist_cutoff, feat_col='element'):
    r"""
    Converts protein in dataframe representation to a graph compatible with Pytorch-Geometric, where each node is an atom.

    :param df: Protein structure in dataframe format.
    :type df: pandas.DataFrame
    :param node_col: Column of dataframe to find node feature values. For example, for atoms use ``feat_col="element"`` and for residues use ``feat_col="resname"``
    :type node_col: str, optional
    :param allowable_feats: List containing all possible values of node type, to be converted into 1-hot node features. 
        Any elements in ``feat_col`` that are not found in ``allowable_feats`` will be added to an appended "unknown" bin (see :func:`atom3d.util.graph.one_of_k_encoding_unk`).
    :type allowable_feats: list, optional
    :param edge_dist_cutoff: Maximum distance cutoff (in Angstroms) to define an edge between two atoms, defaults to 4.5.
    :type edge_dist_cutoff: float, optional

    :return: tuple containing

        - node_feats (torch.FloatTensor): Features for each node, one-hot encoded by values in ``allowable_feats``.

        - edges (torch.LongTensor): Edges in COO format

        - edge_weights (torch.LongTensor): Edge weights, defined as a function of distance between atoms given by :math:`w_{i,j} = \frac{1}{d(i,j)}`, where :math:`d(i, j)` is the Euclidean distance between node :math:`i` and node :math:`j`.

        - node_pos (torch.FloatTensor): x-y-z coordinates of each node
    :rtype: Tuple
    """ 

    allowable_feats = atom_mapping
    
    try : 
        node_pos = torch.FloatTensor(df[['x', 'y', 'z']].to_numpy())
        kd_tree = ss.KDTree(node_pos)
        edge_tuples = list(kd_tree.query_pairs(edge_dist_cutoff))
        edges = torch.LongTensor(edge_tuples).t().contiguous()
        edges = to_undirected(edges)
    except:
        print(f"Problem with PDB Id is {item['id']}")
        
    node_feats = torch.FloatTensor([one_of_k_encoding_unk_indices(e-1, allowable_feats) for e in df[feat_col]])
    edge_weights = torch.FloatTensor(
        [1.0 / (np.linalg.norm(node_pos[i] - node_pos[j]) + 1e-5) for i, j in edges.t()]).view(-1)

    
    return node_feats, edges, edge_weights, node_pos


def mol_df_to_graph_for_qm(df, bonds=None, allowable_atoms=None, edge_dist_cutoff=4.5, onehot_edges=True):
    """
    Converts molecule in dataframe to a graph compatible with Pytorch-Geometric
    :param df: Molecule structure in dataframe format
    :type mol: pandas.DataFrame
    :param bonds: Molecule structure in dataframe format
    :type bonds: pandas.DataFrame
    :param allowable_atoms: List containing allowable atom types
    :type allowable_atoms: list[str], optional
    :return: Tuple containing \n
        - node_feats (torch.FloatTensor): Features for each node, one-hot encoded by atom type in ``allowable_atoms``.
        - edge_index (torch.LongTensor): Edges from chemical bond graph in COO format.
        - edge_feats (torch.FloatTensor): Edge features given by bond type. Single = 1.0, Double = 2.0, Triple = 3.0, Aromatic = 1.5.
        - node_pos (torch.FloatTensor): x-y-z coordinates of each node.
    """
    if allowable_atoms is None:
        allowable_atoms = ligand_atoms_mapping
    node_pos = torch.FloatTensor(df[['x', 'y', 'z']].to_numpy())
    
    if bonds is not None:
        N = df.shape[0]
        bond_mapping = {1.0: 0, 2.0: 1, 3.0: 2, 1.5: 3}
        bond_data = torch.FloatTensor(bonds)
        edge_tuples = torch.cat((bond_data[:, :2], torch.flip(bond_data[:, :2], dims=(1,))), dim=0)
        edge_index = edge_tuples.t().long().contiguous()
        
        if onehot_edges:
            bond_idx = list(map(lambda x: bond_mapping[x], bond_data[:,-1].tolist())) + list(map(lambda x: bond_mapping[x], bond_data[:,-1].tolist()))
            edge_attr = F.one_hot(torch.tensor(bond_idx), num_classes=4).to(torch.float)
            edge_index, edge_attr = coalesce(edge_index, edge_attr, N, N)
  
        else:
            edge_attr = torch.cat((torch.FloatTensor(bond_data[:,-1]).view(-1), torch.FloatTensor(bond_data[:,-1]).view(-1)), dim=0)
    else:
        kd_tree = ss.KDTree(node_pos)
        edge_tuples = list(kd_tree.query_pairs(edge_dist_cutoff))
        edge_index = torch.LongTensor(edge_tuples).t().contiguous()
        edge_index = to_undirected(edge_index)
        edge_attr = torch.FloatTensor([1.0 / (np.linalg.norm(node_pos[i] - node_pos[j]) + 1e-5) for i, j in edge_index.t()]).view(-1)
        edge_attr = edge_attr.unsqueeze(1)
    
    node_feats = torch.FloatTensor([one_of_k_encoding_unk_indices_qm(e, allowable_atoms) for e in df['element']])    
    
    return node_feats, edge_index, edge_attr, node_pos


def one_of_k_encoding_unk_indices(x, allowable_set):
    """Converts input to 1-hot encoding given a set of allowable values. Additionally maps inputs not in the allowable set to the last element."""
    one_hot_encoding = [0] * len(allowable_set)
    if x in allowable_set:
        one_hot_encoding[x] = 1
    else:
        one_hot_encoding[-1] = 1
    return one_hot_encoding

def one_of_k_encoding_unk_indices_qm(x, allowable_set):
    """Converts input to 1-hot encoding given a set of allowable values. Additionally maps inputs not in the allowable set to the last element."""
    one_hot_encoding = [0] * (len(allowable_set)+1)
    if x in allowable_set:
        one_hot_encoding[allowable_set[x]] = 1
    else:
        one_hot_encoding[-1] = 1
    return one_hot_encoding