import torch
from transforms import mol_graph_transform_for_qm


ATOMS_KEYS = {8: 0, 16: 1, 6: 2, 7: 3, 1: 4, 15: 5, 17: 6, 9: 7, 53: 8, 35: 9, 5: 10, 33: 11, 26: 12, 14: 13, 34: 14, 44: 15, 12: 16, 23: 17, 77: 18, 27: 19, 52: 20, 30: 21, 4: 22, 45: 23}

class GNNTransformQM(object):
    """
    Transform the dict returned by the MolDataset class to a pyTorch Geometric graph
    """

    def __init__(self, atoms_keys=ATOMS_KEYS, use_bonds=False, onehot_edges=False, edge_dist_cutoff=4.5):
        """

        Args:
            atoms_keys (dict, optional): one hot encoding for atoms. Defaults to ATOMS_KEYS.
            use_bonds (bool, optional): use the bond information or neighboring distance. Defaults to False.
            onehot_edges (bool, optional): one hot encoding for different edge types. Defaults to False.
        """
        self.atoms_keys = atoms_keys 
        self.use_bonds = use_bonds
        self.onehot_edges = onehot_edges
        self.edge_dist_cutoff = edge_dist_cutoff
        
    def __call__(self, item):
        item = mol_graph_transform_for_qm(item, 'atoms', 'labels', allowable_atoms=self.atoms_keys, use_bonds=self.use_bonds, onehot_edges=self.onehot_edges, edge_dist_cutoff=self.edge_dist_cutoff)

        graph = item['atoms']   
        graph.x = graph.x.to(torch.float)
        graph.y = item['labels']
        graph.id = item['id']
      
        return graph
    