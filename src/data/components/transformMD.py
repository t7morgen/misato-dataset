from transforms import prot_graph_transform
        
class GNNTransformMD(object):
    """
    Transform the dict returned by the ProtDataset class to a pyTorch Geometric graph
    """

    def __init__(self, edge_dist_cutoff=4.5):
        """

        Args:
            edge_dist_cutoff (float, optional): distence between the edges. Defaults to 4.5.
        """
        self.edge_dist_cutoff = edge_dist_cutoff 

    def __call__(self, item):
        item = prot_graph_transform(item, atom_keys=['atoms_protein'], label_key='scores', edge_dist_cutoff=self.edge_dist_cutoff)
        return item['atoms_protein']
  

    