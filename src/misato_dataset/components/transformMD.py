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

from .transforms import prot_graph_transform
        
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
  

    