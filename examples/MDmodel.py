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
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GNN_MD(torch.nn.Module):
    def __init__(self, num_features, hidden_dim):
        super(GNN_MD, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim*2)
        self.bn2 = nn.BatchNorm1d(hidden_dim*2)
        self.conv3 = GCNConv(hidden_dim*2, hidden_dim*4)
        self.bn3 = nn.BatchNorm1d(hidden_dim*4)
        self.conv4 = GCNConv(hidden_dim*4, hidden_dim*4)
        self.bn4 = nn.BatchNorm1d(hidden_dim*4)
        self.conv5 = GCNConv(hidden_dim*4, hidden_dim*8)
        self.bn5 = nn.BatchNorm1d(hidden_dim*8)
        self.fc1 = nn.Linear(hidden_dim*8, hidden_dim*4)
        self.fc2 = nn.Linear(hidden_dim*4, 1)


    def forward(self, data):
        x = self.conv1(data.x, data.edge_index, data.edge_attr.view(-1))
        x = F.relu(x)
        x = self.bn1(x)
        x = self.conv2(x, data.edge_index, data.edge_attr.view(-1))
        x = F.relu(x)
        x = self.bn2(x)
        x = self.conv3(x, data.edge_index, data.edge_attr.view(-1))
        x = F.relu(x)
        x = self.bn3(x)
        x = self.conv4(x, data.edge_index, data.edge_attr.view(-1))
        x = self.bn4(x)
        x = F.relu(x)
        x = self.conv5(x, data.edge_index, data.edge_attr.view(-1))
        x = self.bn5(x)
        #x = global_add_pool(x, x.batch)
        x = F.relu(x)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.25)
        return self.fc2(x).view(-1)
