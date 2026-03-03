import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import SAGEConv

class GNN_old(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        # The first layer aggregates features from neighbors
        self.conv1 = SAGEConv((-1, -1), hidden_channels)
        # The second layer refines the "fraud signature"
        self.conv2 = SAGEConv((-1, -1), hidden_channels)
        # Final linear layer to produce the fraud probability
        self.lin = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        # x and edge_index are now dictionaries (node_type -> tensor)
        # to_hetero handles the magic of applying conv1 to every relation
        x = self.conv1(x, edge_index)
        
        x = {key: F.relu(val) for key, val in x.items()}
        
        x = self.conv2(x, edge_index)
        x = {key: F.relu(val) for key, val in x.items()}
        
        return self.lin(x['transaction']) # We only need the output for transactions

class GNN(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        # Use (-1, -1) to handle different feature dims (340 for trans, 16 for cards)
        self.conv1 = SAGEConv((-1, -1), hidden_channels)
        self.conv2 = SAGEConv((-1, -1), hidden_channels)
        self.dropout = nn.Dropout(p=0.3)
        self.lin = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        # DO NOT use dictionaries or loops here. 
        # to_hetero will apply this logic to every node type automatically.
        x = self.conv1(x, edge_index).relu()
        x = self.dropout(x)
        x = self.conv2(x, edge_index).relu()
        x = self.dropout(x)
        return self.lin(x)
    
