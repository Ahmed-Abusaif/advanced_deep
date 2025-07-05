import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv

class GCN(torch.nn.Module):
    def __init__(self, num_features, hidden_units, num_classes, dropout):
        super().__init__()
        self.conv1 = GCNConv(num_features, hidden_units)
        self.conv2 = GCNConv(hidden_units, num_classes)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

class GraphSAGE(torch.nn.Module):
    def __init__(self, num_features, hidden_units, num_classes, dropout):
        super().__init__()
        self.conv1 = SAGEConv(num_features, hidden_units)
        self.conv2 = SAGEConv(hidden_units, num_classes)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)
