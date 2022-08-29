import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
from torch.nn import Linear, Parameter
from torch_geometric.nn import SAGEConv, GATConv, GCNConv, GATv2Conv

class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, drop_prob=0):
        super(MLP, self).__init__()
        self.lin_1 = glorot_init(in_dim, hidden_dim)
        self.lin_2 = glorot_init(hidden_dim, out_dim)
        self.drop_prob = drop_prob

    def forward(self, data):
        x = data.x
        x = F.dropout(x, p=self.drop_prob, training=self.training)
        x = F.relu(torch.matmul(x, self.lin_1))
        x = F.dropout(x, p=self.drop_prob, training=self.training)
        x = torch.matmul(x, self.lin_2)

        return x

class SAGENet(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, drop_prob=0, bias=True):
        super(SAGENet, self).__init__()
        self.drop_prob = drop_prob
        self.conv1 = SAGEConv(in_dim, hidden_dim, bias=bias)
        self.conv2 = SAGEConv(hidden_dim, out_dim, bias=bias)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.drop_prob, training=self.training)
        x = self.conv2(x, edge_index)

        return x

class GATNet(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, heads=8, drop_edge=0, drop_prob=0, bias=True):
        super(GATNet, self).__init__()
        self.drop_prob = drop_prob
        self.conv1 = GATConv(in_dim, hidden_dim, heads, dropout=drop_edge, bias=bias)
        self.conv2 = GATConv(hidden_dim*heads, out_dim, heads, concat=False, dropout=drop_edge, bias=bias)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.dropout(x, p=self.drop_prob, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.drop_prob, training=self.training)
        x = self.conv2(x, edge_index)

        return x


class GCNNet(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, drop_prob=0, bias=True):
        super(GCNNet, self).__init__()
        self.drop_prob = drop_prob
        self.conv1 = GCNConv(in_dim, hidden_dim, bias=bias)
        self.conv2 = GCNConv(hidden_dim, out_dim, bias=bias)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.drop_prob, training=self.training)
        x = self.conv2(x, edge_index)

        return x



class GATv2Net(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, heads=8, drop_edge=0, share_weights=False, drop_prob=0, bias=True):
        super(GATv2Net, self).__init__()
        self.drop_prob = drop_prob
        self.conv1 = GATv2Conv(in_dim, hidden_dim, heads, share_weights=share_weights, dropout=drop_edge, bias=bias)

        self.conv2 = GATv2Conv(hidden_dim*heads, out_dim, heads, share_weights=share_weights, concat=False, dropout=drop_edge, bias=bias)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.dropout(x, p=self.drop_prob, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.drop_prob, training=self.training)
        x = self.conv2(x, edge_index)

        return x


def glorot_init(input_dim, output_dim):
    init_range = np.sqrt(6.0 / (input_dim + output_dim))
    initial = torch.rand(input_dim, output_dim) * 2 * init_range - init_range
    return Parameter(initial)

def glorot_init_2(input_dim, output_dim):
    init_range = np.sqrt(6.0 / (input_dim + output_dim))
    initial = torch.rand(input_dim, output_dim) * 2 * init_range - init_range
    return initial