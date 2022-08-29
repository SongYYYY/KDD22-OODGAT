import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
from torch.nn import Linear, Parameter
from layer import OODGATConv

class OODGAT(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, heads=4, adjust=True, drop_edge=0, add_self_loop=True, drop_prob=0, bias=True, drop_input=0):
        super(OODGAT, self).__init__()

        self.conv1 = OODGATConv(in_dim, hidden_dim, heads, adjust, True, drop_edge, add_self_loop, bias)
        self.conv2 = OODGATConv(hidden_dim * heads, out_dim, heads, adjust, False, drop_edge, add_self_loop, bias)
        self.drop_prob = drop_prob
        self.drop_input = drop_input

    def forward(self, data, return_attention_weights=False):
        x, edge_index = data.x, data.edge_index
        if not return_attention_weights:
            x = F.dropout(x, p=self.drop_input, training=self.training)
            x = F.elu(self.conv1(x, edge_index, False))
            x = F.dropout(x, p=self.drop_prob, training=self.training)
            x = self.conv2(x, edge_index, False)
            return x
        else:
            attention = []
            x = F.dropout(x, p=self.drop_input, training=self.training)
            x, a = self.conv1(x, edge_index, True)
            attention.append(a)
            x = F.elu(x)
            x = F.dropout(x, p=self.drop_prob, training=self.training)
            x, a = self.conv2(x, edge_index, True)
            attention.append(a)
            return (x, attention)



def glorot_init(input_dim, output_dim):
    init_range = np.sqrt(6.0 / (input_dim + output_dim))
    initial = torch.rand(input_dim, output_dim) * 2 * init_range - init_range
    return Parameter(initial)

def glorot_init_2(input_dim, output_dim):
    init_range = np.sqrt(6.0 / (input_dim + output_dim))
    initial = torch.rand(input_dim, output_dim) * 2 * init_range - init_range
    return initial