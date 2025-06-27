'''
From: https://github.com/tkipf/pygcn
'''

import math
import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

@torch.no_grad()
def norm_adj(adj: torch.Tensor, dtype=torch.float32):
    adj = adj + torch.eye(adj.shape[-1]).to(adj.device).to(dtype)
    rowsum = torch.sum(adj, dim=-1)
    degree_mat_inv_sqrt = torch.diag_embed(torch.pow(rowsum, -0.5))
    adj_norm = torch.matmul(torch.matmul(degree_mat_inv_sqrt, adj), degree_mat_inv_sqrt)
    return adj_norm

class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, max_order=2, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.max_order = max_order

        self.weight_C = Parameter(torch.FloatTensor(in_features, out_features))
        self.weight_A_list = torch.nn.ParameterList([Parameter(torch.FloatTensor(in_features, out_features)) for i in range(max_order)])

        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight_C.size(1))
        self.weight_C.data.uniform_(-stdv, stdv)

        for i in range(self.max_order):
            self.weight_A_list[i].data.uniform_(-stdv, stdv)

        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj_list):

        output = torch.matmul(input, self.weight_C)

        for i in range(self.max_order):
            output = output + torch.matmul(adj_list[i], torch.matmul(input, self.weight_A_list[i]))

        if self.bias is not None:
            return output + self.bias
        else:
            return output