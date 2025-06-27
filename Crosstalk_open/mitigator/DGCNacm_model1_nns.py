import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from models_GNNlayers.ACmulti_GCNlayer import GraphConvolution

'''
#######################################################################
############# Graph Convolutional Neural Networks  ####################
#######################################################################
'''

class model1_GCN_single(nn.Module):
    def __init__(self, inputs, hidden_layers, max_order=2, outputs=1):
        super(model1_GCN_single, self).__init__()

        self.max_order = max_order
        
        hidden_layers_over4 = hidden_layers // 4
        hidden_layers_times4 = hidden_layers * 4

        self.gc_in = GraphConvolution(inputs, hidden_layers_over4, max_order)
        self.gc1 = GraphConvolution(hidden_layers_over4, hidden_layers_over4, max_order)
        self.gc2 = GraphConvolution(hidden_layers_over4, hidden_layers, max_order)
        self.gc3 = GraphConvolution(hidden_layers, hidden_layers, max_order)
        self.gc4 = GraphConvolution(hidden_layers, hidden_layers_times4, max_order)
        self.gc5 = GraphConvolution(hidden_layers_times4, hidden_layers_times4, max_order)
        self.gc6 = GraphConvolution(hidden_layers_times4, hidden_layers_times4, max_order)
        self.gc7 = GraphConvolution(hidden_layers_times4, hidden_layers_times4, max_order)
        self.gc8 = GraphConvolution(hidden_layers_times4, hidden_layers_times4, max_order)
        self.gc9 = GraphConvolution(hidden_layers_times4, hidden_layers_times4, max_order)
        self.gc10 = GraphConvolution(hidden_layers_times4, hidden_layers_times4, max_order)
        self.gc11 = GraphConvolution(hidden_layers_times4, hidden_layers_times4, max_order)
        self.gc12 = GraphConvolution(hidden_layers_times4, hidden_layers_times4, max_order)
        
        self.fc1 = nn.Linear(hidden_layers_times4, hidden_layers)
        self.fc2 = nn.Linear(hidden_layers, hidden_layers)
        self.fc_out = nn.Linear(hidden_layers, outputs)

    def forward(self, x, adj_list):
        assert len(adj_list) == self.max_order

        x = torch.tanh(self.gc_in(x, adj_list))
        x = torch.tanh(self.gc1(x, adj_list)) + x
        x = torch.tanh(self.gc2(x, adj_list))
        x = torch.tanh(self.gc3(x, adj_list)) + x
        x = torch.tanh(self.gc4(x, adj_list))
        x = torch.tanh(self.gc5(x, adj_list)) + x
        x = torch.tanh(self.gc6(x, adj_list)) + x
        x = torch.tanh(self.gc7(x, adj_list)) + x
        x = torch.tanh(self.gc8(x, adj_list)) + x
        x = torch.tanh(self.gc9(x, adj_list)) + x
        x = torch.tanh(self.gc10(x, adj_list)) + x
        x = torch.tanh(self.gc11(x, adj_list)) + x
        x = torch.tanh(self.gc12(x, adj_list)) + x
        
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x)) + x
        x = torch.tanh(self.fc_out(x))
        return (x + 1) / 2

class model1_GCN_iSWAP(nn.Module):
    def __init__(self, inputs, hidden_layers, outputs, max_order=2):
        super(model1_GCN_iSWAP, self).__init__()

        self.max_order = max_order

        self.gc_in = GraphConvolution(inputs, hidden_layers, max_order)
        self.gc1 = GraphConvolution(hidden_layers, hidden_layers, max_order)
        self.gc2 = GraphConvolution(hidden_layers, hidden_layers, max_order)
        
        self.fc1 = nn.Linear(hidden_layers, hidden_layers)
        self.fc2 = nn.Linear(hidden_layers, hidden_layers)
        self.fc_out = nn.Linear(hidden_layers, outputs)

    def forward(self, x, adj_list):
        assert len(adj_list) == self.max_order

        omegaQs = x[:, :, -1:].clone()                          # (bg, N, 1)

        x = torch.tanh(self.gc_in(x, adj_list))
        x = torch.tanh(self.gc1(x, adj_list)) + x
        x = torch.tanh(self.gc2(x, adj_list)) + x
        
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x)) + x
        x = torch.tanh(self.fc_out(x))

        out = torch.cat([x, omegaQs], dim=-1)
        return out

'''
#######################################################################
############# Full-Connected Neural Networks  #########################
#######################################################################
'''
    
class model1_FC_iSWAP(torch.nn.Module):
    def __init__(self, input, hidden_layers) -> None:
        super().__init__()

        self.fc1 = nn.Linear(input, hidden_layers)
        self.fc2 = nn.Linear(hidden_layers, hidden_layers)
        self.fc3 = nn.Linear(hidden_layers, 2)

        self.sotfmax = nn.Softmax(dim=-1)

    def forward(self, x):                               # (bge, 2k+2)

        bge_size = x.shape[0]
        x = x.reshape(bge_size, 2, -1)                  # (bge, 2, k+1)
        omegaQs = x[:, :, -1].clone()                           # (bge, 2)
        x = x[:, :, :-1].reshape(bge_size, -1)          # (bge, 2k)

        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x)) + x
        x = self.fc3(x)
        x = self.sotfmax(x)                             # (bge, 2)

        out = (x * omegaQs).sum(dim=-1, keepdim=True)   # (bge, 1)
        return out