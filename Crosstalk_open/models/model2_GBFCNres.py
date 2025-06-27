import torch
import math

class bw_DataEmbedder:
    def __init__(self, adj_mat, dev=torch.device('cpu'), dtype=torch.float32) -> None:
        self.adj_mat = adj_mat.to(dev).to(dtype)
        self.dev = dev
        self.dtype = dtype

        self.adj1, self.adj2, self.adj3, self.adj4, self.adj5 = self._get_adjs()
        self.node_relation = self._get_node_relation()      # (node_num, node_num, 5)

        # other used variables
        self.Q_num = self.adj_mat.shape[0]
        self.adj_total = self.adj1 + self.adj2 + self.adj3 + self.adj4 + self.adj5
        self.adj_total[range(self.Q_num), range(self.Q_num)] = 0
        self.adj_total[self.adj_total > 0] = 1

        # for iSWAP gate
        self.E_num = round(self.adj_mat.sum().item() / 2)
        self.ExN_relation_i, self.ExN_relation_j = self._get_ExN_relations()
        self.ExN_adj_total = self._get_ExN_adj_total()

    @torch.no_grad()
    def _get_adjs(self):
        adj1 = self.adj_mat
        adj2 = torch.matmul(adj1, adj1)
        adj3 = torch.matmul(adj2, adj1)
        adj4 = torch.matmul(adj3, adj1)
        adj5 = torch.matmul(adj4, adj1)

        Q_num = adj1.shape[0]
        adj2[range(Q_num), range(Q_num)] = 0
        adj3[range(Q_num), range(Q_num)] = 0
        adj4[range(Q_num), range(Q_num)] = 0
        adj5[range(Q_num), range(Q_num)] = 0

        adj2[adj2 > 0] = 1
        adj3[adj3 > 0] = 1
        adj4[adj4 > 0] = 1
        adj5[adj5 > 0] = 1

        adj2[(adj1 - 1).abs() < 1e-5] = 0
        adj3[(adj1 - 1).abs() < 1e-5] = 0
        adj4[(adj1 - 1).abs() < 1e-5] = 0
        adj5[(adj1 - 1).abs() < 1e-5] = 0
        adj3[(adj2 - 1).abs() < 1e-5] = 0
        adj4[(adj2 - 1).abs() < 1e-5] = 0
        adj5[(adj2 - 1).abs() < 1e-5] = 0
        adj4[(adj3 - 1).abs() < 1e-5] = 0
        adj5[(adj3 - 1).abs() < 1e-5] = 0
        adj5[(adj4 - 1).abs() < 1e-5] = 0

        return adj1, adj2, adj3, adj4, adj5
    
    @torch.no_grad()
    def _get_node_relation(self):
        return torch.stack([self.adj1, self.adj2, self.adj3, self.adj4, self.adj5], dim=2)
    
    ############################################################
    ################ single qubit gate #########################
    ############################################################

    @torch.no_grad()
    def bw_datain_to_modelin_single_gate(self, data_in):
        assert data_in.dim() == 2, f"the datain dim is {data_in.dim()}, while the required dim is 2"
        assert data_in.shape[1] == self.Q_num, f"the datain shape is {data_in.shape}, while the Q_num is {self.Q_num}"
        
        bw_size = data_in.shape[0]
        model_in_NxN = torch.zeros((bw_size, self.Q_num, self.Q_num, 7), dtype=self.dtype, device=self.dev)
        model_in_NxN[:, :, :, 0] = data_in.reshape(bw_size, self.Q_num, 1).repeat(1, 1, self.Q_num)
        model_in_NxN[:, :, :, 1] = data_in.reshape(bw_size, 1, self.Q_num).repeat(1, self.Q_num, 1)
        model_in_NxN[:, :, :, 2:] = self.node_relation.reshape(1, self.Q_num, self.Q_num, 5).repeat(bw_size, 1, 1, 1)

        model_in = model_in_NxN.reshape(bw_size, self.Q_num * self.Q_num, 7)
        model_in = model_in[:, (self.adj_total.reshape(-1) - 1).abs() < 1e-5]

        model_in = model_in.reshape(-1, 7)
        return model_in
    
    @torch.no_grad()
    def bw_dataout_to_modelout_single_gate(self, data_out):
        assert data_out.dim() == 3, f"the dataout dim is {data_out.dim()}, while the required dim is 3"
        assert data_out.shape[1] == data_out.shape[2] == self.Q_num, f"the dataout shape is {data_out.shape}, while the Q_num is {self.Q_num}"

        bw_size = data_out.shape[0]
        model_out_NxN = data_out.reshape(bw_size, self.Q_num, self.Q_num, 1)
        model_out = model_out_NxN.reshape(bw_size, self.Q_num * self.Q_num, 1)
        model_out = model_out[:, (self.adj_total.reshape(-1) - 1).abs() < 1e-5]

        model_out = torch.log10(model_out.clamp(1e-4, 1e-1))

        model_out = model_out.reshape(-1, 1)
        return model_out
    
    @torch.no_grad()
    def bw_modelout_to_dataout_single_gate(self, model_out):
        assert model_out.dim() == 2, f"the modelout dim is {model_out.dim()}, while the required dim is 2"
        assert model_out.shape[1] == 1, f"the modelout shape is {model_out.shape}, while the required shape is (?, 1)"

        adj_total_edge_num = ((self.adj_total.reshape(-1) - 1).abs() < 1e-5).to(self.dtype).sum().round().int().item()
        model_out = model_out.reshape(-1, adj_total_edge_num)

        data_out = torch.zeros((model_out.shape[0], self.Q_num * self.Q_num), dtype=self.dtype, device=self.dev)
        data_out[:, (self.adj_total.reshape(-1) - 1).abs() < 1e-5] = 10 ** model_out
        data_out = data_out.reshape(-1, self.Q_num, self.Q_num)
        return data_out
    
    ############################################################
    ################ iSWAP gate ################################
    ############################################################

    @staticmethod
    def bwe_loader(adj_mat, bw_N_X):
        assert adj_mat.dim() == 2, f"the adj_mat dim is {adj_mat.dim()}, while the required dim is 2"
        assert adj_mat.shape[0] == adj_mat.shape[1], f"the adj_mat shape is {adj_mat.shape}, while the required shape is (N, N)"
        assert bw_N_X.dim() == 3, f"the bw_N_X dim is {bw_N_X.dim()}, while the required dim is 3"
        assert bw_N_X.shape[1] == adj_mat.shape[0], f"the bw_N_X shape is {bw_N_X.shape}, while the adj_mat shape is {adj_mat.shape}"

        bw_size = bw_N_X.shape[0]
        N = bw_N_X.shape[1]
        X = bw_N_X.shape[2]

        bw_adj = adj_mat.reshape(1, N, N).repeat(bw_size, 1, 1)
        bw_adj_triu = torch.triu(bw_adj, diagonal=1)            # (bw_size, N, N)

        bw_N_N_X_i = bw_N_X.reshape(bw_size, N, 1, X).repeat(1, 1, N, 1)
        bw_N_N_X_j = bw_N_X.reshape(bw_size, 1, N, X).repeat(1, N, 1, 1)

        bwe_X_i = bw_N_N_X_i.reshape(-1, X)
        bwe_X_j = bw_N_N_X_j.reshape(-1, X)
        bwe_X_i = bwe_X_i[(bw_adj_triu.reshape(-1) - 1).abs() < 1e-5]
        bwe_X_j = bwe_X_j[(bw_adj_triu.reshape(-1) - 1).abs() < 1e-5]

        bwe_X = torch.cat([bwe_X_i, bwe_X_j], dim=1)
        return bwe_X
    
    @torch.no_grad()
    def _get_ExN_relations(self):
        ExN_relations_list = [self.bwe_loader(self.adj_mat, self.node_relation[:, :, i].reshape(1, self.Q_num, self.Q_num)) 
                              for i in range(5)]     # [(E, 2N), ...]
        
        ExN_relations_i_list = [ExN_relations_list[i][:, :self.Q_num].reshape(-1, self.Q_num, 1) for i in range(5)]    # [(E, N, 1), ...]
        ExN_relations_j_list = [ExN_relations_list[i][:, self.Q_num:].reshape(-1, self.Q_num, 1) for i in range(5)]    # [(E, N, 1), ...]

        ExN_relation_i = torch.cat(ExN_relations_i_list, dim=2)     # (E, N, 5)
        ExN_relation_j = torch.cat(ExN_relations_j_list, dim=2)     # (E, N, 5)
        return ExN_relation_i, ExN_relation_j
    
    @torch.no_grad()
    def _get_ExN_adj_total(self):
        ExN_adj_total_ij = self.bwe_loader(self.adj_mat, self.adj_total.reshape(1, self.Q_num, self.Q_num))    # (E, 2N)

        ExN_adj_total_i = ExN_adj_total_ij[:, :self.Q_num]   # (E, N), 0 or 1
        ExN_adj_total_j = ExN_adj_total_ij[:, self.Q_num:]   # (E, N), 0 or 1

        ExN_adj_total = ExN_adj_total_i * ExN_adj_total_j   # (E, N), 0 or 1
        return ExN_adj_total
    
    @torch.no_grad()
    def bw_datain_to_modelin_iSWAP(self, data_in_Q, data_in_E):
        assert data_in_Q.dim() == 2, f"the datain_Q dim is {data_in_Q.dim()}, while the required dim is 2"
        assert data_in_Q.shape[1] == self.Q_num, f"the datain_Q shape is {data_in_Q.shape}, while the Q_num is {self.Q_num}"
        assert data_in_E.dim() == 2, f"the datain_E dim is {data_in_E.dim()}, while the required dim is 2"
        assert data_in_E.shape[1] == self.E_num, f"the datain_E shape is {data_in_E.shape}, while the E_num is {self.E_num}"
        assert data_in_Q.shape[0] == data_in_E.shape[0], f"the datain_Q shape is {data_in_Q.shape}, while the datain_E shape is {data_in_E.shape}"

        bw_size = data_in_Q.shape[0]

        bw_omegaQs_NxN = data_in_Q.reshape(bw_size, 1, self.Q_num).repeat(1, self.Q_num, 1)
        bwe_omegaQs_Nx1 = self.bwe_loader(self.adj_mat, bw_omegaQs_NxN)         # (bw_size * E, 2N)
        bwe_omegaQs_Nx1 = bwe_omegaQs_Nx1[:, :self.Q_num]                       # (bw_size * E, N)

        bwe_omegaQs_ij = self.bwe_loader(self.adj_mat, data_in_Q.reshape(bw_size, self.Q_num, 1))    # (bw_size * E, 2)

        model_in_ExN = torch.zeros((bw_size, self.E_num, self.Q_num, 14), dtype=self.dtype, device=self.dev)
        model_in_ExN[:, :, :, 0] = data_in_E.reshape(bw_size, self.E_num, 1).repeat(1, 1, self.Q_num)               # wE
        model_in_ExN[:, :, :, 1] = bwe_omegaQs_Nx1.reshape(bw_size, self.E_num, self.Q_num)                         # wk
        model_in_ExN[:, :, :, 2] = bwe_omegaQs_ij[:, 0].reshape(bw_size, self.E_num, 1).repeat(1, 1, self.Q_num)    # wi
        model_in_ExN[:, :, :, 3] = bwe_omegaQs_ij[:, 1].reshape(bw_size, self.E_num, 1).repeat(1, 1, self.Q_num)    # wj
        model_in_ExN[:, :, :, 4:9] =  self.ExN_relation_i.reshape(1, self.E_num, self.Q_num, 5).repeat(bw_size, 1, 1, 1)  # adjs[i, k]
        model_in_ExN[:, :, :, 9:] =  self.ExN_relation_j.reshape(1, self.E_num, self.Q_num, 5).repeat(bw_size, 1, 1, 1)  # adjs[j, k]

        model_in = model_in_ExN.reshape(bw_size, self.E_num * self.Q_num, 14)
        model_in = model_in[:, (self.ExN_adj_total.reshape(-1) - 1).abs() < 1e-5]

        model_in = model_in.reshape(-1, 14)
        return model_in

    @torch.no_grad()
    def bw_dataout_to_modelout_iSWAP(self, data_out):
        assert data_out.dim() == 3, f"the dataout dim is {data_out.dim()}, while the required dim is 3"
        assert data_out.shape[1] == self.E_num, f"the dataout shape is {data_out.shape}, while the E_num is {self.E_num}"
        assert data_out.shape[2] == self.Q_num, f"the dataout shape is {data_out.shape}, while the Q_num is {self.Q_num}"

        bw_size = data_out.shape[0]
        model_out_ExN = data_out.reshape(bw_size, self.E_num, self.Q_num, 1)
        model_out = model_out_ExN.reshape(bw_size, self.E_num * self.Q_num, 1)
        model_out = model_out[:, (self.ExN_adj_total.reshape(-1) - 1).abs() < 1e-5]

        model_out = torch.log10(model_out.clamp(1e-4, 1e-1))

        model_out = model_out.reshape(-1, 1)
        return model_out
    
    @torch.no_grad()
    def bw_modelout_to_dataout_iSWAP(self, model_out):
        assert model_out.dim() == 2, f"the modelout dim is {model_out.dim()}, while the required dim is 2"
        assert model_out.shape[1] == 1, f"the modelout shape is {model_out.shape}, while the required shape is (?, 1)"

        ExN_adj_total_edge_num = ((self.ExN_adj_total.reshape(-1) - 1).abs() < 1e-5).to(self.dtype).sum().round().int().item()
        model_out = model_out.reshape(-1, ExN_adj_total_edge_num)

        data_out = torch.zeros((model_out.shape[0], self.E_num * self.Q_num), dtype=self.dtype, device=self.dev)
        data_out[:, (self.ExN_adj_total.reshape(-1) - 1).abs() < 1e-5] = 10 ** model_out
        data_out = data_out.reshape(-1, self.E_num, self.Q_num)
        return data_out


'''
############################################################
################ fit funcs #################################    
############################################################
'''

class GBFCN2_single(torch.nn.Module):
    def __init__(self, hidden_layers=100, output=1, dropout=0.2) -> None:
        super().__init__()

        # input = 7         # 2 + 5

        self.fc_w_in = torch.nn.Linear(1, hidden_layers)
        self.fc_w1 = torch.nn.Linear(hidden_layers, hidden_layers)
        self.fc_w2 = torch.nn.Linear(hidden_layers, hidden_layers)

        self.fc_adj_in = torch.nn.Linear(5, hidden_layers)      # xadj[:, 4] = 0, so only 4 features are considered
        self.fc_adj1 = torch.nn.Linear(hidden_layers, hidden_layers)
        self.fc_adj2 = torch.nn.Linear(hidden_layers, hidden_layers)

        self.fc3 = torch.nn.Linear(hidden_layers*2, hidden_layers)
        self.fc_out = torch.nn.Linear(hidden_layers, output)

        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x):
        assert x.dim() == 2, f"the input dim is {x.dim()}, while the required dim is 2"
        assert x.shape[1] == 7, f"the input shape is {x.shape}, while the required shape is (?, 7)"

        xw = x[:, :2]
        xadj = x[:, 2:]

        xw = (xw[:, 0] - xw[:, 1]).reshape(-1, 1)

        xw = torch.tanh(self.fc_w_in(xw))
        xw = torch.tanh(self.fc_w1(xw)) + xw
        # xw = self.dropout(xw)
        xw = torch.tanh(self.fc_w2(xw)) + xw

        xadj[:, 4] = 0      # only neighbors with nearest, 2-hop, 3-hop, 4-hop are considered

        xadj = torch.nn.functional.relu(self.fc_adj_in(xadj))
        xadj = torch.nn.functional.relu(self.fc_adj1(xadj))
        xadj = self.dropout(xadj)
        xadj = torch.nn.functional.relu(self.fc_adj2(xadj))
        xadj = self.dropout(xadj)

        x = torch.cat([xw, xadj], dim=1)
        x = torch.tanh(self.fc3(x))
        x = self.fc_out(x)

        return (torch.exp(x+1) - 4 - math.exp(1)).clamp(-4, -1)

class GBFCN2_iSWAP(torch.nn.Module):
    def __init__(self, hidden_layers=100, output=1, dropout=0.2) -> None:
        super().__init__()

        # input = 14        # 4 + 10

        self.fc_w_in = torch.nn.Linear(6, hidden_layers)
        self.fc_w1 = torch.nn.Linear(hidden_layers, hidden_layers)
        self.fc_w2 = torch.nn.Linear(hidden_layers, hidden_layers)

        self.fc_adj_in = torch.nn.Linear(10, hidden_layers)     # xadj[:, 4] = 0, xadj[:, 9] = 0, so only 8 features are considered
        self.fc_adj1 = torch.nn.Linear(hidden_layers, hidden_layers)
        self.fc_adj2 = torch.nn.Linear(hidden_layers, hidden_layers)

        self.fc3 = torch.nn.Linear(hidden_layers*2, hidden_layers)
        self.fc_out = torch.nn.Linear(hidden_layers, output)

        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x):
        assert x.dim() == 2, f"the input dim is {x.dim()}, while the required dim is 2"
        assert x.shape[1] == 14, f"the input shape is {x.shape}, while the required shape is (?, 14)"

        xw = x[:, :4]
        xadj = x[:, 4:]

        xw_delta1 = xw[:, :3] - xw[:, 3].reshape(-1, 1)
        xw_delta2 = xw[:, :2] - xw[:, 2].reshape(-1, 1)
        xw_delta3 = xw[:, :1] - xw[:, 1].reshape(-1, 1)
        xw = torch.cat([xw_delta1, xw_delta2, xw_delta3], dim=1)

        xw = torch.tanh(self.fc_w_in(xw))
        xw = torch.tanh(self.fc_w1(xw)) + xw
        xw = torch.tanh(self.fc_w2(xw)) + xw

        xadj[:, 4] = 0      # only neighbors with nearest, 2-hop, 3-hop, 4-hop are considered
        xadj[:, 9] = 0      # only neighbors with nearest, 2-hop, 3-hop, 4-hop are considered

        xadj = torch.nn.functional.relu(self.fc_adj_in(xadj))
        xadj = torch.nn.functional.relu(self.fc_adj1(xadj))
        xadj = self.dropout(xadj)
        xadj = torch.nn.functional.relu(self.fc_adj2(xadj))
        xadj = self.dropout(xadj)

        x = torch.cat([xw, xadj], dim=1)
        x = torch.tanh(self.fc3(x))
        x = self.fc_out(x)

        return (torch.exp(x+1) - 4 - math.exp(1)).clamp(-4, -1)
    
    