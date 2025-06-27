import sys, os
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, '..'))

import torch

from DGCNacm_model1_nns import model1_GCN_single, model1_GCN_iSWAP, model1_FC_iSWAP
from models.model2_GBFCNres import GBFCN2_single, GBFCN2_iSWAP

class ModelsConnector:
    def __init__(self, dtype=torch.float32, dev=torch.device('cpu'), max_adj_order=2) -> None:
        self.dtype = dtype
        self.dev = dev
        self.max_adj_order = max_adj_order

    @staticmethod
    def bge_loader(bg_adj, bg_N_X, balance=False):
        assert bg_adj.dim() == 3, 'bg_adj.dim() == 3 failed.'
        assert bg_adj.shape[1] == bg_adj.shape[2], 'bg_adj.shape[1] == bg_adj.shape[2] failed.'
        assert bg_N_X.dim() == 3, 'bg_N_X.dim() == 3 failed.'
        assert bg_N_X.shape[0] == bg_adj.shape[0], 'bg_n_x.shape[0] == bg_adj.shape[0] failed.'
        assert bg_N_X.shape[1] == bg_adj.shape[1], 'bg_n_x.shape[1] == bg_adj.shape[1] failed.'

        bg_size = bg_N_X.shape[0]
        N = bg_N_X.shape[1]
        X = bg_N_X.shape[2]

        bg_adj_triu = torch.triu(bg_adj, diagonal=1)    # [bg_size, N, N]

        bg_N_N_X_i = bg_N_X.reshape(bg_size, N, 1, X).repeat(1, 1, N, 1)        # [bg_size, N, N, X]
        bg_N_N_X_j = bg_N_X.reshape(bg_size, 1, N, X).repeat(1, N, 1, 1)

        bge_X_i = bg_N_N_X_i.reshape(-1, X)
        bge_X_j = bg_N_N_X_j.reshape(-1, X)
        bge_X_i = bge_X_i[(bg_adj_triu.reshape(-1) - 1).abs() < 1e-5]
        bge_X_j = bge_X_j[(bg_adj_triu.reshape(-1) - 1).abs() < 1e-5]

        if not balance:
            bge_X = torch.cat([bge_X_i, bge_X_j], dim=-1)   # [bge, 2X]
        else:
            bge_X_sum = (bge_X_i + bge_X_j) / 2
            bge_X_delta = (bge_X_i - bge_X_j).abs() / 2
            bge_X = torch.cat([bge_X_sum, bge_X_delta], dim=-1)
        return bge_X

    '''
    ###############################################
    ##### bg_renew_graphs #########################
    ###############################################
    '''

    @torch.no_grad()
    def bg_renew_graphs(self, bg_adj_mat: torch.Tensor, trial_omegaQs=None) -> None:
        '''
        '''
        assert bg_adj_mat.dim() == 3, 'bg_adj_mat.dim() == 3 failed.'
        assert bg_adj_mat.shape[1] == bg_adj_mat.shape[2], 'bg_adj_mat.shape[1] == bg_adj_mat.shape[2] failed.'
        
        self.bg_adj_mat = bg_adj_mat.to(self.dev).to(self.dtype)
        self.bg_size = bg_adj_mat.shape[0]
        self.Q_num = bg_adj_mat.shape[1]

        # for single-qubit gates
        self.bg_adj1, self.bg_adj2, self.bg_adj3, self.bg_adj4 = self._get_bg_adjs(self.bg_adj_mat)
        self.bg_adj_total = self._get_bg_adj_total()
        self.bg_node_relation = self._bg_get_node_relation()    # (bg_size, Q_num, Q_num, 5)

        # for iSWAP
        self.bgExN_relation_i, self.bgExN_relation_j = self._get_bgExN_relations()      # (bge, N, 5), (bge, N, 5)
        self.bgExN_adj_total = self._get_bgExN_adj_total()                              # (bge, N), 0 or 1

        # for model1
        self.bg_adj_norm_list = self._get_adj_norm_list(self.bg_adj1)[:self.max_adj_order]
        self.bg_structure_info = self._bg_get_stucture_info(self.bg_adj_mat)     # the input of model1-gcn
        if trial_omegaQs is not None:
            self.bg_structure_info[:, :, -1:] = trial_omegaQs.reshape(-1, self.Q_num, 1)

    ###############################################
    ########## for model1 #########################
    ###############################################

    @torch.no_grad()
    def _get_adj_norm(self, adj: torch.Tensor, dtype=torch.float32):
        '''
        Normalize the adjacent matrix.\n
        Can be used for both batch and single adjacent matrix.\n
        '''
        adj = adj + torch.eye(adj.shape[-1]).to(adj.device).to(dtype)
        rowsum = torch.sum(adj, dim=-1)
        degree_mat_inv_sqrt = torch.diag_embed(torch.pow(rowsum, -0.5))
        adj_norm = torch.matmul(torch.matmul(degree_mat_inv_sqrt, adj), degree_mat_inv_sqrt)
        return adj_norm
    
    @torch.no_grad()
    def _get_adj_norm_list(self, bg_adj1):
        bg_adj2 = torch.matmul(bg_adj1, bg_adj1)
        bg_adj3 = torch.matmul(bg_adj2, bg_adj1)
        bg_adj4 = torch.matmul(bg_adj3, bg_adj1)

        Q_num = bg_adj1.shape[1]

        bg_adj2[:, range(Q_num), range(Q_num)] = 0
        bg_adj3[:, range(Q_num), range(Q_num)] = 0
        bg_adj4[:, range(Q_num), range(Q_num)] = 0

        bg_adj2[bg_adj2 > 0] = 1
        bg_adj3[bg_adj3 > 0] = 1
        bg_adj4[bg_adj4 > 0] = 1

        # 2345 - 1
        bg_adj2[(bg_adj1 - 1).abs() < 1e-5] = 0
        bg_adj3[(bg_adj1 - 1).abs() < 1e-5] = 0
        bg_adj4[(bg_adj1 - 1).abs() < 1e-5] = 0
        # 345 - 2
        bg_adj3[(bg_adj2 - 1).abs() < 1e-5] = 0
        bg_adj4[(bg_adj2 - 1).abs() < 1e-5] = 0
        # 45 - 3
        bg_adj4[(bg_adj3 - 1).abs() < 1e-5] = 0
        # 5 - 4

        return [self._get_adj_norm(bg_adj1), self._get_adj_norm(bg_adj2), self._get_adj_norm(bg_adj3), self._get_adj_norm(bg_adj4)]
    
    @torch.no_grad()
    def _bg_get_stucture_info(self, bg_reduced_adj_mat):
        Q_num = bg_reduced_adj_mat.shape[-1]
        arange_used = torch.arange(Q_num, device=self.dev)

        bg_adj1 = bg_reduced_adj_mat.to(self.dtype)

        bg_adj2 = torch.matmul(bg_adj1, bg_adj1)
        bg_adj2 -= torch.diag_embed(bg_adj2[:, arange_used, arange_used])

        bg_adj3 = torch.matmul(bg_adj2, bg_adj1)
        bg_adj3 -= torch.diag_embed(bg_adj3[:, arange_used, arange_used])

        node_features = bg_adj1.sum(dim=-1).reshape(-1, Q_num, 1)
        node_features = torch.cat([node_features, ((bg_adj2 - 1).abs() < 1e-4).to(self.dtype).sum(dim=-1).reshape(-1, Q_num, 1)], dim=-1)
        node_features = torch.cat([node_features, ((bg_adj2 - 2).abs() < 1e-4).to(self.dtype).sum(dim=-1).reshape(-1, Q_num, 1)], dim=-1)
        node_features = torch.cat([node_features, (bg_adj2 > 2).to(self.dtype).sum(dim=-1).reshape(-1, Q_num, 1)], dim=-1)
        node_features = torch.cat([node_features, ((bg_adj3 - 1).abs() < 1e-4).to(self.dtype).sum(dim=-1).reshape(-1, Q_num, 1)], dim=-1)
        node_features = torch.cat([node_features, ((bg_adj3 - 2).abs() < 1e-4).to(self.dtype).sum(dim=-1).reshape(-1, Q_num, 1)], dim=-1)
        node_features = torch.cat([node_features, (bg_adj3 > 2).to(self.dtype).sum(dim=-1).reshape(-1, Q_num, 1)], dim=-1)
        node_features = torch.cat([node_features, ((bg_adj2 - bg_adj1).abs() < 1e-4).to(self.dtype).sum(dim=-1).reshape(-1, Q_num, 1)], dim=-1)
        node_features = torch.cat([node_features, ((bg_adj3 - bg_adj2).abs() < 1e-4).to(self.dtype).sum(dim=-1).reshape(-1, Q_num, 1)], dim=-1)
        node_features = torch.cat([node_features, ((bg_adj3 - bg_adj1).abs() < 1e-4).to(self.dtype).sum(dim=-1).reshape(-1, Q_num, 1)], dim=-1)

        random_feature = torch.randn_like(node_features[:, :, 0]).reshape(-1, Q_num, 1)
        node_features = torch.cat([node_features, random_feature], dim=-1)

        return node_features   
    
    @torch.no_grad()
    def get_model1_gcn_input_features(self):
        '''
        used for the initialization of model1-gcn.
        '''
        demo_adj = torch.tensor([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=self.dtype, device=self.dev).reshape(1, 3, 3)
        demo_structure_info = self._bg_get_stucture_info(demo_adj)
        return demo_structure_info.shape[-1]
    
    ###############################################
    ########## for single-qubit gates #############
    ###############################################

    @torch.no_grad()
    def _get_bg_adjs(self, bg_adj_mat: torch.Tensor) -> None:
        assert bg_adj_mat.dim() == 3, 'bg_adj_mat.dim() == 3 failed.'
        assert bg_adj_mat.shape[1] == bg_adj_mat.shape[2], 'bg_adj_mat.shape[1] == bg_adj_mat.shape[2] failed.'

        bg_adj1 = bg_adj_mat.to(self.dtype).to(self.dev)
        bg_adj2 = torch.matmul(bg_adj1, bg_adj1)
        bg_adj3 = torch.matmul(bg_adj2, bg_adj1)
        bg_adj4 = torch.matmul(bg_adj3, bg_adj1)

        Q_num = bg_adj_mat.shape[1]

        bg_adj2[:, range(Q_num), range(Q_num)] = 0
        bg_adj3[:, range(Q_num), range(Q_num)] = 0
        bg_adj4[:, range(Q_num), range(Q_num)] = 0

        bg_adj2[bg_adj2 > 0] = 1
        bg_adj3[bg_adj3 > 0] = 1
        bg_adj4[bg_adj4 > 0] = 1

        bg_adj2[(bg_adj1 - 1).abs() < 1e-5] = 0
        bg_adj3[(bg_adj1 - 1).abs() < 1e-5] = 0
        bg_adj4[(bg_adj1 - 1).abs() < 1e-5] = 0
        bg_adj3[(bg_adj2 - 1).abs() < 1e-5] = 0
        bg_adj4[(bg_adj2 - 1).abs() < 1e-5] = 0
        bg_adj4[(bg_adj3 - 1).abs() < 1e-5] = 0

        return bg_adj1, bg_adj2, bg_adj3, bg_adj4
    
    @torch.no_grad()
    def _get_bg_adj_total(self):
        bg_adj_total = self.bg_adj1 + self.bg_adj2 + self.bg_adj3 + self.bg_adj4
        bg_adj_total[:, range(self.Q_num), range(self.Q_num)] = 0
        bg_adj_total[bg_adj_total > 0] = 1
        return bg_adj_total
    
    @torch.no_grad()
    def _bg_get_node_relation(self):
        return torch.stack([self.bg_adj1, self.bg_adj2, self.bg_adj3, self.bg_adj4, torch.zeros_like(self.bg_adj1)], dim=3)

    ###############################################
    ########## for iSWAP gate #####################
    ###############################################

    @torch.no_grad()
    def _get_bgExN_relations(self):
        bgExN_relations_list = [self.bge_loader(self.bg_adj_mat, self.bg_node_relation[:, :, :, i]) for i in range(5)]      # [(bge, 2N), ...]

        bgExN_relations_i_list = [(bgExN_relations_list[i][:, :self.Q_num]).reshape(-1, self.Q_num, 1) for i in range(5)]   # [(bge, N, 1), ...]
        bgExN_relations_j_list = [(bgExN_relations_list[i][:, self.Q_num:]).reshape(-1, self.Q_num, 1) for i in range(5)]   # [(bge, N, 1), ...]

        bgExN_relations_i = torch.cat(bgExN_relations_i_list, dim=-1)    # (bge, N, 5)
        bgExN_relations_j = torch.cat(bgExN_relations_j_list, dim=-1)    # (bge, N, 5)
        return bgExN_relations_i, bgExN_relations_j

    @torch.no_grad()
    def _get_bgExN_adj_total(self):
        bgExN_adj_total_ij = self.bge_loader(self.bg_adj_mat, self.bg_adj_total)    # (bge, 2N)
        
        bgExN_adj_total_i = bgExN_adj_total_ij[:, :self.Q_num]    # (bge, N), 0 or 1
        bgExN_adj_total_j = bgExN_adj_total_ij[:, self.Q_num:]    # (bge, N), 0 or 1

        bgExN_adj_total = bgExN_adj_total_i * bgExN_adj_total_j    # (bge, N), 0 or 1
        return bgExN_adj_total

    '''
    ###############################################
    ##### forward_model1s #########################
    ###############################################
    '''

    def forward_model1s(self, model1_gcn_single: model1_GCN_single, model1_gcn_iSWAP: model1_GCN_iSWAP, model1_fc_iSWAP: model1_FC_iSWAP,
                        model2_X: GBFCN2_single, model2_Y: GBFCN2_single, model2_iSWAP: GBFCN2_iSWAP,
                        train_single=True, train_iSWAP=False):
        '''
        '''
        for param in model2_X.parameters(): assert not param.requires_grad, 'model2_X should not require grad.'
        for param in model2_Y.parameters(): assert not param.requires_grad, 'model2_Y should not require grad.'
        for param in model2_iSWAP.parameters(): assert not param.requires_grad, 'model2_iSWAP should not require grad.'

        bg_input = self.bg_structure_info

        # model1-gcn-single
        bg_omegaQs = model1_gcn_single(bg_input, self.bg_adj_norm_list)                                    # (bg, N, 1)

        # model1-gcn-iSWAP
        bg_model1_gcn_out = model1_gcn_iSWAP(bg_omegaQs, self.bg_adj_norm_list)                                    # (bg, N, output)

        # model1-fc-iSWAP
        bge_model1_fc_iSWAP_in = self.bge_loader(self.bg_adj_mat, bg_model1_gcn_out, balance=False)                  # (bge, 2X), useless
        bge_omegaEs = model1_fc_iSWAP(bge_model1_fc_iSWAP_in)                                                       # (bge, 1)

        # model2-single
        batch_model_in_single = self.bg_omegaQs_to_modelin_single_gate(bg_omegaQs.reshape(self.bg_size, self.Q_num))    # (bgn, 7)
        model2_out_X = model2_X(batch_model_in_single)                                                                  # (bgn, 1), -4 ~ -1
        model2_out_Y = model2_Y(batch_model_in_single)                                                                  # (bgn, 1), -4 ~ -1
        
        # model2-iSWAP
        batch_model_in_iSWAP = self.bge_omegaQs_omegaEs_to_modelin_iSWAP(bg_omegaQs.reshape(self.bg_size, self.Q_num), bge_omegaEs.reshape(-1))     # (bge, 14)
        model2_out_iSWAP = model2_iSWAP(batch_model_in_iSWAP)                                                           # (bge, 1), -4 ~ -1

        if not train_single:
            model2_out_X = model2_out_X.detach()
            model2_out_Y = model2_out_Y.detach()
        if not train_iSWAP:
            model2_out_iSWAP = model2_out_iSWAP.detach()

        log10_loss_X = self.bgn_modelout_to_aveloss_single(model2_out_X)    # (bg, N, 1), -4 ~ -1
        log10_loss_Y = self.bgn_modelout_to_aveloss_single(model2_out_Y)    # (bg, N, 1), -4 ~ -1
        log10_loss_iSWAP = self.bge_modelout_to_aveloss_iSWAP(model2_out_iSWAP)    # (bge, N, 1), -4 ~ -1

        return bg_omegaQs, bge_omegaEs, log10_loss_X, log10_loss_Y, log10_loss_iSWAP

    ###############################################
    ########## for single-qubit gates #############
    ###############################################

    def bg_omegaQs_to_modelin_single_gate(self, bg_omegaQs):
        assert bg_omegaQs.shape == (self.bg_size, self.Q_num), 'bg_omegaQs.shape == (self.bg_size, self.Q_num) failed.'

        model_in_NxN = torch.zeros((self.bg_size, self.Q_num, self.Q_num, 7), device=self.dev, dtype=self.dtype)
        model_in_NxN[:, :, :, 0] = bg_omegaQs.reshape(self.bg_size, self.Q_num, 1).repeat(1, 1, self.Q_num)
        model_in_NxN[:, :, :, 1] = bg_omegaQs.reshape(self.bg_size, 1, self.Q_num).repeat(1, self.Q_num, 1)
        model_in_NxN[:, :, :, 2:] = self.bg_node_relation

        model_in = model_in_NxN.reshape(self.bg_size * self.Q_num * self.Q_num, 7)
        model_in = model_in[(self.bg_adj_total.reshape(-1) - 1).abs() < 1e-5]
        return model_in
    
    def bgn_modelout_to_aveloss_single(self, modelout):
        assert modelout.dim() == 2, 'modelout.dim() == 2 failed.'
        assert modelout.shape[1] == 1, 'modelout.shape[1] == 1 failed.'

        loss_NxN = torch.zeros((self.bg_size * self.Q_num * self.Q_num, 1), device=self.dev, dtype=self.dtype)
        loss_NxN[(self.bg_adj_total.reshape(-1) - 1).abs() < 1e-5] = 10 ** modelout - 1e-4      # in estimator, min loss is 1e-4
        loss_NxN = loss_NxN.reshape(self.bg_size, self.Q_num, self.Q_num, 1)
        loss_NxN = torch.sum(loss_NxN, dim=2, keepdim=False)
        return loss_NxN

    ###############################################
    ########## iSWAP gate #########################
    ###############################################

    def bge_omegaQs_omegaEs_to_modelin_iSWAP(self, bg_omegaQs, bge_omegaEs):
        assert bg_omegaQs.shape == (self.bg_size, self.Q_num), 'bg_omegaQs.shape == (self.bg_size, self.Q_num) failed.'
        assert bge_omegaEs.dim() == 1, 'bge_omegaEs.dim() == 1 failed.'

        bge_num = bge_omegaEs.shape[0]

        bg_omegaQs_NxN = bg_omegaQs.reshape(self.bg_size, 1, self.Q_num).repeat(1, self.Q_num, 1)
        bge_omegaQs_Nx1 = self.bge_loader(self.bg_adj_mat, bg_omegaQs_NxN)      # (bge, 2N)
        bge_omegaQs_Nx1 = bge_omegaQs_Nx1[:, :self.Q_num]                           # (bge, N)

        bge_omegaQs_ij = self.bge_loader(self.bg_adj_mat, bg_omegaQs.reshape(self.bg_size, self.Q_num, 1))               # (bge, 2)

        model_in_bgExN = torch.zeros((bge_num, self.Q_num, 14), device=self.dev, dtype=self.dtype)
        model_in_bgExN[:, :, 0] = bge_omegaEs.reshape(bge_num, 1).repeat(1, self.Q_num)
        model_in_bgExN[:, :, 1] = bge_omegaQs_Nx1
        model_in_bgExN[:, :, 2] = bge_omegaQs_ij[:, 0].reshape(bge_num, 1).repeat(1, self.Q_num)
        model_in_bgExN[:, :, 3] = bge_omegaQs_ij[:, 1].reshape(bge_num, 1).repeat(1, self.Q_num)
        model_in_bgExN[:, :, 4:9] = self.bgExN_relation_i
        model_in_bgExN[:, :, 9:] = self.bgExN_relation_j

        model_in = model_in_bgExN.reshape(-1, 14)
        model_in = model_in[(self.bgExN_adj_total.reshape(-1) - 1).abs() < 1e-5]
        return model_in
    
    def bge_modelout_to_aveloss_iSWAP(self, modelout):
        assert modelout.dim() == 2, 'modelout.dim() == 2 failed.'
        assert modelout.shape[1] == 1, 'modelout.shape[1] == 1 failed.'

        loss_bgExN = torch.zeros_like(self.bgExN_adj_total, device=self.dev, dtype=self.dtype).reshape(-1, 1)  # (bge * N, 1)
        loss_bgExN[(self.bgExN_adj_total.reshape(-1) - 1).abs() < 1e-5] = 10 ** modelout - 1e-4      # in estimator, min loss is 1e-4
        loss_bgExN = loss_bgExN.reshape(-1, self.Q_num, 1)
        loss_bgExN = torch.sum(loss_bgExN, dim=1, keepdim=False)
        return loss_bgExN

'''
###############################################
######### simple test #########################
###############################################
'''

if __name__ == '__main__':

    import sys, os
    import datetime

    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ###############################################
    ##### load trained model2 #####################
    ###############################################

    # the load path of model2
    current_dir_models = os.path.dirname(os.path.realpath(__file__))
    sys.path.append(os.path.join(current_dir_models, '..'))
    model2_saving_dir = 'envs/estimator/results_train_use/'

    # load model2
    model2_X = torch.load(model2_saving_dir + 'model_X.pt', map_location=dev)
    model2_Y = torch.load(model2_saving_dir + 'model_Y.pt', map_location=dev)
    model2_iSWAP = torch.load(model2_saving_dir + 'model_iSWAP.pt', map_location=dev)

    model2_X.requires_grad_(False)
    model2_Y.requires_grad_(False)
    model2_iSWAP.requires_grad_(False)

    print('model2 loaded.', datetime.datetime.now())

    ###############################################
    ##### test ####################################
    ###############################################

    models_connector = ModelsConnector(dtype=torch.float32, dev=dev)
    model1_gcn_single = model1_GCN_single(models_connector.get_model1_gcn_input_features(), 16).to(dev).to(torch.float32)
    model1_gcn_iSWAP = model1_GCN_iSWAP(1, 16, 10).to(dev).to(torch.float32)
    model1_fc_iSWAP = model1_FC_iSWAP(20, 16).to(dev).to(torch.float32)

    bg_adj = torch.zeros((4, 6, 6), device=dev, dtype=torch.float32)
    bg_adj[0, :, :] = torch.tensor([[0, 1, 0, 0, 0, 0], 
                                    [1, 0, 1, 0, 0, 0], 
                                    [0, 1, 0, 1, 0, 0], 
                                    [0, 0, 1, 0, 1, 0], 
                                    [0, 0, 0, 1, 0, 1], 
                                    [0, 0, 0, 0, 1, 0]], device=dev, dtype=torch.float32)
    bg_adj[1, :, :] = torch.tensor([[0, 1, 0, 1, 0, 0], 
                                    [1, 0, 1, 0, 1, 0], 
                                    [0, 1, 0, 0, 0, 1], 
                                    [1, 0, 0, 0, 1, 0], 
                                    [0, 1, 0, 1, 0, 1], 
                                    [0, 0, 1, 0, 1, 0]], device=dev, dtype=torch.float32)
    bg_adj[2, :, :] = torch.tensor([[0, 1, 0, 1, 1, 0], 
                                    [1, 0, 1, 0, 1, 1], 
                                    [0, 1, 0, 0, 0, 1], 
                                    [1, 0, 0, 0, 1, 0], 
                                    [1, 1, 0, 1, 0, 1], 
                                    [0, 1, 1, 0, 1, 0]], device=dev, dtype=torch.float32)
    bg_adj[3, :, :] = torch.tensor([[0, 1, 0, 0, 0, 0], 
                                    [1, 0, 1, 0, 1, 1], 
                                    [0, 1, 0, 1, 1, 1], 
                                    [0, 0, 1, 0, 0, 0], 
                                    [0, 1, 1, 0, 0, 0], 
                                    [0, 1, 1, 0, 0, 0]], device=dev, dtype=torch.float32)
    
    models_connector.bg_renew_graphs(bg_adj)
    models_connector.forward_model1s(model1_gcn_single, model1_gcn_iSWAP, model1_fc_iSWAP, model2_X, model2_Y, model2_iSWAP)