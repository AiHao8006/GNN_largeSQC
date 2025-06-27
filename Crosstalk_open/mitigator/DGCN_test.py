import sys, os
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, '..'))
saving_dir = current_dir + '/results_test/'

print('PID:', os.getpid())

import torch
import matplotlib.pyplot as plt
import datetime
import statistics
import math

from QubitsGraph.QuGraphs_list import QuGraph_list
from models.model2_GBFCNres import GBFCN2_single, GBFCN2_iSWAP
from DGCNacm_model1_nns import model1_GCN_single, model1_GCN_iSWAP, model1_FC_iSWAP
from DGCNacm_ModelsConnector_1loss_2loader import ModelsConnector

torch.manual_seed(2)

dev = torch.device('cpu')
dtype = torch.float32

###############################################
##### load trained model2 #####################
###############################################

# the load path of model2
current_dir_models = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(current_dir_models, '..'))
model2_saving_dir = current_dir_models + '/../envs/estimator/results_train_use/'

# load model2
model2_X = torch.load(model2_saving_dir + 'model_X.pt', map_location=dev)
model2_Y = torch.load(model2_saving_dir + 'model_Y.pt', map_location=dev)
model2_iSWAP = torch.load(model2_saving_dir + 'model_iSWAP.pt', map_location=dev)

model2_X.requires_grad_(False)
model2_Y.requires_grad_(False)
model2_iSWAP.requires_grad_(False)

print('model2 loaded.', datetime.datetime.now())
sys.stdout.flush()

###############################################
##### load trained model1 #####################
###############################################

model1_saving_dir = current_dir_models + '/../mitigator/results/DGCNac3_save_data/'

models_connector = ModelsConnector(dtype=torch.float32, dev=dev, max_adj_order=3)
model1_gcn_single = torch.load(model1_saving_dir + 'model1_gcn_single.pt', map_location=dev)
model1_gcn_iSWAP = torch.load(model1_saving_dir + 'model1_gcn_iSWAP.pt', map_location=dev)
model1_fc_iSWAP = torch.load(model1_saving_dir + 'model1_fc_iSWAP.pt', map_location=dev)

'''
###############################################
##### test func ###############################
###############################################
'''

@torch.no_grad()
def DGCNac3_test(QuGraph_xy_num, QuGraph_keeping_rate, plot=False, batch_size=128):

    qubits_graph = QuGraph_list(QuGraph_xy_num, QuGraph_xy_num, dtype=dtype, device=dev)
    Q_num = qubits_graph.basic_node_num

    reduced_adj_mat = qubits_graph.bg_generate_reduced_graph(bg_size=1, node_keeping_rate=QuGraph_keeping_rate, edge_keeping_rate=QuGraph_keeping_rate)
    bg_reduced_adj_mat = reduced_adj_mat.reshape(1, 3, Q_num, Q_num).repeat(batch_size, 1, 1, 1)

    models_connector.bg_renew_graphs(bg_reduced_adj_mat.reshape(-1, Q_num, Q_num))
    omegaQs, omegaEs, loss_X, loss_Y, loss_iSWAP = models_connector.forward_model1s(model1_gcn_single, model1_gcn_iSWAP, model1_fc_iSWAP, model2_X, model2_Y, model2_iSWAP)

    bg_omegaQs = omegaQs.reshape(batch_size, -1)
    bg_omegaEs = omegaEs.reshape(batch_size, -1)
    bg_loss_X = loss_X.reshape(batch_size, -1).mean(dim=1)
    bg_loss_Y = loss_Y.reshape(batch_size, -1).mean(dim=1)
    bg_loss_iSWAP = loss_iSWAP.reshape(batch_size, -1).mean(dim=1)

    bg_loss = bg_loss_X + bg_loss_Y + bg_loss_iSWAP         # (batch_size,)
    bg_loss_min = bg_loss.min().item()
    bg_loss_argmin = bg_loss.argmin().item()

    if plot:
        omegaQs = bg_omegaQs[bg_loss_argmin].reshape(-1)
        omegaEs = bg_omegaEs[bg_loss_argmin].reshape(-1)
        qubits_graph.plot_graph(adj_mat=reduced_adj_mat, saving_path=saving_dir + 'DGCNac3_test_graph.png',
                                plot_color=True, node_freqs=omegaQs, edge_freqs=omegaEs)
        
        save_data = (reduced_adj_mat, bg_omegaQs[bg_loss_argmin], bg_omegaEs[bg_loss_argmin])
        torch.save(save_data, saving_dir + 'DGCNac3_test_data.pt')

    return bg_loss_X[bg_loss_argmin].mean().item(), bg_loss_Y[bg_loss_argmin].mean().item(), bg_loss_iSWAP[bg_loss_argmin].mean().item()


def test(QuGraph_xy_num=8, batch_size = 512):
    print("The estimated qubit num is: {}^2 * 0.85 = {:.2f}".format(QuGraph_xy_num, QuGraph_xy_num**2 * 0.85))
    print("The test batch size is: {}".format(batch_size))
    sys.stdout.flush()

    plot = False
    for i in range(0, 10):
        # if i == 9:
        #     plot = True
        loss_X, loss_Y, loss_iSWAP = DGCNac3_test(QuGraph_xy_num, 0.85, plot=plot, batch_size=batch_size)
        print('loss: {:.4f} = {:.4f} + {:.4f} + {:.4f}'.format(loss_X + loss_Y + loss_iSWAP, loss_X, loss_Y, loss_iSWAP))
        sys.stdout.flush()

    print('test finished.', datetime.datetime.now())
    sys.stdout.flush()


if __name__ == '__main__':
    test(6, 512)
    test(8, 512)
    test(11, 512)
    test(16, 64)
    test(23, 8)
    test(32, 2)
