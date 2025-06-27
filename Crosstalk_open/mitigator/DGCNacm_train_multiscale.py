import sys, os
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, '..'))
saving_dir = current_dir + '/results/'
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

dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dtype = torch.float32
bg_size = 32
hidden_layers_single = 64
hidden_layers_iSWAP = 64
output_gcn1 = 64

###############################################
##### Qubits Graph ############################
###############################################

qubits_graph0 = QuGraph_list(8, 8, dtype=dtype, device=dev)
Q_num0 = qubits_graph0.basic_node_num
qubits_graph1 = QuGraph_list(11, 11, dtype=dtype, device=dev)
Q_num1 = qubits_graph1.basic_node_num
qubits_graph2 = QuGraph_list(16, 16, dtype=dtype, device=dev)
Q_num2 = qubits_graph2.basic_node_num

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

###############################################
##### models connector and model1 #############
###############################################

models_connector0 = ModelsConnector(dtype=torch.float32, dev=dev, max_adj_order=3)
models_connector1 = ModelsConnector(dtype=torch.float32, dev=dev, max_adj_order=3)
models_connector2 = ModelsConnector(dtype=torch.float32, dev=dev, max_adj_order=3)
model1_gcn_single = model1_GCN_single(models_connector0.get_model1_gcn_input_features(), hidden_layers_single, max_order=models_connector0.max_adj_order).to(dev).to(torch.float32)
model1_gcn_iSWAP = model1_GCN_iSWAP(1, hidden_layers_iSWAP, output_gcn1, max_order=models_connector0.max_adj_order).to(dev).to(torch.float32)
model1_fc_iSWAP = model1_FC_iSWAP(output_gcn1*2, hidden_layers_iSWAP).to(dev).to(torch.float32)

optimizer = torch.optim.Adam(list(model1_gcn_single.parameters()) + list(model1_gcn_iSWAP.parameters()) + list(model1_fc_iSWAP.parameters()), 
                             lr=1e-4)

###############################################
##### training ################################
###############################################

loss_list = []
loss_X_list = []
loss_Y_list = []
loss_iSWAP_list = []
train_single = True
train_iSWAP = True

for epoch in range(120_0001):

    # renew the graphs
    if epoch % 10 == 0:
        bg_reduced_adj_mat0 = qubits_graph0.bg_generate_reduced_graph(bg_size=bg_size, node_keeping_rate=0.85, edge_keeping_rate=0.85)
        models_connector0.bg_renew_graphs(bg_reduced_adj_mat0)
        bg_reduced_adj_mat1 = qubits_graph1.bg_generate_reduced_graph(bg_size=bg_size, node_keeping_rate=0.85, edge_keeping_rate=0.85)
        models_connector1.bg_renew_graphs(bg_reduced_adj_mat1)
        bg_reduced_adj_mat2 = qubits_graph2.bg_generate_reduced_graph(bg_size=bg_size, node_keeping_rate=0.85, edge_keeping_rate=0.85)
        models_connector2.bg_renew_graphs(bg_reduced_adj_mat2)

    # forward
    _, _, log10_loss_X0, log10_loss_Y0, log10_loss_iSWAP0 = models_connector0.forward_model1s(model1_gcn_single, model1_gcn_iSWAP, model1_fc_iSWAP, model2_X, model2_Y, model2_iSWAP, train_single=train_single, train_iSWAP=train_iSWAP)
    _, _, log10_loss_X1, log10_loss_Y1, log10_loss_iSWAP1 = models_connector1.forward_model1s(model1_gcn_single, model1_gcn_iSWAP, model1_fc_iSWAP, model2_X, model2_Y, model2_iSWAP, train_single=train_single, train_iSWAP=train_iSWAP)
    _, _, log10_loss_X2, log10_loss_Y2, log10_loss_iSWAP2 = models_connector2.forward_model1s(model1_gcn_single, model1_gcn_iSWAP, model1_fc_iSWAP, model2_X, model2_Y, model2_iSWAP, train_single=train_single, train_iSWAP=train_iSWAP)

    mean_loss_X = (log10_loss_X0.mean() + log10_loss_X1.mean() + log10_loss_X2.mean()) / 3
    mean_loss_Y = (log10_loss_Y0.mean() + log10_loss_Y1.mean() + log10_loss_Y2.mean()) / 3
    mean_loss_iSWAP = (log10_loss_iSWAP0.mean() + log10_loss_iSWAP1.mean() + log10_loss_iSWAP2.mean()) / 3
    loss = mean_loss_X + mean_loss_Y + mean_loss_iSWAP

    # backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    loss_list.append(loss.item())
    loss_X_list.append(mean_loss_X.item())
    loss_Y_list.append(mean_loss_Y.item())
    loss_iSWAP_list.append(mean_loss_iSWAP.item())

    # print and plot
    if epoch % 10 == 0:
        print("epoch: {}, loss: {:.4f} = {:.4f} + {:.4f} + {:.4f}, ave-300: {:.4f}".format(epoch, loss.item(), mean_loss_X.item(), mean_loss_Y.item(), mean_loss_iSWAP.item(), statistics.mean(loss_list[-300:])))
        if epoch % 100 == 0:

            # lr decay
            for param_group in optimizer.param_groups:
                # param_group['lr'] *= 0.99
                param_group['lr'] = max(0.9996*param_group['lr'], 2e-5)

            # see memory usage
            if dev == torch.device('cuda'):
                memoryuse = torch.cuda.memory_reserved()/1024**3
                print('memory: {:.4f} GB; '.format(memoryuse), 'time:', datetime.datetime.now())
                sys.stdout.flush()
                if memoryuse > 20:
                    torch.cuda.empty_cache()
                    print('memory cleaned.')

    # test
    if epoch % 100 == 99:
        with torch.no_grad():
            plot_adj = bg_reduced_adj_mat0.reshape(3, -1, Q_num0, Q_num0)[:, 0, :, :]
            models_connector0.bg_renew_graphs(plot_adj)
            omegaQs, omegaEs, _, _, _ = models_connector0.forward_model1s(model1_gcn_single, model1_gcn_iSWAP, model1_fc_iSWAP, 
                                                                                                            model2_X, model2_Y, model2_iSWAP)
            omegaQs = omegaQs.reshape(-1).detach()
            omegaEs = omegaEs.reshape(-1).detach()
            qubits_graph0.plot_graph(adj_mat=plot_adj.reshape(3, Q_num0, Q_num0), saving_path=saving_dir + 'solved_graph.png', 
                                    plot_color=True, node_freqs=omegaQs, edge_freqs=omegaEs)

    # save
    if epoch % 1000 == 999:
        data_dir = saving_dir + 'save_data/'
        torch.save(model1_gcn_single, data_dir + 'model1_gcn_single.pt')
        torch.save(model1_gcn_iSWAP, data_dir + 'model1_gcn_iSWAP.pt')
        torch.save(model1_fc_iSWAP, data_dir + 'model1_fc_iSWAP.pt')
        torch.save(torch.tensor(loss_list), data_dir + 'loss_list.pt')
        torch.save(torch.tensor(loss_X_list), data_dir + 'loss_X_list.pt')
        torch.save(torch.tensor(loss_Y_list), data_dir + 'loss_Y_list.pt')
        torch.save(torch.tensor(loss_iSWAP_list), data_dir + 'loss_iSWAP_list.pt')
        print('saved.')

        # save specific
        if epoch == 65_0000 - 1:
            data_dir = saving_dir + 'save_data/'
            torch.save(model1_gcn_single, data_dir + 'model1_gcn_single_65w.pt')
            torch.save(model1_gcn_iSWAP, data_dir + 'model1_gcn_iSWAP_65w.pt')
            torch.save(model1_fc_iSWAP, data_dir + 'model1_fc_iSWAP_65w.pt')
            torch.save(torch.tensor(loss_list), data_dir + 'loss_list_65w.pt')
            torch.save(torch.tensor(loss_X_list), data_dir + 'loss_X_list_65w.pt')
            torch.save(torch.tensor(loss_Y_list), data_dir + 'loss_Y_list_65w.pt')
            torch.save(torch.tensor(loss_iSWAP_list), data_dir + 'loss_iSWAP_list_65w.pt')
            print('saved.')
