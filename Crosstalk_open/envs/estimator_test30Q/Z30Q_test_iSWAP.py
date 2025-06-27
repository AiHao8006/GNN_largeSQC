import sys, os
current_dir = os.path.dirname(os.path.realpath(__file__))
outer_dir_env = os.path.join(current_dir, '..')
sys.path.append(os.path.join(outer_dir_env, '..'))

import torch
import matplotlib.pyplot as plt
import datetime
import math

from envs.estimator.SmallGraphs import SmallGraphs
from models.model2_GBFCNres import bw_DataEmbedder, GBFCN2_iSWAP
from envs.estimator.cal_r2 import r_square

###############################################
##### global parameters #######################
###############################################

dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
dtype = torch.float32

data_datetime_path = '20250409_/'
model_dir = current_dir + '/../estimator/results_train_use/'

def test_3oQ_iSWAP(seed=2, model_dir=model_dir):

    gate_type = 'iSWAP'

    torch.manual_seed(seed)

    ###############################################
    ##### model ###################################
    ###############################################

    model = torch.load(model_dir + f'model_{gate_type}.pt')
    model = model.to(dev)

    ###############################################
    ##### data ####################################
    ###############################################

    # data embedder

    smallgraphs = SmallGraphs(device=dev)
    graph_1x30l_adj_mat = smallgraphs.load_predefined_line_1x30().to(dtype).to(dev)
    graph_adj_mats = [graph_1x30l_adj_mat]
    graph_names = ['1x30l']

    graph_data_embedders = [bw_DataEmbedder(graph_adj_mat, dev=dev, dtype=dtype) for graph_adj_mat in graph_adj_mats]

    # load data
    data_dir = outer_dir_env + '/../../data_saving/' + data_datetime_path
    loaded_datas = [torch.load(os.path.join(data_dir, f"graph_{graph_names[graph_adj_mat_id]}_{gate_type}_data.pt")) for graph_adj_mat_id in range(len(graph_adj_mats))]
    loaded_datas = [(data[0].to(dtype).to(dev), data[1].to(dtype).to(dev), data[2].to(dtype).to(dev)) for data in loaded_datas]
    print("data loaded.", datetime.datetime.now())
    sys.stdout.flush()

    # embed data
    embedded_datas = []
    for graph_adj_mat_id in range(len(graph_adj_mats)):
        graph_data_embedder = graph_data_embedders[graph_adj_mat_id]
        loaded_data = loaded_datas[graph_adj_mat_id]
        
        embedded_data_in = graph_data_embedder.bw_datain_to_modelin_iSWAP(loaded_data[0], loaded_data[1])
        embedded_data_out = graph_data_embedder.bw_dataout_to_modelout_iSWAP(loaded_data[2])

        embedded_datas.append((embedded_data_in, embedded_data_out))

    print("data embedded.", datetime.datetime.now())
    sys.stdout.flush()

    test_datas = embedded_datas

    # test
    model.eval()

    model_in = test_datas[0][0]
    model_out = test_datas[0][1]

    no_data_mask = (model_in[:, :4]).abs().sum(dim=1) < 1e-5
    model_in = model_in[~no_data_mask]
    model_out = model_out[~no_data_mask]

    with torch.no_grad():
        model_out_pred = model(model_in)

    loss_test_data = torch.mean((model_out_pred - model_out).pow(2))
    r2_test_data = r_square(model_out, model_out_pred)

    print("test loss:", loss_test_data.item())
    print("test r2:", r2_test_data.item())
    sys.stdout.flush()


if __name__ == "__main__":
    test_3oQ_iSWAP(seed=2)