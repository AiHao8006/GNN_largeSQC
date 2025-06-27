import sys, os
current_dir = os.path.dirname(os.path.realpath(__file__))
outer_dir_env = os.path.join(current_dir, '..')
sys.path.append(os.path.join(outer_dir_env, '..'))
saving_dir = current_dir + '/results_train/'

import torch
import matplotlib.pyplot as plt
import datetime
import math

from SmallGraphs import SmallGraphs
from models.model2_GBFCNres import bw_DataEmbedder, GBFCN2_single
from envs.estimator.cal_r2 import r_square

###############################################
##### global parameters #######################
###############################################

plt.figure(figsize=(10, 6))

dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
dtype = torch.float32

bw_size = 256
lr_decay = 0.97

data_datetime_path = '0601-1958_/'

def fit_smallgraphs_single(seed=2, gate_type='Y', print_or_plot_or_savemodel=True):

    torch.manual_seed(seed)

    ###############################################
    ##### model ###################################
    ###############################################

    model = GBFCN2_single(dropout=0.2, hidden_layers=32)
    model.to(dev)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0e-4)

    ###############################################
    ##### data ####################################
    ###############################################

    # data embedder

    smallgraphs = SmallGraphs(device=dev)
    graph_1x6l_adj_mat = smallgraphs.load_predefined_line_1x6().to(dtype).to(dev)
    graph_2x3t_adj_mat = smallgraphs.load_predefined_triangle_2x3().to(dtype).to(dev)
    graph_2x3s_adj_mat = smallgraphs.load_predefined_square_2x3().to(dtype).to(dev)
    graph_s1_adj_mat = smallgraphs.load_predefined_triangle_special1().to(dtype).to(dev)
    graph_s2_adj_mat = smallgraphs.load_predefined_squaredc_special2().to(dtype).to(dev)
    graph_adj_mats = [graph_1x6l_adj_mat, graph_s2_adj_mat, graph_2x3s_adj_mat, graph_s1_adj_mat, graph_2x3t_adj_mat]
    graph_names = ['1x6l', 's2', '2x3s', 's1', '2x3t']


    graph_data_embedders = [bw_DataEmbedder(graph_adj_mat, dev=dev, dtype=dtype) for graph_adj_mat in graph_adj_mats]

    data_dir = outer_dir_env + '/../../data_saving/' + data_datetime_path
    # load data
    loaded_datas = [torch.load(os.path.join(data_dir, f"graph_{graph_names[graph_adj_mat_id]}_{gate_type}_data.pt")) for graph_adj_mat_id in range(len(graph_adj_mats))]
    loaded_datas = [(data[0].to(dtype).to(dev), data[1].to(dtype).to(dev)) for data in loaded_datas]
    print("data loaded.", datetime.datetime.now())
    sys.stdout.flush()

    # embed data
    embedded_datas = []
    for graph_adj_mat_id in range(len(graph_adj_mats)):
        graph_data_embedder = graph_data_embedders[graph_adj_mat_id]
        loaded_data = loaded_datas[graph_adj_mat_id]
        
        embedded_data_in = graph_data_embedder.bw_datain_to_modelin_single_gate(loaded_data[0])
        embedded_data_out = graph_data_embedder.bw_dataout_to_modelout_single_gate(loaded_data[1])

        embedded_datas.append((embedded_data_in, embedded_data_out))

    print("data embedded.", datetime.datetime.now())
    sys.stdout.flush()

    # split data
    train_ratio = 0.8
    valid_ratio = 0.1
    test_ratio = 0.1
    embedded_data_size = embedded_datas[0][0].shape[0]
    train_size = int(embedded_data_size * train_ratio)
    trainvalid_size = int(embedded_data_size * (train_ratio + valid_ratio))

    train_datas = [(data[0][:train_size], data[1][:train_size]) for data in embedded_datas]
    valid_datas = [(data[0][train_size:trainvalid_size], data[1][train_size:trainvalid_size]) for data in embedded_datas]
    test_datas = [(data[0][trainvalid_size:], data[1][trainvalid_size:]) for data in embedded_datas]

    ###############################################
    ##### train ###################################
    ###############################################

    MSE_train_list = []
    MSE_valid_list = []
    MSE_valid_s12_list = []
    r2_train_list = []
    r2_valid_list = []
    r2_valid_s12_list = []

    for epoch in range(181):

        # train
        model.train()
        MSE_train_this_epoch = 0
        r2_train_this_epoch = 0

        for i in range(80):

            loss_train_datas = []
            r2_train_datas = []

            for data in train_datas[:-2]:
                model_in = data[0][i*bw_size:(i+1)*bw_size]
                model_out = data[1][i*bw_size:(i+1)*bw_size]

                model_out_pred = model(model_in)

                # loss
                loss_train_data = torch.mean((model_out_pred - model_out).pow(2))
                loss_train_datas.append(loss_train_data)

                # r2
                r2_train_data = r_square(model_out, model_out_pred)
                r2_train_datas.append(r2_train_data)

            loss = torch.mean(torch.stack(loss_train_datas))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            MSE_train_this_epoch += loss.item()
            r2_train_this_epoch += torch.mean(torch.stack(r2_train_datas)).item()

        MSE_train_list.append(MSE_train_this_epoch / 80)
        r2_train_list.append(r2_train_this_epoch / 80)

        # valid
        model.eval()
        MSE_valid_this_epoch = 0
        MSE_valid_s12_this_epoch = 0
        r2_valid_this_epoch = 0
        r2_valid_s12_this_epoch = 0

        for i in range(10):

            loss_valid_datas = []
            r2_valid_datas = []

            for data in valid_datas:
                model_in = data[0][i*bw_size:(i+1)*bw_size]
                model_out = data[1][i*bw_size:(i+1)*bw_size]

                with torch.no_grad():
                    model_out_pred = model(model_in)

                # loss
                loss_valid_data = torch.mean((model_out_pred - model_out).pow(2))
                loss_valid_datas.append(loss_valid_data)

                # r2
                r2_valid_data = r_square(model_out, model_out_pred)
                r2_valid_datas.append(r2_valid_data)

            MSE_valid_this_epoch += torch.mean(torch.stack(loss_valid_datas[:-2])).item()
            r2_valid_this_epoch += torch.mean(torch.stack(r2_valid_datas[:-2])).item()
            MSE_valid_s12_this_epoch += torch.mean(torch.stack(loss_valid_datas[-2:])).item()
            r2_valid_s12_this_epoch += torch.mean(torch.stack(r2_valid_datas[-2:])).item()

        MSE_valid_list.append(MSE_valid_this_epoch / 10)
        MSE_valid_s12_list.append(MSE_valid_s12_this_epoch / 10)
        r2_valid_list.append(r2_valid_this_epoch / 10)
        r2_valid_s12_list.append(r2_valid_s12_this_epoch / 10)

        # lr decay
        if epoch % 100 == 0 and epoch != 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['lr'] * lr_decay

        if print_or_plot_or_savemodel:

            print("epoch: {}, MSE: {:.4f}, {:.4f}, {:.4f}, r2: {:.2f}, {:.2f}, {:.2f}.".format(
                epoch, MSE_train_list[-1], MSE_valid_list[-1], MSE_valid_s12_list[-1], 
                r2_train_list[-1], r2_valid_list[-1], r2_valid_s12_list[-1]), datetime.datetime.now())
            sys.stdout.flush()

            if epoch % 10 == 0:

                # plot
                plt.clf()

                # delta
                plt.subplot(2, 1, 1)
                plt.plot(MSE_train_list, 'b-', label='train')
                plt.plot(MSE_valid_list, 'g-', label='valid')
                plt.plot(MSE_valid_s12_list, 'r-', label='valid_s')
                plt.legend()
                plt.ylabel('MSE')
                plt.yscale('log')
                plt.ylim(0.01, 1)
                plt.title(r'MSE and $r^2$')
                plt.grid()

                # r2
                plt.subplot(2, 1, 2)
                plt.plot(r2_train_list, 'b:', label='train')
                plt.plot(r2_valid_list, 'g:', label='valid')
                plt.plot(r2_valid_s12_list, 'r:', label='valid_s')
                plt.legend()
                plt.ylabel(r'$r^2$')
                plt.ylim(0.8, 1)
                plt.grid()

                plt.savefig(saving_dir + f"delta_r2_{gate_type}.png")

                # save model
                torch.save(model, saving_dir + f"model_{gate_type}.pt")

    # test
    model.eval()
    MSE_test_this_epoch = 0
    MSE_test_s12_this_epoch = 0
    r2_test_this_epoch = 0
    r2_test_s12_this_epoch = 0

    for i in range(10):

        loss_test_datas = []
        r2_test_datas = []

        for data in test_datas:
            model_in = data[0][i*bw_size:(i+1)*bw_size]
            model_out = data[1][i*bw_size:(i+1)*bw_size]

            with torch.no_grad():
                model_out_pred = model(model_in)

            # loss
            loss_test_data = torch.mean((model_out_pred - model_out).pow(2))
            loss_test_datas.append(loss_test_data)

            # r2
            r2_test_data = r_square(model_out, model_out_pred)
            r2_test_datas.append(r2_test_data)

        MSE_test_this_epoch += torch.mean(torch.stack(loss_test_datas[:-2])).item()
        r2_test_this_epoch += torch.mean(torch.stack(r2_test_datas[:-2])).item()
        MSE_test_s12_this_epoch += torch.mean(torch.stack(loss_test_datas[-2:])).item()
        r2_test_s12_this_epoch += torch.mean(torch.stack(r2_test_datas[-2:])).item()

    MSE_test = MSE_test_this_epoch / 10
    MSE_test_s12 = MSE_test_s12_this_epoch / 10
    r2_test = r2_test_this_epoch / 10
    r2_test_s12 = r2_test_s12_this_epoch / 10

    if print_or_plot_or_savemodel:
        print("test: MSE: {:.4f}, {:.4f}, r2: {:.2f}, {:.2f}.".format(
            MSE_test, MSE_test_s12, r2_test, r2_test_s12), datetime.datetime.now())

    return MSE_train_list, MSE_valid_list, MSE_valid_s12_list, r2_train_list, r2_valid_list, r2_valid_s12_list, MSE_test, MSE_test_s12, r2_test, r2_test_s12


if __name__ == '__main__':
    outY = fit_smallgraphs_single(seed=3, gate_type='Y')
    torch.save(outY, saving_dir + "data_outY.pt")

    outX = fit_smallgraphs_single(seed=3, gate_type='X')
    torch.save(outX, saving_dir + "data_outX.pt")
    
    print(outY[-4:])
    print(outX[-4:])

