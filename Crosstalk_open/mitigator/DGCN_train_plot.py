import sys, os
currentdir = os.path.dirname(os.path.realpath(__file__))

import torch
import matplotlib.pyplot as plt
import numpy as np

loss_curve_dir = currentdir + '/results/DGCNac3_save_data/'

def plot_data():
    loss_list = torch.load(loss_curve_dir + 'loss_list.pt', map_location=torch.device('cpu'))
    loss_X_list = torch.load(loss_curve_dir + 'loss_X_list.pt', map_location=torch.device('cpu'))
    loss_Y_list = torch.load(loss_curve_dir + 'loss_Y_list.pt', map_location=torch.device('cpu'))
    loss_iSWAP_list = torch.load(loss_curve_dir + 'loss_iSWAP_list.pt', map_location=torch.device('cpu'))

    epoch_num = loss_list.shape[0]
    epoch_list = torch.arange(1, epoch_num + 1)

    epoch_list_log10 = 10 ** torch.linspace(0, np.log10(epoch_num), epoch_num)

    epoch_list = epoch_list.numpy()
    loss_list = loss_list.numpy()
    loss_X_list = loss_X_list.numpy()
    loss_Y_list = loss_Y_list.numpy()
    loss_iSWAP_list = loss_iSWAP_list.numpy()
    epoch_list_log10 = epoch_list_log10.numpy()

    loss_list_log10 = np.interp(epoch_list_log10, epoch_list, loss_list)
    loss_X_list_log10 = np.interp(epoch_list_log10, epoch_list, loss_X_list)
    loss_Y_list_log10 = np.interp(epoch_list_log10, epoch_list, loss_Y_list)
    loss_iSWAP_list_log10 = np.interp(epoch_list_log10, epoch_list, loss_iSWAP_list)

    # save data
    data = (epoch_list_log10, loss_list_log10, loss_X_list_log10, loss_Y_list_log10, loss_iSWAP_list_log10)

    epoch_list = data[0]
    loss_list = data[1]
    loss_X_list = data[2]
    loss_Y_list = data[3]
    loss_iSWAP_list = data[4]

    fig, ax = plt.subplots(figsize=(8, 3))

    ax.plot(epoch_list, loss_list, linewidth=3, label='total loss', color='black')
    ax.plot(epoch_list, loss_X_list, linewidth=2, label='X gate loss', color='blue', alpha=0.5)
    ax.plot(epoch_list, loss_Y_list, linewidth=2, label='Y gate loss', color='green', alpha=0.5)
    ax.plot(epoch_list, loss_iSWAP_list, linewidth=2, label='iSWAP gate loss', color='red', alpha=0.5)

    ax.set_xlabel('Epoch', fontsize=15)
    ax.set_ylabel('Loss', fontsize=15)
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.legend(fontsize=12, loc='upper right')
    ax.set_xlim(1, epoch_num)
    ax.set_ylim(5e-4, 2)

    ax.grid(axis='y', linestyle='--', alpha=1)

    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.spines['top'].set_linewidth(1)
    ax.spines['right'].set_linewidth(1)
    ax.spines['bottom'].set_linewidth(1)
    ax.spines['left'].set_linewidth(1)   

    plt.tight_layout()
    plt.savefig(currentdir + '/results/loss.png')
    print("max plotted epoch:", epoch_list[-10:])


if __name__ == '__main__':
    import time
    start = time.time()
    plot_data()
    print('Plot saved')
    print('Time:', time.time() - start)