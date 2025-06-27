'''
generate the qubits graph
'''

import torch
import math
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
import os

'''
############################################################
######### QuGraphs_specificstructure #######################
############################################################
'''

class QuGraphs_specificstructure:
    @torch.no_grad()
    def __init__(self, basic_x_num, basic_y_num, structure='square', dtype=torch.float32, device=torch.device('cpu')):
        self.basic_x_num = basic_x_num
        self.basic_y_num = basic_y_num
        self.structure = structure
        self.dtype = dtype
        self.device = device
        self.dtype_int = torch.int32

        self.basic_node_num = basic_x_num * basic_y_num
        
        # node ids (basic)
        basic_x_ids = torch.arange(basic_x_num, device=device).reshape(-1, 1).expand(basic_x_num, basic_y_num).reshape(-1, 1)
        basic_y_ids = torch.arange(basic_y_num, device=device).reshape(1, -1).expand(basic_x_num, basic_y_num).reshape(-1, 1)
        self.basic_node_xy_ids = torch.cat([basic_x_ids, basic_y_ids], dim=1)

        # node locations (basic)
        basic_node_xy_locations = self._get_node_xy_locations(self.basic_node_xy_ids)
        # graph structure (basic)
        self.basic_adj_mat = self._get_adj_mat(basic_node_xy_locations)

        # other used variables
        self.upper_triangular_mat = torch.triu(torch.ones(self.basic_node_num, self.basic_node_num, dtype=self.dtype_int, device=self.device), diagonal=1)

    @torch.no_grad()
    def _get_node_xy_locations(self, node_xy_ids):
        node_xy_locations = torch.zeros(node_xy_ids.shape[0], 2, dtype=self.dtype, device=self.device)

        if self.structure == 'square' or self.structure == 'square_dc':
            node_xy_locations[:, 0] = node_xy_ids[:, 0] * 1.0
            node_xy_locations[:, 1] = node_xy_ids[:, 1] * 1.0
        elif self.structure == 'triangle':
            node_xy_locations[:, 0] = node_xy_ids[:, 0] * 1.0 + 0.5 * (node_xy_ids[:, 1] % 2)
            node_xy_locations[:, 1] = node_xy_ids[:, 1] * math.sqrt(3)/2
        else:
            raise ValueError('Invalid structure type.')

        return node_xy_locations
    
    @torch.no_grad()
    def _get_adj_mat(self, basic_node_xy_locations):
        assert basic_node_xy_locations.shape[0] == self.basic_node_num, 'The number of nodes is not consistent.'

        # graph structure (basic)
        basic_node_dx_mat = basic_node_xy_locations[:, 0].reshape(-1, 1) - basic_node_xy_locations[:, 0].reshape(1, -1)
        basic_node_dy_mat = basic_node_xy_locations[:, 1].reshape(-1, 1) - basic_node_xy_locations[:, 1].reshape(1, -1)
        basic_node_dist_mat = torch.sqrt(basic_node_dx_mat.pow(2) + basic_node_dy_mat.pow(2))

        if self.structure == 'square' or self.structure == 'triangle':
            basic_adj_mat = (basic_node_dist_mat < 1.1).to(dtype=self.dtype_int, device=self.device)
        elif self.structure == 'square_dc':
            basic_adj_mat = (basic_node_dist_mat < 1.5).to(dtype=self.dtype_int, device=self.device)
        else:
            raise ValueError('Invalid structure type.')

        basic_adj_mat = basic_adj_mat - torch.diag(torch.diag(basic_adj_mat))

        return basic_adj_mat

    @torch.no_grad()
    def generate_reduced_graph(self, node_keeping_rate, edge_keeping_rate, del_isolated_node=True):
        # reduce by nodes
        if del_isolated_node:
            reduced_node_ids = torch.randperm(self.basic_node_num, device=self.device)[:int(self.basic_node_num * node_keeping_rate)]
            reduced_node_xy_ids = self.basic_node_xy_ids[reduced_node_ids]
            reduced_adj_mat = self.basic_adj_mat[reduced_node_ids][:, reduced_node_ids]
        else:
            reduced_node_ids = torch.randint(0, self.basic_node_num, (int(self.basic_node_num * (1 - node_keeping_rate)),), device=self.device)
            reduced_node_xy_ids = self.basic_node_xy_ids.clone()
            reduced_adj_mat = self.basic_adj_mat.clone()
            reduced_adj_mat[reduced_node_ids, :] = 0
            reduced_adj_mat[:, reduced_node_ids] = 0

        # delete some edges
        reduced_node_num = reduced_node_xy_ids.shape[0]
        reduced_adj_mat = reduced_adj_mat * self.upper_triangular_mat[:reduced_node_num, :reduced_node_num]
        reduced_adj_mat = reduced_adj_mat * (torch.rand(reduced_node_num, reduced_node_num, device=self.device) < edge_keeping_rate).to(dtype=self.dtype_int, device=self.device)
        reduced_adj_mat = reduced_adj_mat + reduced_adj_mat.t()

        # delete isolated nodes
        if del_isolated_node:
            reduced_node_ids = torch.nonzero(reduced_adj_mat.sum(dim=1) > 0, as_tuple=True)[0]
            reduced_node_xy_ids = reduced_node_xy_ids[reduced_node_ids]
            reduced_adj_mat = reduced_adj_mat[reduced_node_ids][:, reduced_node_ids]

        return reduced_node_xy_ids, reduced_adj_mat
    
    @torch.no_grad()
    def bg_generate_reduced_graph(self, bg_size, node_keeping_rate, edge_keeping_rate):
        bg_reduced_adj_mat = self.basic_adj_mat.repeat(bg_size, 1, 1)

        # reduce by nodes
        bg_reduced_node_ids = torch.randint(0, self.basic_node_num, (bg_size, int(self.basic_node_num * (1 - node_keeping_rate))), device=self.device)
        bg_reduced_adj_mat[torch.arange(bg_size, device=self.device).reshape(-1, 1), bg_reduced_node_ids, :] = 0
        bg_reduced_adj_mat[torch.arange(bg_size, device=self.device).reshape(-1, 1), :, bg_reduced_node_ids] = 0

        # delete some edges
        bg_reduced_adj_mat = bg_reduced_adj_mat * self.upper_triangular_mat.repeat(bg_size, 1, 1)
        bg_reduced_adj_mat = bg_reduced_adj_mat *\
              (torch.rand(bg_size, self.basic_node_num, self.basic_node_num, device=self.device) < edge_keeping_rate)\
                .to(dtype=self.dtype_int, device=self.device)
        bg_reduced_adj_mat = bg_reduced_adj_mat + bg_reduced_adj_mat.transpose(1, 2)

        return bg_reduced_adj_mat
    
    @torch.no_grad()
    def get_latticecolor_freqs(self, colors=[0, 0.33, 0.67, 1.0]):
        freqs = torch.zeros(self.basic_node_num, dtype=self.dtype, device=self.device)
        x_ids = self.basic_node_xy_ids[:, 0]
        y_ids = self.basic_node_xy_ids[:, 1]

        color_num = len(colors)
        side_len = int(color_num ** 0.5)
        
        if side_len * side_len != color_num:
            raise ValueError('Invalid number of colors, it should be a perfect square (e.g., 4, 9, 16, 25).')

        for i in range(side_len):
            for j in range(side_len):
                freqs[(x_ids % side_len == i) & (y_ids % side_len == j)] = colors[i*side_len + j]

        return freqs
    
    @torch.no_grad()
    def plot_graph(self, node_xy_ids=None, adj_mat=None, 
                   saving_path=os.path.join(os.path.dirname(__file__), 'temp.png'), 
                   plot_color=False, node_freqs=None, edge_freqs=None,
                   plot_identity_nodes=False):
        if node_xy_ids is None:
            node_xy_ids = self.basic_node_xy_ids
        if adj_mat is None:
            adj_mat = self.basic_adj_mat
        assert adj_mat.dim() == 2
        assert (adj_mat - adj_mat.T).abs().sum() < 1e-5

        if not plot_color:
            assert node_xy_ids.shape[0] == adj_mat.shape[0], 'The number of nodes is not consistent.'

            node_xy_locations = self._get_node_xy_locations(node_xy_ids).detach().cpu()

            plt.rcParams['font.size'] = 20
            fig, ax = plt.subplots(figsize=(self.basic_x_num+2, self.basic_y_num+2))

            # plot edges
            node_num = node_xy_locations.shape[0]
            for i in range(node_num):
                for j in range(i, node_num):
                    if adj_mat[i, j] == 1:
                        x1, y1 = node_xy_locations[i, 0], node_xy_locations[i, 1]
                        x2, y2 = node_xy_locations[j, 0], node_xy_locations[j, 1]
                        dx, dy = x2 - x1, y2 - y1
                        dr = math.sqrt(dx**2 + dy**2)
                        ax.plot([x1+0.25*dx/dr, x2-0.25*dx/dr], [y1+0.25*dy/dr, y2-0.25*dy/dr],
                                color='gray', linewidth=5, zorder=1)
                        
            if not plot_identity_nodes:
                not_identity_node_ids = torch.nonzero(torch.sum(adj_mat, dim=1)>0.1, as_tuple=True)[0].detach().cpu()
                node_xy_locations = node_xy_locations[not_identity_node_ids]

            # plot nodes
            ax.scatter(node_xy_locations[:, 0], node_xy_locations[:, 1], 
                    c='black', s=100, zorder=2)

            # settings
            ax.set_aspect('equal')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_title('Qubits Graph')
            ax.set_xlim(-1, self.basic_x_num)
            ax.set_ylim(-1, self.basic_y_num)

            # color bar
            cmap = cm.gray
            norm = Normalize(vmin=00., vmax=0.99)
            sm = cm.ScalarMappable(norm=norm, cmap=cmap)
            cbar = plt.colorbar(sm, ax=ax, shrink=0.5)
            cbar.set_label('Frequency')

            fig.savefig(saving_path)

            plt.close('all')

        else:
            node_num = node_xy_ids.shape[0]
            edge_num = (adj_mat.sum() / 2).round().int().item()
            if node_freqs is None:
                node_freqs = torch.rand(node_num, device=self.device)
            if edge_freqs is None:
                edge_freqs = torch.rand(edge_num, device=self.device)
            assert node_num == adj_mat.shape[0] == node_freqs.shape[0], 'The number of nodes is not consistent.'
            assert edge_num == edge_freqs.shape[0], 'The number of edges is not consistent.'

            plt.rcParams['font.size'] = 20
            fig, ax = plt.subplots(figsize=(self.basic_x_num+2, self.basic_y_num+2))

            # color map
            cmap = cm.rainbow
            norm = Normalize(vmin=0.0, vmax=1.0)

            node_xy_locations = self._get_node_xy_locations(node_xy_ids).detach().cpu()
            # plot edges
            edge_id = 0
            for i in range(node_num):
                for j in range(i, node_num):
                    if adj_mat[i, j] == 1:
                        x1, y1 = node_xy_locations[i, 0], node_xy_locations[i, 1]
                        x2, y2 = node_xy_locations[j, 0], node_xy_locations[j, 1]
                        dx, dy = x2 - x1, y2 - y1
                        dr = math.sqrt(dx**2 + dy**2)
                        ax.plot([x1+0.25*dx/dr, x2-0.25*dx/dr], [y1+0.25*dy/dr, y2-0.25*dy/dr],
                                c=cmap(norm(edge_freqs[edge_id].item())),
                                linewidth=5, zorder=1)
                        edge_id += 1

            # plot nodes
            if not plot_identity_nodes:
                not_identity_node_ids = torch.nonzero(torch.sum(adj_mat, dim=1)>0.1, as_tuple=True)[0].detach().cpu()
                node_xy_locations = node_xy_locations[not_identity_node_ids]
                node_freqs = node_freqs[not_identity_node_ids]

            ax.scatter(node_xy_locations[:, 0], node_xy_locations[:, 1], 
                    c=cmap(norm(node_freqs.detach().cpu())),
                    s=100, zorder=2)

            # settings
            ax.set_aspect('equal')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_title('Qubits Graph')
            ax.set_xlim(-1, self.basic_x_num)
            ax.set_ylim(-1, self.basic_y_num)

            # color bar
            sm = cm.ScalarMappable(norm=norm, cmap=cmap)
            cbar = plt.colorbar(sm, ax=ax, shrink=0.5)
            cbar.set_label('Frequency')

            # save
            fig.savefig(saving_path)

            # cbar.remove()
            # ax.cla()
            plt.close('all')

############################################################
######### test QuGraphs_specificstructure ##################
############################################################

def test_specificstructure():
    qg = QuGraphs_specificstructure(8, 8, structure='square_dc')


    reduced_node_xy_ids, reduced_adj_mat = qg.generate_reduced_graph(0.85, 0.85, del_isolated_node=False)
    qg.plot_graph(node_xy_ids=reduced_node_xy_ids, adj_mat=reduced_adj_mat, plot_identity_nodes=False,
                  saving_path=os.path.join(os.path.dirname(__file__), 'color_graph.png'), plot_color=True)
    qg.plot_graph(node_xy_ids=reduced_node_xy_ids, adj_mat=reduced_adj_mat, plot_identity_nodes=False,
                  saving_path=os.path.join(os.path.dirname(__file__), 'black_graph.png'), plot_color=False)

    print('done')

def test_specificstructure_large():
    import time
    start_time = time.time()
    qg = QuGraphs_specificstructure(32, 32, structure='square')


    reduced_node_xy_ids, reduced_adj_mat = qg.generate_reduced_graph(0.85, 0.85, del_isolated_node=False)
    qg.plot_graph(node_xy_ids=reduced_node_xy_ids, adj_mat=reduced_adj_mat, plot_identity_nodes=False,
                  saving_path=os.path.join(os.path.dirname(__file__), 'large_color_graph.png'), plot_color=True)
    qg.plot_graph(node_xy_ids=reduced_node_xy_ids, adj_mat=reduced_adj_mat, plot_identity_nodes=False,
                  saving_path=os.path.join(os.path.dirname(__file__), 'large_black_graph.png'), plot_color=False)

    print('time:', time.time()-start_time, 's')

'''
############################################################
######### QuGraphs_list ####################################
############################################################
'''

class QuGraph_list:
    @torch.no_grad()
    def __init__(self, basic_x_num, basic_y_num, dtype=torch.float32, device=torch.device('cpu')) -> None:
        self.basic_x_num = basic_x_num
        self.basic_y_num = basic_y_num
        self.dtype = dtype
        self.device = device
        self.dtype_int = torch.int32

        self.basic_node_num = basic_x_num * basic_y_num

        self.qg_list = [QuGraphs_specificstructure(basic_x_num, basic_y_num, structure='square', dtype=dtype, device=device),
                        QuGraphs_specificstructure(basic_x_num, basic_y_num, structure='triangle', dtype=dtype, device=device),
                        QuGraphs_specificstructure(basic_x_num, basic_y_num, structure='square_dc', dtype=dtype, device=device)]
        self.qg_num = len(self.qg_list)
        
    @torch.no_grad()
    def bg_generate_reduced_graph(self, bg_size, node_keeping_rate, edge_keeping_rate):
        bg_reduced_adj_mat_list = [self.qg_list[i].bg_generate_reduced_graph(bg_size, node_keeping_rate, edge_keeping_rate) for i in range(self.qg_num)]
        return torch.cat(bg_reduced_adj_mat_list, dim=0)
    
    @torch.no_grad()
    def plot_graph(self, adj_mat=None, 
                   saving_path=os.path.join(os.path.dirname(__file__), 'temp.png'), 
                   plot_color=False, node_freqs=None, edge_freqs=None,
                   plot_identity_nodes=False):
        assert adj_mat.dim() == 3
        assert adj_mat.shape[0] == self.qg_num

        Q_num_list = [self.qg_list[i].basic_node_num for i in range(self.qg_num)]
        E_num_list = [round((adj_mat[i] > 0.1).sum().item()/2) for i in range(self.qg_num)]

        if node_freqs is None:
            node_freqs = torch.rand(sum(Q_num_list), device=self.device)
        if edge_freqs is None:
            edge_freqs = torch.rand(sum(E_num_list), device=self.device)
        assert node_freqs.shape == (sum(Q_num_list),)
        assert edge_freqs.shape == (sum(E_num_list),)

        node_freqs_list = [node_freqs[:Q_num_list[0]]]
        edge_freqs_list = [edge_freqs[:E_num_list[0]]]
        for i in range(1, self.qg_num):
            node_freqs_list.append(node_freqs[sum(Q_num_list[:i]):sum(Q_num_list[:i+1])])
            edge_freqs_list.append(edge_freqs[sum(E_num_list[:i]):sum(E_num_list[:i+1])])

        extension_name = '.' + saving_path.split('.')[-1]
        for gq_id in range(self.qg_num):
            self.qg_list[gq_id].plot_graph(node_xy_ids=None, adj_mat=adj_mat[gq_id], 
                                           saving_path=saving_path.replace(extension_name, f'_{gq_id}'+extension_name), 
                                           plot_color=plot_color, node_freqs=node_freqs_list[gq_id], edge_freqs=edge_freqs_list[gq_id],
                                           plot_identity_nodes=plot_identity_nodes)
        
############################################################
######### test QuGraphs_list ###############################
############################################################

def test_qglist():
    qg = QuGraph_list(8, 8)

    reduced_adj_mat = qg.bg_generate_reduced_graph(1, 0.85, 0.85)
    qg.plot_graph(adj_mat=reduced_adj_mat, plot_identity_nodes=False,
                  saving_path=os.path.join(os.path.dirname(__file__), 'color_graph.png'), plot_color=True)
    qg.plot_graph(adj_mat=reduced_adj_mat, plot_identity_nodes=False,
                  saving_path=os.path.join(os.path.dirname(__file__), 'black_graph.png'), plot_color=False)

    print('done')

if __name__ == "__main__":
    test_specificstructure_large()