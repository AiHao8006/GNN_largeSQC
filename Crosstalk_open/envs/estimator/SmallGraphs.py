import torch
import math
import matplotlib.pyplot as plt

class SmallGraphs:
    @torch.no_grad()
    def __init__(self, device=torch.device('cpu')):
        self.device = device
        self.dtype_int = torch.int32

        # dynamically changing params
        self.Q_num = 3
        self.adj_mat = torch.zeros(self.Q_num, self.Q_num, dtype=torch.int32, device=self.device)

    @torch.no_grad()
    def add_coup(self, Qid1, Qid2):
        self.adj_mat[Qid1, Qid2] = 1
        self.adj_mat[Qid2, Qid1] = 1

    @torch.no_grad()
    def add_coup_by_list(self, coup_list):
        for coup in coup_list:
            self.add_coup(coup[0], coup[1])

    @torch.no_grad()
    def load_predefined_line_1x7(self):
        self.Q_num = 7
        self.adj_mat = torch.zeros(self.Q_num, self.Q_num, dtype=torch.int32, device=self.device)
        self.add_coup_by_list([[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])
        return self.adj_mat
    
    @torch.no_grad()
    def load_predefined_triangle_2x4(self):
        self.Q_num = 8
        self.adj_mat = torch.zeros(self.Q_num, self.Q_num, dtype=torch.int32, device=self.device)
        self.add_coup_by_list([[0, 1], [1, 2], [2, 3],
                               [0, 4], [1, 5], [2, 6], [3, 7],
                               [1, 4], [2, 5], [3, 6],
                               [4, 5], [5, 6], [6, 7]])
        return self.adj_mat
    
    @torch.no_grad()
    def load_predefined_triangle_3x3(self):
        self.Q_num = 9
        self.adj_mat = torch.zeros(self.Q_num, self.Q_num, dtype=torch.int32, device=self.device)
        self.add_coup_by_list([[0, 1], [1, 2],
                               [0, 3], [1, 4], [2, 5],
                               [1, 3], [2, 4],
                               [3, 4], [4, 5],
                               [3, 6], [4, 7], [5, 8],
                               [3, 7], [4, 8],
                               [6, 7], [7, 8]])
        return self.adj_mat
    
    @torch.no_grad()
    def load_predefined_triangle_special1(self):
        self.Q_num = 6
        self.adj_mat = torch.zeros(self.Q_num, self.Q_num, dtype=torch.int32, device=self.device)
        self.add_coup_by_list([[0, 1], [1, 2], [2, 3],
                               [1, 5], [2, 5], [1, 4], [2, 4]])
        return self.adj_mat

    # predefined graphs in 2402

    @torch.no_grad()
    def load_predefined_line_1x6(self):
        self.Q_num = 6
        self.adj_mat = torch.zeros(self.Q_num, self.Q_num, dtype=torch.int32, device=self.device)
        self.add_coup_by_list([[0, 1], [1, 2], [2, 3], [3, 4], [4, 5]])
        return self.adj_mat
    
    @torch.no_grad()
    def load_predefined_triangle_2x3(self):
        self.Q_num = 6
        self.adj_mat = torch.zeros(self.Q_num, self.Q_num, dtype=torch.int32, device=self.device)
        self.add_coup_by_list([[0, 1], [1, 2],
                               [0, 3], [1, 4], [2, 5],
                            #    [1, 3], [2, 4],
                               [0, 4], [1, 5],
                               [3, 4], [4, 5]])
        return self.adj_mat
    
    @torch.no_grad()
    def load_predefined_square_2x3(self):
        self.Q_num = 6
        self.adj_mat = torch.zeros(self.Q_num, self.Q_num, dtype=torch.int32, device=self.device)
        self.add_coup_by_list([[0, 1], [1, 2], 
                               [3, 4], [4, 5],
                               [0, 3], [1, 4], [2, 5]])
        return self.adj_mat
    
    # predefined graphs in 2406

    @torch.no_grad()
    def load_predefined_squaredc_special2(self):
        self.Q_num = 6
        self.adj_mat = torch.zeros(self.Q_num, self.Q_num, dtype=torch.int32, device=self.device)
        self.add_coup_by_list([[0, 1], [1, 2], [3, 4], [4, 5],
                               [0, 3], [1, 4], [2, 5],
                               [0, 4], [1, 5],
                               [1, 3], [2, 4]])
        return self.adj_mat
    
    # add in 2504

    @torch.no_grad()
    def load_predefined_line_1x30(self):
        self.Q_num = 30
        self.adj_mat = torch.zeros(self.Q_num, self.Q_num, dtype=torch.int32, device=self.device)
        self.add_coup_by_list([[i, i+1] for i in range(self.Q_num-1)])
        return self.adj_mat
    
    @torch.no_grad()
    def load_predefined_line_1x16p14(self):
        self.Q_num = 30
        self.adj_mat = torch.zeros(self.Q_num, self.Q_num, dtype=torch.int32, device=self.device)
        self.add_coup_by_list([[i, i+1] for i in range(16)])
        for i in range(16, 30):
            self.add_coup(i-15, i)      # (1, 16), (2, 17), (3, 18), ..., (14, 29)
        return self.adj_mat