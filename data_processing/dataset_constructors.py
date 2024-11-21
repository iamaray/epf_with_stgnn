import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader
from torch_geometric.data import Dataset
from torch_geometric.data import Data

from data_processing.construct_pairs import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class MyDataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list

    def __getitem__(self, index):
        return self.data_list[index]

    def __len__(self):
        return len(self.data_list)


class DatasetConstructor:
    """
    Specifies variables and operations to construct
    different datasets.
    """

    def __init__(
            self,
            # the type of curriculum imposed over the datset
            curriculum_type: str = 'pred_len',
            # an array denoting the difficulty levels of the curriculum
            curriculum: np.array = np.array([6, 12, 18, 24]),
            # possibly varying batch sizes
            batch_size: int = 32,
            copula_adj: bool = False,
            window_hours: int = 168,
            step_hours: int = 24,
            split_hour: int = 3953):

        self.copula_adj = copula_adj
        self.curriculum_type = curriculum_type
        self.curriculum = curriculum
        self.batch_size = batch_size
        self.window_hours = window_hours
        self.step_hours = step_hours
        self.split_hour = split_hour

        self.datasets = []

    def __call__(self, data_tensor: torch.Tensor):
        if self.curriculum_type != 'pred_len':
            raise NotImplementedError('Curriculum type not implemented')
        else:
            # _, in_chans, nodes, seq_len = data_tensor.shape

            for i, level in enumerate(self.curriculum):
                x_train, y_train, tr_adj_lst, x_test, y_test, te_adj_lst = form_training_pairs(
                    data_tensor=data_tensor,
                    pred_contraction=level,
                    copula_adj=self.copula_adj,
                    window_hours=self.window_hours,
                    step_hours=self.step_hours)

            tr_pair_lst = zip(x_train, y_train, [
                              None for _ in range(len(x_train))])
            te_pair_lst = zip(
                x_test, y_test, [None for _ in range(len(x_test))])

            if tr_adj_lst != None:
                tr_pair_lst = zip(x_train, y_train, tr_adj_lst)
                te_pair_lst = zip(x_test, y_test, te_adj_lst)

            train_list = []
            test_list = []

            for x_i, y_i, tr_adj_i in tr_pair_lst:
                train_data_i = Data(
                    x=x_i.unsqueeze(0), edge_attr=tr_adj_i, y=torch.transpose(y_i.unsqueeze(0), 1, 2))
                train_list.append(train_data_i)

            for x_ti, y_ti, te_adj_i in te_pair_lst:
                test_data_i = Data(
                    x=x_ti.unsqueeze(0), edge_attr=te_adj_i, y=torch.transpose(y_ti.unsqueeze(0), 1, 2))
                test_list.append(test_data_i)

            train_dataset = MyDataset(train_list)
            test_dataset = MyDataset(test_list)

            gen = torch.Generator('cpu')

            if torch.cuda.is_available():
                gen = torch.Generator('cuda')

            train_loader = DataLoader(
                train_dataset, batch_size=self.batch_size, shuffle=True, generator=gen)
            test_loader = DataLoader(
                test_dataset, batch_size=int(self.batch_size), shuffle=False, generator=gen)

            self.datasets.append((train_loader, test_loader))

            return self.datasets
