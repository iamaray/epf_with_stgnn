import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader
from torch_geometric.data import Dataset
from torch_geometric.data import Data

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
            dynamic_adj: bool = True,
            curriculum_type: str = 'pred_len',
            curriculum: np.array = np.array([3, 6, 12, 18, 24]),
            batch_sizes: np.array = np.array([64 for i in range(5)]),
            device=device):

        self.dynamic_adj = dynamic_adj
        # self.adj = adj
        self.curriculum_type = curriculum_type
        self.curriculum = curriculum
        self.batch_sizes = batch_sizes
        self.datasets = []

    def __call__(self, data_tensor: torch.Tensor):
        if self.curriculum_type != 'pred_len':
            raise NotImplementedError('Curriculum type not implemented')
        else:
            # _, in_chans, nodes, seq_len = data_tensor.shape

            for k, level in enumerate(self.curriculum):
                x_train, y_train, tr_adj, x_test, y_test, te_adj = form_training_pairs(
                    data_tensor, pred_contraction=level)

                train_list = []
                test_list = []

                if len(tr_adj) == 0:
                    tr_adj = [None] * len(x_train)
                    te_adj = [None] * len(x_test)

                for i, (x_i, y_i) in enumerate(zip(x_train, y_train)):
                    train_data_i = Data(
                        x=x_i.unsqueeze(0), edge_attr=torch.Tensor(tr_adj[i]).unsqueeze(0), y=torch.transpose(y_i.unsqueeze(0), 1, 2))
                    train_list.append(train_data_i)

                for j, (x_ti, y_ti) in enumerate(zip(x_test, y_test)):
                    test_data_i = Data(
                        x=x_ti.unsqueeze(0), edge_attr=torch.Tensor(te_adj[j]).unsqueeze(0), y=torch.transpose(y_ti.unsqueeze(0), 1, 2))
                    test_list.append(test_data_i)

                train_dataset = MyDataset(train_list)
                test_dataset = MyDataset(test_list)

                gen = torch.Generator(device)
                train_loader = DataLoader(
                    train_dataset, batch_size=int(self.batch_sizes[k]), shuffle=True, generator=gen)
                test_loader = DataLoader(
                    test_dataset, batch_size=int(self.batch_sizes[k]), shuffle=False, generator=gen)

                self.datasets.append((train_loader, test_loader))

            return self.datasets
