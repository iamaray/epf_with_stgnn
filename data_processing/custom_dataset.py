import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader
from torch_geometric.data import Dataset
from torch_geometric.data import Data


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
        curriculum: np.array = np.array([3, 6, 12, 18, 24]),
        # possibly varying batch sizes
            batch_sizes: np.array = np.array([32 for i in range(5)])):
        self.curriculum_type = curriculum_type
        self.curriculum = curriculum
        self.batch_sizes = batch_sizes
        self.datasets = []

    def __call__(self, data_tensor: torch.Tensor):
        if self.curriculum_type != 'pred_len':
            raise NotImplementedError('Curriculum type not implemented')
        else:
            # _, in_chans, nodes, seq_len = data_tensor.shape

            for i, level in enumerate(self.curriculum):
                x_train, y_train, x_test, y_test = form_training_pairs(
                    data_tensor, pred_contraction=level)

                train_list = []
                test_list = []

                for x_i, y_i in zip(x_train, y_train):
                    train_data_i = Data(
                        x=x_i.unsqueeze(0), y=torch.transpose(y_i.unsqueeze(0), 1, 2))
                    train_list.append(train_data_i)

                for x_ti, y_ti in zip(x_test, y_test):
                    test_data_i = Data(
                        x=x_ti.unsqueeze(0), y=torch.transpose(y_ti.unsqueeze(0), 1, 2))
                    test_list.append(test_data_i)

                train_dataset = MyDataset(train_list)
                test_dataset = MyDataset(test_list)

                gen = torch.Generator('cpu')

                if torch.cuda.is_available():
                    gen = torch.Generator('cuda')

                train_loader = DataLoader(
                    train_dataset, batch_size=int(self.batch_sizes[i]), shuffle=True, generator=gen)
                test_loader = DataLoader(
                    test_dataset, batch_size=int(self.batch_sizes[i]), shuffle=False, generator=gen)

                self.datasets.append((train_loader, test_loader))

            return self.datasets
