import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.autograd import Variable
import math
device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")


# class Transform(nn.Module):
#     def __init__(self):
#         super(Transform, self).__init__()
#         pass

#     def forward(self):
#         pass


# class PositionalEncoding(nn.Module):
#     def __init__(self):
#         super(PositionalEncoding, self).__init__()
#         pass

#     def forward(self):
#         pass


# class SGNN(nn.Module):
#     def __init__(self):
#         super(SGNN, self).__init__()
#         pass

#     def forward(self):
#         pass


# class GRU(nn.Module):
#     def __init__(self):
#         super(GRU, self).__init__()
#         pass


# class STGNNwithGRU(nn.Module):
#     def __init__(self, outfea):
#         super(STGNNwithGRU, self).__init__()
#         pass

#     def forward(self):
#         pass


class STGNN(nn.Module):
    def __init__(self, infea, outfea, L, d):
        super(STGNN, self).__init__()
        pass

    def forward(self):
        pass
