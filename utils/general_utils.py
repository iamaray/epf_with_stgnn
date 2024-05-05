import torch


def compute_a_tilde(adj):
    rsum = torch.sum(adj, -1)
    d = 1 + rsum
    d_inv = torch.pow(d, -1)
    d_inv[torch.isinf(d_inv)] = 0.
    d_mat_inv = torch.diagflat(d_inv)
    adj_plus_I = adj + torch.eye(adj.shape[0])
    return torch.matmul(d_mat_inv, adj_plus_I)
