import torch


def compute_a_tilde(adj):
    rsum = torch.sum(adj, -1)
    d = 1 + rsum
    d_inv = torch.pow(d, -1)
    d_inv[torch.isinf(d_inv)] = 0.

    d_mat_inv = []
    for d_inv_i in d_inv:
        d_mat_inv.append(torch.diagflat(d_inv_i).unsqueeze(0))

    d_mat_inv = torch.cat(d_mat_inv)
    adj_plus_I = adj + torch.eye(adj.shape[1])
    result = torch.einsum('bvw,bvw->bvw', (d_mat_inv, adj_plus_I))

    return result

def compute_a_tilde_dy(adj):
  d_mat_invs = []
  adj_plus_I = []
  for t in range(adj.shape[-1]):
    adj_t = adj[:, :, :, t]
    rsum = torch.sum(adj_t, -1)
    d = 1 + rsum
    d_inv = torch.pow(d, -1)
    d_inv[torch.isinf(d_inv)] = 0.

    d_mat_inv = []
    for d_inv_i in d_inv:
      d_mat_inv.append(torch.diagflat(d_inv_i).unsqueeze(0))

    d_mat_inv = torch.cat(d_mat_inv)
    d_mat_invs.append(d_mat_inv)
    # print(adj_t.shape)
    curr = adj_t + torch.eye(adj_t.shape[-1])
    adj_plus_I.append(curr)

  adj_plus_I = torch.stack(adj_plus_I)
  d_mat_invs = torch.stack(d_mat_invs)
  # print(adj_plus_I.shape)
  result = torch.einsum('bcvw,bcvw->bcvw', (d_mat_invs, adj_plus_I))
  # print(result.shape)
  return result.permute(1,2,3,0)
