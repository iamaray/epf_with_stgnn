import numpy as np
import torch


def masked_mae(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds-labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def masked_mape(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = (torch.abs(preds-labels)/torch.abs(labels)) * 100
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def masked_mse(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = (preds-labels)**2
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def calc_err(
        y_true: np.array,
        y_pred: np.array,
        err_func: None,
        pred_len: int = 24,
        nodes: int = 3):
    err_vals = []

    func = None
    if err_func == 'mape':
        func = masked_mape
    elif err_func == 'mse':
        func = masked_mse
    elif err_func == 'mae':
        func = masked_mae
    else:
        func = err_func

    for n in range(nodes):
        y_true_n = y_true[:, n]
        y_pred_n = y_pred[:, n]

        err_n = func(y_true_n, y_pred_n)
        err_vals.append(float(err_n))

    return err_vals
