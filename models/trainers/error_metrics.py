import numpy as np
import torch


def masked_mae(preds, labels, null_val=np.nan):
    """
    Calculate the Masked Mean Absolute Error (MAE) between predictions and labels.

    Parameters:
    preds (torch.Tensor): Predictions tensor.
    labels (torch.Tensor): Labels tensor.
    null_val (float, optional): Value indicating missing data. Defaults to np.nan.

    Returns:
    torch.Tensor: The mean absolute error after applying the mask.
    """
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds-labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def masked_mape(preds, labels, null_val=np.nan):
    """
    Calculate the Masked Mean Absolute Percentage Error (MAPE) between predictions and labels.

    Parameters:
    preds (torch.Tensor): Predictions tensor.
    labels (torch.Tensor): Labels tensor.
    null_val (float, optional): Value indicating missing data. Defaults to np.nan.

    Returns:
    torch.Tensor: The mean absolute percentage error after applying the mask.
    """
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = (torch.abs(preds-labels)/torch.abs(labels)) * 100
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def masked_mse(preds, labels, null_val=np.nan):
    """
    Calculate the Masked Mean Squared Error (MSE) between predictions and labels.

    Parameters:
    preds (torch.Tensor): Predictions tensor.
    labels (torch.Tensor): Labels tensor.
    null_val (float, optional): Value indicating missing data. Defaults to np.nan.

    Returns:
    torch.Tensor: The mean squared error after applying the mask.
    """
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = (preds-labels)**2
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def calc_err(y_true: np.array, y_pred: np.array, err_func=None, pred_len: int = 24, nodes: int = 3):
    """
    Calculate error metrics for predictions against true values across multiple nodes.

    Parameters:
    y_true (np.array): True values array.
    y_pred (np.array): Predictions array.
    err_func (callable, optional): Error calculation function. Defaults to None.
    pred_len (int, optional): Length of predictions. Defaults to 24.
    nodes (int, optional): Number of nodes to calculate errors for. Defaults to 3.

    Returns:
    list: List of calculated error values for each node.
    """
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
