import torch
import torch.nn as nn
import numpy as np


def masked_mae(preds, labels, null_val=np.nan, quantile=0):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    quantiles = None
    if len(labels.shape) > 2:
        quantiles = torch.quantile(labels, q=quantile, dim=1)
    else:
        quantiles = torch.quantile(labels, q=quantile, dim=-1)
    quantile_mask = None
    # print(labels.shape, quantiles.shape)
    if len(labels.shape) > 1:
        quantile_mask = (labels >= quantiles.unsqueeze(1))
    else:
        quantile_mask = (labels >= quantiles)
    # if len(quantiles.shape) > 2:
        # quantile_mask = (labels >= quantiles.unsqueeze(1))
    # else:
        # print(labels.shape, quantiles.unsqueeze(1).shape)
        # quantile_mask = (labels >= quantiles.unsqueeze(1))
    quantile_mask = quantile_mask.float()
    mask = mask.float()
    mask = mask * quantile_mask
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


def masked_avg_loc_diff(preds, labels, loc_loss=masked_mae, null_val=np.nan, grad=True, beta=0.5, quantile=0.0):
    locs = np.arange(preds.shape[-1])
    all_loc_pairs = [(a, b) for idx, a in enumerate(locs)
                     for b in locs[idx + 1:]]

    mae_lst = []
    for (i, j) in all_loc_pairs:
        pred_loc_diff = preds[..., i] - preds[..., j]
        label_loc_diff = labels[..., i] - labels[..., j]
        mae_lst.append(
            (loc_loss(pred_loc_diff, label_loc_diff, null_val, quantile=quantile)))

    mae_lst = torch.Tensor(mae_lst)
    ret = beta * torch.mean(mae_lst)
    ret.requires_grad = grad
    return ret


def conditional_mae_gte_quantile(model_output: torch.Tensor, actuals: torch.Tensor, quantile: float) -> torch.Tensor:
    """
    Calculate the conditional MAE restricted to the values greater than or equal to a specified quantile,
    ignoring NaN values.

    Parameters:
    model_output (torch.Tensor): The predicted values from the model.
    actuals (torch.Tensor): The ground truth values.
    quantile (float): The quantile threshold (e.g., 0.5 for the 50th percentile).

    Returns:
    torch.Tensor: The conditional MAE for the values greater than or equal to the specified quantile.
    """
    not_nan_mask = ~torch.isnan(model_output) & ~torch.isnan(actuals)

    filtered_model_output = model_output[not_nan_mask]
    filtered_actuals = actuals[not_nan_mask]

    threshold = torch.quantile(filtered_actuals, quantile)

    quantile_mask = filtered_actuals >= threshold

    masked_mae = torch.abs(
        filtered_model_output[quantile_mask] - filtered_actuals[quantile_mask]).mean()

    return masked_mae


def conditional_mae_lte_quantile(model_output: torch.Tensor, actuals: torch.Tensor, quantile: float) -> torch.Tensor:
    """
    Calculate the conditional MAE restricted to the values less than or equal to a specified quantile,
    ignoring NaN values.

    Parameters:
    model_output (torch.Tensor): The predicted values from the model.
    actuals (torch.Tensor): The ground truth values.
    quantile (float): The quantile threshold (e.g., 0.5 for the 50th percentile).

    Returns:
    torch.Tensor: The conditional MAE for the values less than or equal to the specified quantile.
    """
    not_nan_mask = ~torch.isnan(model_output) & ~torch.isnan(actuals)

    filtered_model_output = model_output[not_nan_mask]
    filtered_actuals = actuals[not_nan_mask]

    threshold = torch.quantile(filtered_actuals, quantile)
    quantile_mask = filtered_actuals <= threshold

    masked_mae = torch.abs(
        filtered_model_output[quantile_mask] - filtered_actuals[quantile_mask]).mean()

    return masked_mae


class LaplacianRegularization(nn.Module):
    def __init__(self, laplacian_matrix):
        super(LaplacianRegularization, self).__init__()
        self.L = laplacian_matrix

    def forward(self, y):
        """
        y: Tensor of shape (batch_size, prediction_sequence_length, num_nodes)
                        These are the forecasted LMPs over a sequence of time steps.
        """
        batch_size, seq_length, num_nodes = y.shape
        predicted_LMPs_reshaped = y.view(-1, num_nodes)

        L_predicted = torch.matmul(self.L, predicted_LMPs_reshaped.T)
        reg_term = torch.sum(predicted_LMPs_reshaped.T * L_predicted)

        return reg_term


def calc_err(
        y_true: np.array,
        y_pred: np.array,
        err_func: None,
        pred_len: int = 24,
        nodes: int = 3,
        quantile=0.0):
    err_vals = []

    func = None
    if err_func == 'mape':
        func = masked_mape
    elif err_func == 'mse':
        func = masked_mse
    elif err_func == 'mae':
        func = masked_mae
    elif err_func == 'loc':
        func = masked_avg_loc_diff
    elif err_func == 'gte_mae':
        func = conditional_mae_gte_quantile
    elif err_func == 'lte_mae':
        func = conditional_mae_lte_quantile
    else:
        func = err_func

    for n in range(nodes):
        y_true_n = y_true[:, n]
        y_pred_n = y_pred[:, n]

        if err_func == 'loc':
            err_n = func(y_pred_n, y_true_n, grad=False, quantile=quantile)
        elif err_func == 'gte_mae' or err_func == 'lte_mae':
            err_n = func(y_pred_n, y_true_n, quantile=quantile)
        else:
            err_n = func(y_pred_n, y_true_n, quantile=quantile)
        err_vals.append(float(err_n))

    return err_vals
