import torch
import torch.nn as nn
import numpy as np
# import pyinform


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


class ContinuityLoss(nn.Module):
    def __init__(self, beta):
        super(ContinuityLoss, self).__init__()
        self.beta = beta
        self.prod = batch_const_mul()

    def forward(self, x):
        num_ats = x.shape[1]
        seq_len = x.shape[2]
        scale = self.beta / (num_ats * seq_len)

        variates = torch.split(x, split_size_or_sections=1, dim=1)
        stds = []

        for v in variates:
            stds.append(torch.std(v, dim=-1))

        stds = torch.cat(stds, dim=-1)
        stds = 1 / stds
        offset = x[:, :, 1:] - x[:, :, :seq_len-1]
        # compute the sum
        batch_loss = scale * torch.sum(  # sum along sequence length
            torch.sum(                    # sum along channel dimension
                torch.pow(                # square the inside
                    self.prod(stds, offset), 2), dim=1), dim=-1)

        return torch.mean(batch_loss)


class GraphContinuityLoss(nn.Module):
    def __init__(self, beta):
        super(GraphContinuityLoss, self).__init__()
        self.variate_loss = ContinuityLoss(beta)

    def forward(self, x):
        agg_loss = 0.0
        vars = [v.squeeze(2) for v in torch.split(
            x, split_size_or_sections=1, dim=2)]

        for v in vars:
            agg_loss += self.variate_loss(v)

        return (1 / x.shape[2]) * agg_loss


# class TransferEntropyLoss(nn.Module):
#     def __init__(self, beta, history_len):
#         super(TransferEntropyLoss, self).__init__()
#         self.history_len = history_len
#         self.beta = beta

#     def forward(self, x):
#         scale = self.beta * (1 / (x.shape[1] * x.shape[2] * x.shape[3]))
#         aggs = []
#         variates = [v[:, :, 0, :]
#                     for v in torch.split(x, split_size_or_sections=1, dim=2)]
#         # print(variates[0].shape)
#         for i, v in enumerate(variates):
#             curr_agg = 0

#             for j in range(1, v.shape[1]):
#                 v_curr = v[:, 0, :].cpu().detach().numpy()
#                 xs = v[:, j, :].cpu().detach().numpy()
#                 # ws = torch.cat(torch.split(torch.cat([v[:, 1:j, :], v[:, j+1:, :]], dim=1), split_size_or_sections=1, dim=1), dim=1).transpose(0,1).detach().numpy()
#                 # print(ws.shape)
#                 curr_agg += transfer_entropy(xs, v_curr, k=self.history_len)

#             aggs.append(curr_agg)

#         return scale * np.mean(aggs)
