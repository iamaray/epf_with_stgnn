import torch
import numpy as np
import pandas as pd
from copulas.multivariate import VineCopula


def form_training_pairs(
        data_tensor: torch.Tensor,
        window_hours: int = 504,
        step_hours: int = 24,
        pred_contraction: int = 6,
        split_hour: int = 6888,
        use_copula: bool = False):

    window = window_hours  # hours

    step = step_hours  # hours
    stop = int(split_hour / step)

    # the numbers must work out to make this an integer
    steps = (data_tensor.shape[-1] - window) / step
    curr_steps = 0
    curr_hour = step

    x = []
    y = []
    adj_list = []
    vines = []

    x_train = []
    y_train = []
    tr_adj_list = []

    x_test = []
    y_test = []
    te_adj_list = []

    lag = 9 - step
    vine = VineCopula('regular')

    while curr_steps <= steps:
        next_window = window + step

        # unsqueeze to add a batch size dimension
        x_i = data_tensor[:, :, curr_hour + lag: window + lag].unsqueeze(0)
        y_i = data_tensor[0, :, window: next_window -
                          24 + pred_contraction].unsqueeze(0)

        if y_i.shape[2] == 0:
            break

        if use_copula:
            x_i_corr_data = unnormalized_df[target_signal_names][curr_hour + lag:window + lag]
            mat = np.zeros((x_i.shape[2], x_i.shape[2]))
            try:
                vine.fit(x_i_corr_data)
                mat = vine.tau_mat
            except:
                if curr_steps > 0:
                    mat = vines[-1]

            if mat.all() != 0:
                vines.append(vine.tau_mat)

        x.append(x_i)
        y.append(y_i)

        curr_hour += step
        window = next_window
        curr_steps += 1

    if len(vines) > 0:
        tr_adj_list = [torch.Tensor(v).unsqueeze(0) for v in vines[:stop]]
        te_adj_list = [torch.Tensor(v).unsqueeze(0)
                       for v in vines[stop: len(vines) - 1]]

    x_train, y_train = x[:stop], y[:stop]
    x_test, y_test = x[stop: len(x) - 1], y[stop: len(y) - 1]

    return (torch.cat(x_train, dim=0),
            torch.cat(y_train, dim=0),
            torch.cat(tr_adj_list, dim=0),
            torch.cat(x_test, dim=0),
            torch.cat(y_test, dim=0),
            torch.cat(te_adj_list, dim=0))
