import torch
import numpy as np
import pandas as pd


def form_training_pairs(
        data_tensor: torch.Tensor,
        window_hours: int = 504,
        step_hours: int = 24,
        pred_contraction: int = 6,
        split_hour: int = 6888):
    # The data is composed of hourly data covering 434 days.

    # The training pairs (x,y) are formed by taking the x to be
    # a week's worth of data and y to be the preceeding day's worth of
    # day-ahead price data. I.e., x is a [9, 3, 168] tensor, and y is a [24, 3]
    # tensor.

    window = window_hours  # hours
    # start = window + 24

    step = step_hours  # hours
    stop = int(split_hour / step)
    # the numbers must work out to make this an integer
    steps = (data_tensor.shape[-1] - window) / step
    curr_steps = 0
    curr_hour = step

    x = []
    y = []

    x_train = []
    y_train = []

    x_test = []
    y_test = []

    lag = 9 - step
    while curr_steps <= steps:

        next_window = window + step

        # unsqueeze to add a batch size dimension
        x_i = data_tensor[:, :, curr_hour + lag: window + lag].unsqueeze(0)
        y_i = data_tensor[0, :, window: next_window -
                          24 + pred_contraction].unsqueeze(0)
        # print(unnormalized_df.index[curr_hour + lag])
        # print(unnormalized_df.index[window + lag])
        # print(y_i.shape)
        # print('\n')

        # print(unnormalized_df.index[window])
        # print(unnormalized_df.index[next_window - 24 + pred_contraction])

        # print('\n')

        if y_i.shape[2] == 0:
            break
        x.append(x_i)
        y.append(y_i)

        curr_hour += step
        window = next_window
        curr_steps += 1

    x_train, y_train = x[:stop], y[:stop]
    x_test, y_test = x[stop: len(x) - 1], y[stop: len(y) - 1]

    return (torch.cat(x_train, dim=0),
            torch.cat(y_train, dim=0),
            torch.cat(x_test, dim=0),
            torch.cat(y_test, dim=0))
