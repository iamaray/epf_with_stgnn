import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from .processing_classes import *
from .transformations_funcs import *


class PreprocessData:
    """
        Class to coordinate preprocessing operations on data
    """

    def __init__(
        self,
        generate_time_dummies: bool = True,
        quantile_cutoff: float = 1.0,
        op_list: Pipeline = Pipeline([])
    ):

        self.generate_time_dummies = generate_time_dummies
        self.quantile_cutoff = quantile_cutoff
        self.op_list = op_list

    def __call__(
            self,
            csv_path: str,
            # { target signals } --> { auxiliary signals }
            desired_feats: dict):

        ret = []
        op_name_history = []
        target_signals = desired_feats.keys()

        # convert to DataFrame -> DF_0
        orig_data = pd.read_csv(csv_path)

        ret.append(orig_data)
        op_name_history.append('to_df')

        # datetime-index DataFrame -> DF_1
        date_time_indexed_data = ret[-1].copy(deep=True)
        date_time_indexed_data['Unnamed: 0'] = pd.to_datetime(
            date_time_indexed_data['Unnamed: 0'])

        date_time_indexed_data.set_index('Unnamed: 0', inplace=True)
        ret.append(date_time_indexed_data)
        op_name_history.append('date_time_indexed')

        # if generate_time_dummies True, time dummy DataFrame -> DF_2
        dummy_var_data = None
        days = []
        hours = []
        if self.generate_time_dummies:
            dummy_var_data = ret[-1].copy(deep=True)

            # daily dummy vars
            for i, x in enumerate(days):
                dummy_var_data[x] = (
                    dummy_var_data.index.get_level_values(0).weekday == i).astype(int)

            # hourly dummy vars
            for i, x in enumerate(hours):
                dummy_var_data[x] = (
                    dummy_var_data.index.get_level_values(0).hour == i).astype(int)

        if not dummy_var_data.empty:
            ret.append(dummy_var_data)
            op_name_history.append('dummy_variables_computed')

        # truncate data -> DF_3 := T(DF_2)
        cutoff_data = None
        if self.quantile_cutoff < 1:
            cutoff_data = ret[-1].copy(deep=True)
            for name in target_signals:
                low = np.quantile(cutoff_data[name], 0.0)
                high = np.quantile(cutoff_data[name], self.quantile_cutoff)
                cutoff_data = cutoff_data[cutoff_data[name].between(low, high)]

            if not cutoff_data.empty:
                ret.append(cutoff_data)
                op_name_history.append(
                    f'truncated_{self.quantile_cutoff}_quantile')

        # desired_cols = list(
        #     set(ret[-1].columns.values.tolist()) - set(days) - set(hours))

        self.op_list.fit(ret[-1])
        self.op_list(ret[-1])
        # ret.extend([op.ret[-1] for op in self.op_list.seq])
        # op_name_history.extend([op.op_history[-1] for op in self.op_list.seq])

        # ret_dict = self.op_list.history(op_name_history, ret)

        # return DF_0, ..., DF_5
        # return ret_dict

    def extract_feats_to_tensor(
            self,
            data: pd.DataFrame,
            desired_feats: dict):
        target_signal_names = desired_feats.keys()
        feats = []
        dummies = []
        for name in data.columns.values.tolist():
            if 'DUMMY' in name:
                dummies.append(name)

        for name in target_signal_names:
            curr_feat_names = [name]
            curr_feat_names += desired_feats[name]
            curr_feat_names += dummies

            curr_feats = torch.unsqueeze(
                torch.transpose(
                    torch.Tensor(data[curr_feat_names].to_numpy()), 0, 1), 0)

            feats.append(curr_feats)

        feats = torch.transpose(torch.cat(feats, 0), 0, 1)
        return feats
