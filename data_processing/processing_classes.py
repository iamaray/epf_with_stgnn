from data_processing.transformations_funcs import *

import numpy as np
import pandas as pd
import torch


class DataTransformation:
    def __init__(
            self,
            col_operation_func: None,
            col_operation_name: str = 'Operation',
            inverse_func: None = None):

        # invertible column-wise function
        self.col_operation_func = col_operation_func
        self.ret = []
        self.op_history = []
        self.col_operation_name = col_operation_name
        self.op_cnt = 0

    def _record(self, ret_data):
        self.op_cnt += 1
        self.ret.append(ret_data)
        self.op_history.append(f'{self.col_operation_name}_{self.op_cnt}')

    def __call__(
            self,
            data: pd.DataFrame,
            desired_cols: list):
        curr_data = data.copy(deep=True)

        for name in desired_cols:
            self.col_operation_func(curr_data, name)

        self._record(curr_data)

        return self.ret[-1]


class ColumnOperator(DataTransformation):
    def __init__(
            self,
            col_operation_func: None = None,
            col_operation_name: str = '',
            inverse_func: None = None,
            desired_cols: list = []):

        super().__init__(col_operation_func, col_operation_name, inverse_func)

        self.col_operation_name
        self.desired_cols = desired_cols

    def _record(self, ret_data):
        super()._record(ret_data)
        # self.ret[-1].rename(
        #     columns={self.desired_cols[0]: self.col_operation_name}, inplace=True)

    def __call__(
            self,
            data: pd.DataFrame):
        curr_data = data.copy(deep=True)

        self.col_operation_func(curr_data, self.desired_cols)

        self._record(curr_data)

        return self.ret[-1]


class OperationHistory:
    def __init__(
        self,
        seq: list
    ):
        self.seq = seq
        self.history_dict = {}

        if len(self.seq) > 0:
            self._to_dict()

    def extend(self, new_ops: list):
        self.seq.extend(new_ops)
        self._to_dict()

    def _to_dict(self):
        for op in self.seq:
            print(len(op.ret))
            self.history_dict[op.col_operation_name] = op.ret[-1]

    def __call__(self, op_names: list, ret_hist: list):
        ret_dict = {}
        for name, ret in zip(op_names, ret_hist):
            ret_dict[name] = ret

        return ret_dict


class OperationSequence:
    def __init__(
            self,
            seq: list = []):
        self.seq = seq

        # if len(self.seq) > 0:
        #   for i, op in enumerate(seq):
        #     if type(op) == OperationSequence:
        #       del(self.seq[i])
        #       self.seq.insert(i - 1, op.seq)

        # self.op_history = []
        # self.ret = []
        self.history = OperationHistory([])

    def __call__(self, df: pd.DataFrame, desired_cols):
        ret = df
        if len(self.seq) > 0:
            for func in self.seq:
                # print(type(self.ret[-1].ret[-1]))
                try:
                    if type(func) == DataTransformation:
                        ret = func(ret, desired_cols)
                        # self.ret.append(func)
                        # self.op_history.extend(func.op_history)

                    elif type(func) == ColumnOperator:
                        ret = func(ret)
                        # self.ret.append(func)
                        # self.op_history.extend(func.op_history)

                except:
                    raise Exception("Invalid element in result chain")
            self.history.extend(self.seq)


class PreprocessData:
    def __init__(
        self,
        generate_time_dummies: bool = True,
        quantile_cutoff: float = 1.0,
        op_list: OperationSequence = OperationSequence([])
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

            days = ['DUMMY_Monday', 'DUMMY_Tuesday', 'DUMMY_Wednesday',
                    'DUMMY_Thursday', 'DUMMY_Friday', 'DUMMY_Saturday',
                    'DUMMY_Sunday']
            hours = [f'DUMMY_hr_{i}' for i in range(24)]

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

        desired_cols = list(
            set(ret[-1].columns.values.tolist()) - set(days) - set(hours))

        self.op_list(ret[-1], desired_cols)
        ret.extend([op.ret[-1] for op in self.op_list.seq])
        op_name_history.extend([op.op_history[-1] for op in self.op_list.seq])

        ret_dict = self.op_list.history(op_name_history, ret)

        # return DF_0, ..., DF_5
        return ret_dict

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
