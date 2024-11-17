from .transformations_funcs import *

import numpy as np
import pandas as pd
import torch
# from sklearn.pipeline import Pipeline


class Operation:
    def __init__(
            self,
            col_operation_func: None,
            inverse_operation_func: None,
            col_operation_name: str):

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


class DataTransformation(Operation):
    def __init__(
            self,
            col_operation_func: None,
            col_operation_name: str = '',
            inverse_func: None = None):

        super().__init__(col_operation_func, col_operation_name, inverse_func)
        self.col_operation_name = col_operation_name
        # self.col_operation_func = col_operation_func

    def __call__(
            self,
            data: pd.DataFrame,
            desired_cols: list):
        curr_data = data.copy(deep=True)

        for name in desired_cols:
            self.col_operation_func(curr_data, name)

        self._record(curr_data)

        return self.ret[-1]
x


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
        self.history = OperationHistory([])

    def __call__(self, df: pd.DataFrame, desired_cols):
        ret = df
        if len(self.seq) > 0:
            for func in self.seq:
                try:
                    if type(func) == DataTransformation:
                        ret = func(ret, desired_cols)

                    elif type(func) == ColumnCombination:
                        ret = func(ret)

                except:
                    raise Exception("Invalid element in result chain")
            self.history.extend(self.seq)
