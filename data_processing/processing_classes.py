# from .transformations_funcs import *

import numpy as np
import pandas as pd
import torch
from sklearn.base import BaseEstimator, TransformerMixin
# from sklearn.pipeline import Pipeline


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


class Operation(BaseEstimator, TransformerMixin):
    def __init__(
            self,
            col_operation_func=None,
            inverse_operation_func=None,
            col_operation_name='',
            desired_cols=None,
            logging=False,
            op_history=None):

        self.col_operation_func = col_operation_func
        self.inverse_operation_func = inverse_operation_func
        self.col_operation_name = col_operation_name
        self.desired_cols = desired_cols
        self.logging = logging
        self.ret = []
        self.op_history = op_history
        self.op_cnt = 0

    def _record(self, ret_data):
        if self.logging:
            self.ret.append(ret_data)
            self.op_cnt += 1
            self.ret.extend([self])


class DataTransformation(Operation):
    def __init__(
            self,
            col_operation_func: None,
            col_operation_name: str = '',
            inverse_operation_func: None = None,
            desired_cols: None = None,
            logging: bool = True,
            op_history: OperationHistory = None):

        super().__init__(col_operation_func=col_operation_func,
                         inverse_operation_func=inverse_operation_func,
                         col_operation_name=col_operation_name,
                         desired_cols=desired_cols,
                         logging=logging,
                         op_history=op_history)

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
