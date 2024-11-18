import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from .processing_classes import DataTransformation, OperationHistory

# def calc_col_mad(df: pd.DataFrame, col_name: str):
#     col = df[col_name]
#     mad = (1/len(col)) * np.sum(np.abs(col - col.mean()))
#     return mad


# def log_transform_col(df: pd.DataFrame, col_name: str):
#     col = df[col_name].to_numpy()
#     sgn = np.where(col < 0, -1, 1)
#     df[col_name] = 1 + np.abs(df[col_name])
#     df[col_name] = sgn * np.log(df[col_name])


# def log_transform_inverse(df: pd.DataFrame, col_name: str, data):
#     sgn = df[col_name] / np.abs(df[col_name])
#     remove_sgn = sgn * data
#     remove_log = np.exp(remove_sgn)
#     abs = remove_log - 1
#     return sgn * abs


# def standardize_col(df: pd.DataFrame, col_name: str):
#     mad = calc_col_mad(df, col_name)
#     median = df.median()

#     df[col_name] = (df[col_name] - median[col_name]) / mad


# def reverse_standardization(
#         data,
#         col_mad: float,
#         col_med: float):

#     return (data * col_mad) + col_med


# def arcsinh_col(df: pd.DataFrame, col_name: str):
#     df[col_name] = np.arcsinh(df[col_name])


# def arcsinh_col_inverse(data):
#     return np.sinh(data)


# def subtr_cols(df: pd.DataFrame, col_names: list):
#     col_name_1, col_name_2 = col_names

#     df[col_name_1] = df[col_name_1] - df[col_name_2]


# def add_cols(df: pd.DataFrame, col_names: list):
#     target = col_names[0]
#     for name in col_names[1:]:

#         df[target] = df[target] + df[name]


# def mul_cols(df: pd.DataFrame, col_names: list):
#     col_name_1, col_name_2 = col_names

#     df[col_name_1] = df[col_name_1] * df[col_name_2]


# def div_cols(df: pd.DataFrame, col_names: list):
#     col_name_1, col_name_2 = col_names

#     df[col_name_1] = df[col_name_1] / df[col_name_2]


# def inverse_norm(
#         unstd_df: pd.DataFrame,
#         col_name: str,
#         start_idx: int,
#         end_idx: int,
#         data):
#     sgn = unstd_df[col_name][start_idx: end_idx] / \
#         np.abs(unstd_df[col_name][start_idx: end_idx])
#     col_mad = calc_col_mad(unstd_df, col_name)
#     col_med = unstd_df.median()[col_name]

#     de_norm = np.sinh(data)

#     log_val = reverse_standardization(de_norm, col_mad, col_med)

#     # log_removed = np.exp(sgn * log_val)
#     # abs_val = log_removed - 1
#     # val = abs_val * sgn

#     # print(unstandardized)

#     return log_val

#     # return unstandardized

days = ['DUMMY_Monday', 'DUMMY_Tuesday', 'DUMMY_Wednesday',
        'DUMMY_Thursday', 'DUMMY_Friday', 'DUMMY_Saturday',
        'DUMMY_Sunday']
hours = [f'DUMMY_hr_{i}' for i in range(24)]


class MADStandardScaler(DataTransformation):
    def __init__(self, desired_cols=None, logging=False, op_history=None):
        super().__init__(col_operation_name='MADStandardScaler',
                         desired_cols=desired_cols,
                         logging=logging,
                         op_history=op_history,
                         col_operation_func=self.transform,
                         inverse_operation_func=self.inverse_transform)
        self.median_ = None
        self.mad_ = None

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        if self.desired_cols is None:
            self.desired_cols = list(
                (set(X.columns.tolist()) - set(days)) - set(hours))
        self.median_ = X[self.desired_cols].median()
        self.mad_ = (X[self.desired_cols] - self.median_).abs().median()
        return self

    def transform(self, X):
        X_transformed = pd.DataFrame(X).copy()
        X_transformed[self.desired_cols] = (
            X_transformed[self.desired_cols] - self.median_) / self.mad_
        self._record(X_transformed)
        return X_transformed

    def inverse_transform(self, X):
        X_inverse_transformed = pd.DataFrame(X).copy()
        X_inverse_transformed[self.desired_cols] = X_inverse_transformed[self.desired_cols] * \
            self.mad_ + self.median_
        return X_inverse_transformed


class ArcsinhTransformer(DataTransformation):
    def __init__(self, desired_cols=None, logging=False, op_history=None):
        super().__init__(
            col_operation_func=np.arcsinh,
            inverse_operation_func=np.sinh,
            col_operation_name='ArcsinhTransformer',
            desired_cols=desired_cols,
            logging=logging,
            op_history=op_history
        )

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        if self.desired_cols is None:
            self.desired_cols = list(
                (set(X.columns.tolist()) - set(days)) - set(hours))
        return self
