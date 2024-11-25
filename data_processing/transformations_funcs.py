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


class MADStandardizer(BaseEstimator, TransformerMixin):
    """
    A scikit-learn transformer for mean absolute deviation (MAD) standardization.
    """

    def __init__(self):
        self.feature_means_ = None
        self.feature_mads_ = None

    def fit(self, X, y=None):
        """
        Fit the transformer to the data by computing the mean and MAD for each feature.
        """
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input must be a Pandas DataFrame.")

        self.feature_means_ = X.mean()
        self.feature_mads_ = X.sub(self.feature_means_, axis=1).abs().mean()
        self.feature_mads_.replace(0, np.nan, inplace=True)

        return self

    def transform(self, X):
        """
        Standardize the data using MAD.
        """
        if self.feature_means_ is None or self.feature_mads_ is None:
            raise ValueError(
                "The transformer must be fitted before calling transform.")

        standardized = (X - self.feature_means_) / self.feature_mads_
        return standardized.to_numpy()

    def inverse_transform(self, X_transformed):
        """
        Revert the standardization to the original data scale.
        """
        if self.feature_means_ is None or self.feature_mads_ is None:
            raise ValueError(
                "The transformer must be fitted before calling inverse_transform.")

        return (X_transformed * self.feature_mads_) + self.feature_means_


class ArcsinhTransformer(BaseEstimator, TransformerMixin):
    """
    A scikit-learn transformer for applying the arcsinh transformation.
    """

    def __init__(self):
        pass

    def fit(self, X, y=None):
        """
        The fit method doesn't need to compute anything for the arcsinh transformation.
        """
        # Ensure input is a Pandas DataFrame
        # if not isinstance(X, pd.DataFrame):
        #     raise ValueError("Input must be a Pandas DataFrame.")
        return self

    def transform(self, X):
        """
        Apply the arcsinh transformation to the data.
        """
        # if not hasattr(self, 'fit'):
        #     raise AttributeError(
        #         "This ArcsinhTransformer instance is not fitted yet.")

        X_transformed = np.arcsinh(X)
        return X_transformed

    def inverse_transform(self, X_transformed):
        """
        Apply the inverse (sinh) transformation to revert data back to original scale.
        """
        X_original = X_transformed.applymap(np.sinh)
        return X_original
