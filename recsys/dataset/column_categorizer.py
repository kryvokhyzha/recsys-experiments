import sklearn
import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator


class ColumnCategorizer(TransformerMixin, BaseEstimator):
    """Creates new categorical features using continuous columns

    Requires
    - quantiles list
    - columns to transform
    - new columns names
    """

    def __init__(self, qs, new_columns, old_columns):
        self.qs = qs  # quantile list
        self.new_columns = new_columns  # columns to create
        self.old_columns = old_columns  # initial continuous columns
        self.transformers = {}  # transformers storage

    def fit(self, X, y=None):
        """Fit transformers on a part of the columns
        X - train data
        """

        for column in self.old_columns:
            bins = np.unique([X[column].quantile(q=q) for q in self.qs])
            labels = [f"{column}_bin_{l_edge}_{r_edge}" for (l_edge, r_edge) in zip(bins[:-1], bins[1:])]
            self.transformers[column] = sklearn.preprocessing.FunctionTransformer(
                pd.cut, kw_args={
                    'bins': bins,
                    'include_lowest': True,
                    'labels': labels,
                    'retbins': False,
                }
            )
            self.transformers[column].fit(X[column])
        return self

    def transform(self, X):
        """Transform data and return Pandas DataFrame
        X - data to transform
        """

        data_new = X.copy()
        for column_old, column_new in zip(self.old_columns, self.new_columns):
            data_new[column_new] = self.transformers[column_old].transform(data_new[column_old]).values
        return data_new


def names_to_cats(data_to_transform, cols):
    """Transform numeric categories using names of features

    data_to_transform - data to transform
    cols - columns to change
    """

    data_new = data_to_transform.copy()
    for col in cols:
        data_new[col] = data_new[col].apply(lambda x: f"{col}_{x}").values
    return data_new
