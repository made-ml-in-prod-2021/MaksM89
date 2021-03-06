import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin

from ml_project.enities.data_params import Features
 
class MyMeanEncoder(TransformerMixin, BaseEstimator):
    """https://towardsdatascience.com/why-you-should-try-mean-encoding-17057262cd0"""

    def fit(self, X: np.ndarray, y: np.ndarray):
        assert isinstance(X, np.ndarray), f'Type must be np.ndarray, but got {type(X)}'
        assert isinstance(y, np.ndarray), f'Type must be np.ndarray, but got {type(y)}'
        self.columns_dict = []
        for i in range(X.shape[1]):
            uv = np.unique(X[:, i])[None, :]
            mask = X[:, i].reshape(-1, 1) == uv
            self.columns_dict.append(
                {uv[0, j]: y[mask[:, j]].sum() / mask[:, j].sum()\
                    for j in range(uv.shape[1])}
                )
        return self

    def transform(self, X: np.ndarray, y=None) -> np.ndarray:
        assert isinstance(X, np.ndarray), f'Type must be np.ndarray, but got {type(X)}'
        assert X.shape[1] == len(self.columns_dict), f'Wrong columns count {X.shape[1]}'
        result = []
        for i in range(X.shape[1]):
            mapping = self.columns_dict[i]
            unknown_value = np.mean(list(mapping.values()))
            result.append(
                list(map(lambda x: mapping.get(x, unknown_value), X[:, i]))
                )
        return np.array(result).T

def build_categorical_pipeline() -> Pipeline:
    categorical_pipeline = Pipeline(
        [
            ("impute", SimpleImputer(missing_values=np.nan, strategy="most_frequent")),
            ("meanEnc", MyMeanEncoder()),
        ]
    )
    return categorical_pipeline

def build_numerical_pipeline() -> Pipeline:
    num_pipeline = Pipeline(
        [
            ("impute", SimpleImputer(missing_values=np.nan, strategy="mean")),
            ("scale", StandardScaler()),
        ]
    )
    return num_pipeline


def make_features(transformer: ColumnTransformer, df: pd.DataFrame) -> np.ndarray:
    return transformer.transform(df)

def build_transformer(params: Features) -> ColumnTransformer:
    transformer = ColumnTransformer(
        [
            (
                "categorical_pipeline",
                build_categorical_pipeline(),
                params.categorical_features,
            ),
            (
                "numerical_pipeline",
                build_numerical_pipeline(),
                params.numerical_features,
            ),
        ]
    )
    return transformer


def extract_target(df: pd.DataFrame, target_col) -> np.ndarray:
    target = df[target_col].apply(lambda x: 1 if x == 'Yes' else 0)
    return target.values