import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin

from ml_project.enities.data_params import Features

class MyScaleTransformer(TransformerMixin, BaseEstimator):
    """Bad implementation of sklearn StandardScaler"""

    def fit(self, X: np.ndarray):
        self.mean_ = X.mean(0)
        self.std_ = X.std(0)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean_) / self.std_

def build_categorical_pipeline() -> Pipeline:
    categorical_pipeline = Pipeline(
        [
            ("impute", SimpleImputer(missing_values=np.nan, strategy="most_frequent")),
            ("ohe", OneHotEncoder(sparse=False, drop='if_binary')),
        ]
    )
    return categorical_pipeline

def build_numerical_pipeline() -> Pipeline:
    num_pipeline = Pipeline(
        [
            ("impute", SimpleImputer(missing_values=np.nan, strategy="mean")),
            # ("scale", StandardScaler()),
            ("scale", MyScaleTransformer()),
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
    # from pdb import set_trace; set_trace()
    target = df[target_col].apply(lambda x: 1 if x == 'Yes' else 0)
    return target.values