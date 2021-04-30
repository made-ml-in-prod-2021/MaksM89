import os
import pickle
from typing import List, Tuple

import numpy as np
import pytest
from py._path.local import LocalPath
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

from ml_project.data.make_dataset import read_data
from ml_project.enities.train_params import TrainingParams
from ml_project.enities.data_params import InputDataset
from ml_project.features.build_features import make_features, extract_target, build_transformer
from ml_project.models.model_fit_predict import train_model, serialize_model


@pytest.fixture
def features_and_target(dataset_info) -> Tuple[np.ndarray, np.ndarray]:
    data = read_data(dataset_info)
    transformer = build_transformer(dataset_info.features)
    transformer.fit(data)
    features = make_features(transformer, data)
    target = extract_target(data, dataset_info.target_col)
    return features, target


def test_train_model(features_and_target: Tuple[np.ndarray, np.ndarray]):
    features, target = features_and_target
    model = train_model(features, target, train_params=TrainingParams())
    assert isinstance(model, LogisticRegression)
    assert model.predict(features).shape == target.shape


def test_serialize_model(tmpdir: LocalPath):
    expected_output = tmpdir.join("model.pkl")
    model = LogisticRegression()
    transformer = ColumnTransformer([("simple", SimpleImputer, ['col'])])
    real_output = serialize_model(model, transformer, expected_output)
    assert real_output == expected_output
    with open(real_output, "rb") as f:
        model, transformer = pickle.load(f)
    assert isinstance(model, LogisticRegression)\
            and isinstance(transformer, ColumnTransformer),\
            'Cannot serialize model'