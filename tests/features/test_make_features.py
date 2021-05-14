from typing import List

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from sklearn.preprocessing import StandardScaler

from ml_project.data.make_dataset import read_data
from ml_project.enities.data_params import InputDataset
from ml_project.features.build_features import (
    make_features,
    extract_target,
    build_transformer,
    # MyScaleTransformer,
    MyMeanEncoder,
    )

# def test_own_transformer():
    # data = pd.DataFrame(np.random.randint(100, size=(50, 1)))
    # skTr = StandardScaler()
    # ownTr = MyScaleTransformer()
    # skTr.fit(data)
    # ownTr.fit(data)
    # # from pdb import set_trace; set_trace()
    # assert pytest.approx(skTr.mean_) == ownTr.mean_, 'Different means'
    # # assert pytest.approx(np.sqrt(skTr.var_)) == ownTr.std_, 'Different variance'
    # # assert_allclose(ownTr.fit_transform(data), skTr.fit_transform(data))
    # assert np.all(ownTr.transform(data).std(0) == [pytest.approx(1)]), 'Variance must be 1'

def test_own_transformer():
    X = np.array([
        ['one', 'bad', 'man'],
        ['two', 'bad', 'woman'],
        ['one', 'good', 'woman'],
        ['one', 'bad', 'man'],
        ['three', 'bad', 'man'],
        ['one', 'good', 'man'],
    ])
    y = np.array([0, 1, 1, 0, 0, 0])
    rightans = np.array([
        [0.25, 0.25, 0.],
        [1., 0.25, 1.],
        [0.25, 0.5, 1.],
        [0.25, 0.25, 0.],
        [0., 0.25, 0.],
        [0.25, 0.5, 0.],
    ])
    encoder = MyMeanEncoder()
    result = encoder.fit_transform(X, y)
    assert np.array_equal(result, rightans), 'Wrong results'

def test_make_features(dataset_info: InputDataset):
    data = read_data(dataset_info)
    target = extract_target(data, dataset_info.target_col)
    # from pdb import set_trace; set_trace()
    assert isinstance(target, np.ndarray), 'Target must be np.ndarray'
    assert np.array_equiv(np.unique(target), np.array([0, 1])), 'Wrong target values'
    transformer = build_transformer(dataset_info.features)
    transformer.fit(data, target)
    features = make_features(transformer, data)
    intfeaturelength = len(transformer.transformers_[1][2])
    assert isinstance(features, np.ndarray), 'Features must be np.ndarray'
    assert not np.isnan(features).any(), 'Why nan values in features?'
    assert np.all(pytest.approx(features[:, -intfeaturelength:].mean(0)) == 0),\
            'Mean value of int features is not 0'
    # assert np.array_equiv(np.unique(features[:, :intfeaturelength]), np.array([0, 1])),\
            # 'Categorical features without onehot'
    assert np.all([np.unique(features[:, i]).shape == np.unique(data.dropna().values[:, i]).shape\
                    for i in range(data.shape[1]- intfeaturelength - 1)]), 'Wrong values for categorical features'
    


# def test_extract_target(dataset_info: InputDataset):
    # data = read_data(dataset_info)
    # target = extract_target(data, dataset_info.target_col)
    # assert isinstance(target, np.ndarray), 'Target must be np.ndarray'
    # assert np.array_equiv(np.unique(target), np.array([0, 1])), 'Wrong target values'
