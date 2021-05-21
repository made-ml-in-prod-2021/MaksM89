import os

import pytest
from typing import List
import numpy as np
import pandas as pd

from ml_project.enities import Features, InputDataset

@pytest.fixture(scope='session')
def categorical_features() -> List[str]:
    return [
        'Sex',
        'Fbs',
        'RestECG',
        'ExAng',
        'Slope',
        'ChestPain',
        'Ca',
        'Thal'
    ]


@pytest.fixture(scope='session')
def numerical_features() -> List[str]:
    return [
        'Age',
        'RestBP',
        'Chol',
        'MaxHR',
        'Oldpeak'
    ]

@pytest.fixture(scope='session')
def dataset_info(categorical_features, numerical_features) -> InputDataset:
    np.random.seed(42)
    datalength = 300
    data = dict()
    for col in categorical_features:
        values = ['one', 'two', 'three', 'four']
        nacount = np.random.randint(10)
        column = np.random.choice(values, datalength)
        column[np.random.randint(datalength, size=nacount)] = np.nan
        data[col] = column
    for col in numerical_features:
        nacount = np.random.randint(10)
        column = np.random.randint(100, size=datalength).astype(np.float)
        column[np.random.randint(datalength, size=nacount)] = np.nan
        data[col] = column
    data['AHD'] = np.random.choice(['Yes', 'No'], datalength)
    curdir = os.path.dirname(__file__)
    filepath = os.path.join(curdir, 'fake_dataset.csv')
    pd.DataFrame(data).to_csv(filepath)
    features = Features(
        categorical_features=categorical_features,
        numerical_features=numerical_features,
    )
    yield InputDataset(
        path=filepath,
        download_path='',
        features=features,
        target_col='AHD'
        )
    os.remove(filepath)
