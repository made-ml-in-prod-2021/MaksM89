# -*- coding: utf-8 -*-
from typing import Tuple

import os
import requests
import pandas as pd
from sklearn.model_selection import train_test_split

from ml_project.enities.data_params import InputDataset, SplittingParams
# from ml_project.train_pipeline import logger


def read_data(dataset_info: InputDataset) -> pd.DataFrame:
    """
    path: InputDataset - paths to load, download file
    
    :rt: Dataframe
    """
    if not os.path.exists(dataset_info.path):
        responce = requests.get(dataset_info.download_path)
        assert responce.ok, 'Cannot download dataset'
        # from pdb import set_trace; set_trace();
        if not os.path.isdir(os.path.dirname(dataset_info.path)):
            os.mkdir(os.path.dirname(dataset_info.path))
        with open(dataset_info.path, 'wb') as fout:
            fout.write(responce.content)
            # print('Dataset download')
            # logger.debug('Dataset download')
    columns = dataset_info.features.categorical_features\
                + dataset_info.features.numerical_features\
                + [dataset_info.target_col]
    # data = pd.read_csv(dataset_info.path, index_col=0, usecols=columns)
    data = pd.read_csv(dataset_info.path, usecols=columns)
    return data


def split_train_val_data(
    data: pd.DataFrame, params: SplittingParams
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """

    :rtype: object
    """
    train_data, val_data = train_test_split(
        data, test_size=params.val_size, random_state=params.random_state
    )
    return train_data, val_data
