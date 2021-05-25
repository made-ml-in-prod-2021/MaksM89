import json
import logging
import os
import sys

import click
import pandas as pd
import numpy as np

from ml_project.data import read_data, split_train_val_data
from ml_project.features.build_features import make_features

from ml_project.models import (
    deserialize_model,
    predict_model,
)

logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.setLevel(logging.INFO)
logger.addHandler(handler)

@click.command(name="inference_pipeline")
@click.argument("modelname")
@click.argument("data")
def inference_pipeline_command(modelname: str, data: str):
    path = os.path.join('models', modelname + '.pkl')
    try:
        model, transformer = deserialize_model(path)
    except:
        print(f'Cannot load model "{path}"')
        return -1
    try:
        dataframe = pd.DataFrame.from_dict(json.loads(data))
    except:
        print(f'Cannot parse json data: {data}')
        return -1
    columns = transformer.transformers_[0][2] + transformer.transformers_[1][2]
    try:
        dataframe = dataframe[columns]
        print(dataframe.columns)
    except:
        print(f"Columns must be {columns}")
        return -1
    logger.info(f'Model {modelname} loaded, starting prediction')
    predicts = predict_model(
        model,
        make_features(transformer, dataframe)
    )
    predicts = np.where(predicts, 'Yes', 'No').tolist()
    logger.info('Prediction done')
    print(predicts)
    return 0


if __name__ == "__main__":
    inference_pipeline_command()
