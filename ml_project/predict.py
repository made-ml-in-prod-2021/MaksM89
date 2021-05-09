import json
import logging
import os
import sys

import click
import pandas as pd
import numpy as np

from flask import (
    Flask,
    abort,
    request,
    jsonify,
)

app = Flask(__name__)

from ml_project.data import read_data, split_train_val_data

from ml_project.models import (
    deserialize_model,
    predict_model,
)

logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.setLevel(logging.INFO)
logger.addHandler(handler)

@app.route('/')
def main():
    return 'Usage: /query?model=<model_name>&data=<json_data>'
    
@app.route('/query')
def inference_pipeline():
    modelname = os.path.join('models', request.args.get('modelname') + '.pkl')
    data = request.args.get('data')
    try:
        model, transformer = deserialize_model(modelname)
    except:
        return f'Cannot load model "{modelname}"', 501
    app.logger.info(f'Model {modelname} loaded')
    # from pdb import set_trace; set_trace()
    try:
        dataframe = pd.DataFrame.from_dict(json.loads(data))
    except:
        return f'Cannot parse json data: {data}', 501
    columns = set(transformer.transformers_[0][2] + transformer.transformers_[1][2])
    if (columns - set(dataframe.columns)) != set():
        return f'Wrong query data. Columns must be {columns}', 501
    app.logger.info(f'Model {modelname} loaded, starting prediction')
    predicts = predict_model(
        model,
        transformer.transform(dataframe),
    )
    predicts = np.where(predicts, 'Yes', 'No').tolist()
    app.logger.info('Prediction done')
    return jsonify(predicts)
    


@click.command(name="inference_pipeline")
@click.argument("modelname")
@click.argument("data")
def inference_pipeline_command(modelname: str, data: str):
    # from pdb import set_trace; set_trace()
    path = os.path.join('ml_project/models', modelname + '.pkl')
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
    columns = set(transformer.transformers_[0][2] + transformer.transformers_[1][2])
    # columns = set(transformer.transformers[0][2] + transformer.transformers[1][2])
    if (columns - set(dataframe.columns)) != set():
        print(f'Wrong query data. Columns must be {columns}')
        return -1
    logger.info(f'Model {modelname} loaded, starting prediction')
    predicts = predict_model(
        model,
        transformer.transform(dataframe),
    )
    predicts = np.where(predicts, 'Yes', 'No').tolist()
    logger.info('Prediction done')
    print(predicts)
    # from pdb import set_trace; set_trace()
    return 0


if __name__ == "__main__":
    inference_pipeline_command()
