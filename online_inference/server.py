import json
import os
import sys
import glob

import pandas as pd
import numpy as np
import pickle

from flask import (
    Flask,
    abort,
    request,
    jsonify,
)

app = Flask(__name__)
app.model_path = os.environ.get('MODELS_PATH', 'models')

@app.route('/')
def main():
    return 'Usage: host:port/query?modelname=model_name&data=json_data', 200
    
@app.route('/models')
def get_models():
    # from pdb import set_trace; set_trace()
    files = glob.glob(os.path.join(app.model_path, '*.pkl'))
    files = [os.path.splitext(os.path.basename(f))[0] for f in files]
    return jsonify(files)
    
@app.route('/query')
def inference_pipeline():
    # from pdb import set_trace; set_trace()
    model_path = os.path.join(app.model_path, request.args.get('modelname') + '.pkl')
    data = request.args.get('data')
    try:
        with open(model_path, "rb") as fin:
            model, transformer = pickle.load(fin)
    except:
        return f'Cannot load model from "{model_path}"', 501
    app.logger.info(f'Model {model_path} loaded')
    # from pdb import set_trace; set_trace()
    try:
        dataframe = pd.DataFrame.from_dict(json.loads(data))
    except:
        return f'Cannot parse json data: {data}', 502
    # columns = set(transformer.transformers_[0][2] + transformer.transformers_[1][2])
    # # dataframe = dataframe[transformer.transformers_[0][2] + transformer.transformers_[1][2]]
    # if (columns - set(dataframe.columns)) != set():
        # return f'Wrong query data. Columns must be {columns}', 501
    columns = transformer.transformers_[0][2] + transformer.transformers_[1][2]
    try:
        dataframe = dataframe[columns]
    except:
        return f"Columns must be {columns}", 503
    app.logger.info(f'Model {model_path} loaded, starting prediction')
    predicts = model.predict(transformer.transform(dataframe))
    predicts = np.where(predicts, 'Yes', 'No').tolist()
    app.logger.info('Prediction done')
    return jsonify(predicts)