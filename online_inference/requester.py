import json
import logging
import os
import sys
import requests

import click
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.setLevel(logging.INFO)
logger.addHandler(handler)

@click.command(name="get_request")
@click.argument("modelname")
@click.argument("datapath")
def inference_pipeline_command(modelname: str, datapath: str):
    # from pdb import set_trace; set_trace()
    try:
        dataframe = pd.read_csv(datapath)
    except:
        logger.info(f'Cannot load data form {datapath}')
        return -1
    try:
        logger.info(f'Send request to server.')
        responce = requests.get(
                    'http://192.168.99.100:5000/query', 
                    # 'http://localhost:5000/query', # for local debug
                    params={'modelname': modelname, 'data': dataframe.to_json()})
    except ConnectionError:
        logger.error('No connection with host')
        return -1
    if not responce.ok:
        logger.error(f'Cannot make request. Error code {responce.status_code}, message "{responce.text}"')
        return -1
    predicts = responce.text
    logger.info(f'Get answer from server.')
    print(predicts)
    return 0


if __name__ == "__main__":
    inference_pipeline_command()
