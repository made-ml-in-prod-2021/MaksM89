import os
import pandas as pd
import pytest
import requests
import json
from sklearn.linear_model import LogisticRegression

from unittest.mock import patch, MagicMock
from ml_project.features.build_features import build_transformer, make_features, extract_target
from ml_project.models import serialize_model
from ml_project.predict import (
    inference_pipeline_command,
)
from click.testing import CliRunner

@pytest.fixture
def client():
    with webapp.test_client() as client:
        yield client
        
@pytest.fixture()
def data():
    return '{\"Age\":{\"1\":63,\"2\":67},\"Sex\":{\"1\":1,\"2\":1},\"ChestPain\":{\"1\":\"typical\",\"2\":\"asymptomatic\"},\"RestBP\":{\"1\":145,\"2\":160},\"Chol\":{\"1\":233,\"2\":286},\"Fbs\":{\"1\":1,\"2\":0},\"RestECG\":{\"1\":2,\"2\":2},\"MaxHR\":{\"1\":150,\"2\":108},\"ExAng\":{\"1\":0,\"2\":1},\"Oldpeak\":{\"1\":2.3,\"2\":1.5},\"Slope\":{\"1\":3,\"2\":2},\"Ca\":{\"1\":0.0,\"2\":3.0},\"Thal\":{\"1\":\"fixed\",\"2\":\"normal\"},\"AHD\":{\"1\":\"No\",\"2\":\"Yes\"}}'

@pytest.fixture()
@patch.object(os.path, 'join')
def model_path(mock_os, tmpdir, data, dataset_info):
    modelpath = tmpdir.join('model.pkl')
    mock_os.return_value = modelpath
    train_df = pd.DataFrame.from_dict(json.loads(data))
    train_target = extract_target(train_df, dataset_info.target_col)
    transformer = build_transformer(dataset_info.features)
    transformer.fit(train_df, train_target)
    train_features = make_features(transformer, train_df)
    model = LogisticRegression().fit(train_features, train_target)
    return serialize_model(model, transformer, modelpath)
    
def test_click_inference(tmpdir, dataset_info, model_path, data, capsys):
    runner = CliRunner()
    result = runner.invoke(inference_pipeline_command, ['model', data]).exit_code
    assert result == 0, 'Get error in prediction'