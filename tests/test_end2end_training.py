import os
from typing import List

from py._path.local import LocalPath
from ml_project.enities import InputDataset

from ml_project.train_pipeline import train_pipeline
from ml_project.enities import (
    TrainingPipelineParams,
    SplittingParams,
    InputDataset,
    TrainingParams,
)


def test_train_e2e(
    tmpdir: LocalPath,
    dataset_info: InputDataset,
):
    expected_output_model_path = tmpdir.join("model.pkl")
    expected_metric_path = tmpdir.join("metrics.json")
    params = TrainingPipelineParams(
        input_dataset=dataset_info,
        output_model_path=expected_output_model_path,
        metric_path=expected_metric_path,
        splitting_params=SplittingParams(val_size=0.2, random_state=42),
        train_params=TrainingParams(model_type="LogisticRegression"),
    )
    real_model_path, metrics = train_pipeline(params)
    assert metrics["accuracy"] > 0
    assert os.path.exists(real_model_path)
    assert os.path.exists(params.metric_path)
