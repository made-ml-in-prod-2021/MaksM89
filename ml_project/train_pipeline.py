import json
import logging
import sys
import os

import click
import pandas as pd

from ml_project.data import read_data, split_train_val_data
from ml_project.enities.train_pipeline_params import (
    TrainingPipelineParams,
    read_training_pipeline_params,
)
from ml_project.features import make_features
from ml_project.features.build_features import extract_target, build_transformer
from ml_project.models import (
    train_model,
    serialize_model,
    predict_model,
    evaluate_model,
)

logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.setLevel(logging.INFO)
logger.addHandler(handler)


def train_pipeline(training_pipeline_params: TrainingPipelineParams):
    logger.info(f"start train pipeline with params {training_pipeline_params}")
    data = read_data(training_pipeline_params.input_dataset)
    logger.info(f"data.shape is {data.shape}")
    train_df, val_df = split_train_val_data(
        data, training_pipeline_params.splitting_params
    )
    logger.info(f"train_df.shape is {train_df.shape}")
    logger.info(f"val_df.shape is {val_df.shape}")
    
    train_target = extract_target(train_df, training_pipeline_params.input_dataset.target_col)
    transformer = build_transformer(training_pipeline_params.input_dataset.features)
    transformer.fit(train_df.drop(columns=[training_pipeline_params.input_dataset.target_col]), train_target)    
    train_features = make_features(transformer, train_df.drop(columns=[training_pipeline_params.input_dataset.target_col]))

    logger.info(f"train_features.shape is {train_features.shape}")

    model = train_model(
        train_features, train_target, training_pipeline_params.train_params
    )

    val_features = make_features(transformer, val_df.drop(columns=[training_pipeline_params.input_dataset.target_col]))
    val_target = extract_target(val_df, training_pipeline_params.input_dataset.target_col)

    logger.info(f"val_features.shape is {val_features.shape}")
    predicts = predict_model(
        model,
        val_features,
    )

    metrics = evaluate_model(
        predicts,
        val_target,
    )
    
    if not os.path.isdir(os.path.dirname(training_pipeline_params.metric_path)):
        os.mkdir(os.path.dirname(training_pipeline_params.metric_path))
        
    with open(training_pipeline_params.metric_path, "w") as metric_file:
        json.dump(metrics, metric_file)
    logger.info(f"metrics is {metrics}")

    path_to_model = serialize_model(model, transformer, training_pipeline_params.output_model_path)

    return path_to_model, metrics


@click.command(name="train_pipeline")
@click.argument("config_path")
def train_pipeline_command(config_path: str):
    params = read_training_pipeline_params(config_path)
    train_pipeline(params)


if __name__ == "__main__":
    train_pipeline_command()
