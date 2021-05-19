from dataclasses import dataclass
from .data_params import Features, InputDataset, SplittingParams
from .train_params import TrainingParams
from marshmallow_dataclass import class_schema
import yaml


@dataclass
class TrainingPipelineParams:
    input_dataset: InputDataset
    output_model_path: str
    metric_path: str
    splitting_params: SplittingParams
    train_params: TrainingParams


TrainingPipelineParamsSchema = class_schema(TrainingPipelineParams)


def read_training_pipeline_params(path: str) -> TrainingPipelineParams:
    with open(path, "r") as input_stream:
        schema = TrainingPipelineParamsSchema()
        return schema.load(yaml.safe_load(input_stream))