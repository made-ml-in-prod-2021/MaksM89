import pickle
from typing import Dict, Union

import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

from ml_project.enities.train_params import TrainingParams

SklearnModel = LogisticRegression

def train_model(
    features: np.ndarray, 
    target: np.ndarray, 
    train_params: TrainingParams
) -> SklearnModel:
    if train_params.model_type == "LogisticRegression":
        model = LogisticRegression()
    else:
        raise NotImplementedError()
    model.fit(features, target)
    return model


def predict_model(model: SklearnModel, features: np.ndarray) -> np.ndarray:
    predicts = model.predict(features)
    return predicts


def evaluate_model(predicts: np.ndarray, target: np.ndarray) -> Dict[str, float]:
    return {
        "accuracy": accuracy_score(target, predicts),
        "roc_auc": roc_auc_score(target, predicts),
        "f1_score": f1_score(target, predicts),
    }


def serialize_model(model: SklearnModel, 
                    transformer: ColumnTransformer, 
                    output: str) -> str:
    with open(output, "wb") as fout:
        pickle.dump((model, transformer), fout)
        
    return output
    
def deserialize_model(model_path: str) -> str:
    with open(model_path, "rb") as fin:
        model, transformer = pickle.load(fin)
    return model, transformer
