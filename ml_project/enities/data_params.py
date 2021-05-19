from typing import List
from dataclasses import dataclass, field
from marshmallow_dataclass import class_schema

@dataclass
class Features:
    categorical_features: List[str]
    numerical_features: List[str]
    
@dataclass
class InputDataset:
    path: str
    download_path: str
    features: Features
    target_col: str

@dataclass
class SplittingParams:
    val_size: float = field(default=0.2)
    random_state: int = field(default=42)