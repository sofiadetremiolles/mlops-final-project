"""
This module is just a convenience to train a simple classifier.
Its presence is a bit artificial for the exercice and not required to develop an MLOps platform.
The MLOps course is not about model training.
"""

from dataclasses import dataclass
import logging
import pandas as pd
from datetime import datetime
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.base import ClassifierMixin, RegressorMixin

from dsba.model_registry import ClassifierMetadata
from dsba.preprocessing import split_features_and_target, preprocess_dataframe


def train_simple_classifier(
    df: pd.DataFrame, target_column: str, model_id: str
) -> tuple[ClassifierMixin, ClassifierMetadata]:
    logging.info("Start training a simple classifier")
    df = preprocess_dataframe(df)
    X, y = split_features_and_target(df, target_column)
    model = HistGradientBoostingClassifier(random_state=42)
    model.fit(X, y)

    logging.info("Done training a simple classifier")
    metadata = ClassifierMetadata(
        id=model_id,
        created_at=str(datetime.now()),
        algorithm="xgboost",
        target_column=target_column,
        hyperparameters={"random_state": 42},
        description="",
        performance_metrics={},
    )
    return model, metadata
