import os
import logging
import csv
import pandas as pd
from pathlib import Path
from dsba.model_registry import list_models_ids, load_model_metadata, _get_absolute_path
from dsba.model_registry import ClassifierMetadata


def report_benchmark(metric: str) -> None:
    """
    Generates a benchmark report ranking models by the specified performance metric.
    """
    sorted_models = make_benchmark(metric)
    data = [{
        "model_id": metadata.id,
        "description": metadata.description,
        "metric": metric,
        "score": score
    } for metadata, score in sorted_models]
    df = pd.DataFrame(data)
    reports_dir = _get_reports_dir()
    report_file_path = reports_dir / f"benchmark_{metric}.csv"
    df.to_csv(report_file_path, index=False)
    logging.info(f"Benchmark report saved to {report_file_path}")
    return df


def make_benchmark(metric: str) -> list[tuple[ClassifierMetadata, float]]:
    """
    Creates a list of existing models sorted by the specified performance metric.
    """
    # Retrieve the list of saved model IDs
    model_ids = list_models_ids()
    models_with_score = []
    # Iterate over each model ID and load its metadata
    for model_id in model_ids:
        metadata = load_model_metadata(model_id)
        score = metadata.performance_metrics[metric]
        models_with_score.append((metadata, score))
    sorted_models = sorted(models_with_score, key=lambda x: x[1], reverse=True)
    return sorted_models


def _get_reports_dir() -> Path:
    """
    Retrieve the reports directory from the environment variable DSBA_REPORTS_ROOT_PATH.
    If the directory does not exist, it is created.
    """
    DSBA_REPORTS_ROOT_PATH = os.getenv("DSBA_REPORTS_ROOT_PATH")
    if DSBA_REPORTS_ROOT_PATH is None:
        raise ValueError(
            "Environment variable DSBA_REPORTS_ROOT_PATH is not set. "
            "Please set it to the root path where reports should be stored."
        )
    reports_dir = _get_absolute_path(DSBA_REPORTS_ROOT_PATH)
    if not reports_dir.exists():
        logging.info(f"Reports directory does not exist, creating it at {reports_dir}")
        reports_dir.mkdir(parents=True)
    return reports_dir



