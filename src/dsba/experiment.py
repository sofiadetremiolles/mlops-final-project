from dsba.data_ingestion import load_csv_from_url
from dsba.preprocessing import split_dataframe
from dsba.model_training import train_simple_classifier
from dsba.model_registry import save_model
from dsba.model_evaluation import evaluate_classifier

from dsba.model_registry import ClassifierMetadata


def run_experiment(input_url: str, target_column: str, model_id: str) -> ClassifierMetadata:
    """
    This function allows the user to run an experiment. 
    The user can choose a dataset to classify and save a trained model under model_id.
    The returned output is the performance metrics of the model on a subset of the data.
    """
    df = load_csv_from_url(input_url)
    df_train, df_test = split_dataframe(df, test_size=0.2)
    clf, metadata = train_simple_classifier(df_train, target_column, model_id)
    evaluation_results = evaluate_classifier(clf, target_column, df_test)
    metadata.performance_metrics = {
        "accuracy": evaluation_results.accuracy,
        "precision": evaluation_results.precision,
        "recall": evaluation_results.recall,
        "f1_score": evaluation_results.f1_score
        }
    metadata.description = f" Classifier created with data from {input_url}"
    save_model(clf, metadata)
    return metadata
    
