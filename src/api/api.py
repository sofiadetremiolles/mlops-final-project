import json
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dsba.model_registry import list_models_ids, load_model, load_model_metadata
from dsba.model_prediction import classify_record
from dsba.experiment import run_experiment
from dsba.benchmark import report_benchmark


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S,",
)

app = FastAPI()


# using FastAPI with defaults is very convenient
# we just add this "decorator" with the "route" we want.
# If I deploy this app on "https//mywebsite.com", this function can be called by visiting "https//mywebsite.com/models/"
@app.get("/models/")
async def list_models():
    return list_models_ids()


@app.api_route("/predict/", methods=["GET", "POST"])
async def predict(query: str, model_id: str):
    """
    Predict the target column of a record using a model.
    The query should be a json string representing a record.
    """
    # This function is a bit naive and focuses on the logic.
    # To make it more production-ready you would want to validate the input, manage authentication,
    # process the various possible errors and raise an appropriate HTTP exception, etc.
    try:
        record = json.loads(query)
        model = load_model(model_id)
        metadata = load_model_metadata(model_id)
        prediction = classify_record(model, record, metadata.target_column)
        return {"prediction": prediction}
    except Exception as e:
        # We do want users to be able to see the exception message in the response
        # FastAPI will by default block the Exception and send a 500 status code
        # (In the HTTP protocol, a 500 status code just means "Internal Server Error" aka "Something went wrong but we're not going to tell you what")
        # So we raise an HTTPException that contains the same details as the original Exception and FastAPI will send to the client.
        raise HTTPException(status_code=500, detail=str(e))


@app.api_route("/run/", methods=["GET","POST"])
async def run(input_url: str, target_column: str, model_id: str):
    """
    Run an experiment where a model is trained and evaluated.
    The input_url should point to the data that will be used for training and testing.
    The target column should be the name of the column to classify in the input data.
    """
    # This function is a bit naive and focuses on the logic.
    # To make it more production-ready you would want to validate the input, manage authentication,
    # process the various possible errors and raise an appropriate HTTP exception, etc.
    try:
        metadata = run_experiment(input_url, target_column, model_id)
        return {
            "Accuracy": f"{metadata.performance_metrics['accuracy']:.2f}",
            "Precision": f"{metadata.performance_metrics['precision']:.2f}",
            "Recall": f"{metadata.performance_metrics['recall']:.2f}",
            "F1 Score": f"{metadata.performance_metrics['f1_score']:.2f}"
            }

    except Exception as e:
        # We do want users to be able to see the exception message in the response
        # FastAPI will by default block the Exception and send a 500 status code
        # (In the HTTP protocol, a 500 status code just means "Internal Server Error" aka "Something went wrong but we're not going to tell you what")
        # So we raise an HTTPException that contains the same details as the original Exception and FastAPI will send to the client.
        raise HTTPException(status_code=500, detail=str(e))


@app.api_route("/report/", methods=["GET", "POST"])
async def report(metric: str):
    """
    Report a benchmark with all models trained
    """
    # This function is a bit naive and focuses on the logic.
    # To make it more production-ready you would want to validate the input, manage authentication,
    # process the various possible errors and raise an appropriate HTTP exception, etc.
    try:
        df = report_benchmark(metric)
        if df.empty:
            return {"message": "No models available", "best_model": None}
        best_model = df.iloc[0].to_dict()
        return {"message": "Report generated successfully", "best_model": best_model}

    except Exception as e:
        # We do want users to be able to see the exception message in the response
        # FastAPI will by default block the Exception and send a 500 status code
        # (In the HTTP protocol, a 500 status code just means "Internal Server Error" aka "Something went wrong but we're not going to tell you what")
        # So we raise an HTTPException that contains the same details as the original Exception and FastAPI will send to the client.
        raise HTTPException(status_code=500, detail=str(e))

    