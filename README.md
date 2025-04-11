# DSBA Platform

A toy MLOps Platform for educational purposes

## Project Structure

This project has a fairly standard structure (but it is still adapted to be simplified compared to a slightly more typical structure):

- a `pyproject.toml` file contains the project metadata, including the dependencies. It is common to see a "setup.py" file in Python projects but we use this more modern approach to define the project metadata.
- The `src` folder contains the code code (dsba) as well as the code for the CLI, the API, the web app, the notebooks, as well as the Dockerfiles.
- The `tests` folder contains some unit and integration tests
- `.gitignore` is a special file name that will be detected by git. This file contains a list of files and folders that should not be committed to the repository. For example (see below for setup), the `.env` file is specific to your own deployment so it should not be committed to the repository (it may contain specific file paths that are only meaningful on your machine, and it may contain secrets like API keys - API keys and passwords should never be stored in a git repository).

## Installation (dev mode)

### Requirements

Your machine should have the following software installed:

- Python 3.12
- git
- to use the model training notebook (not required), you may need to install openmp (libomp) which is required by xgboost. But you can also not use the model_training module from this example or adapt it to use scikit-learn rather than xgboost.

### Clone the repository

- The first things to do is to copy this repository, to have a copy that you own on GitHub. This is because you are not allowed to push directly to the main repository owned by Joachim. Copying a repository on GitHub to have your own is called a "fork". You should understand that "forking" and "cloning" are not the same. Forking is a GitHub concept to copy a repository in your own GitHub account. Cloning basically means "downloading for the first time a repo to your computer". Just click on the fork button above when seeing this document from GitHub.

- Move into the folder you want to work in (I saw many students not choosing a folder and just working in their home directory, you don't want to do that)

- To be certain things are ok type:

```bash
git status
```

This should fail and tell you there is no repository at this location. I saw many students trying to clone a repository inside a repository, you also don't want to be in this situation.

Now you can clone the repository:

```bash
git clone <the address of your fork>
```

### Installing the project

cd into the repository folder.

Create a virtual environment with the following command (for windows, python, not python3).
Using the name ".venv" for your virtual environment is recommended.
It is quite standard and tools like vscode will automatically find it.

```bash
python3 -m venv .venv
```

Install dependencies (as specified in pyproject.toml):

```bash
pip install -e .
```

This will install the project in editable mode, meaning that any changes you make to the code will be reflected in your local environment.

## Running the tests

To run the tests, you can use the following command:

```bash
pytest
```

This will run all the tests in the `tests` folder.

## Usage

You must set the environment variables `DSBA_MODELS_ROOT_PATH` and `DSBA_REPORTS_ROOT_PATH` to the addresses you want to respectively store the models and reports in before you can use the platform.

For example as a MacOS user I set `/Users/sofia/dev/dsba/models` and `/Users/sofia/dev/dsba/reports`.

There are many ways to set environment variables depending on the context.

In a python notebook, you can use the following code:

```python
import os
os.environ["DSBA_MODELS_ROOT_PATH"] = "/path/to/your/models"
```

In a terminal or shell script, you can use the following code (Linux and MacOS):

```bash
export DSBA_MODELS_ROOT_PATH="/path/to/your/models"
```

For windows, something of the sort may work:

```bash
set DSBA_MODELS_ROOT_PATH="C:\path\to\your\models"
```
Don't forget to do the same for DSBA_REPORTS_ROOT_PATH and you are all set

## CLI

List models registered on your system:

```bash
src/cli/dsba_cli list
```

Use a model to predict on a file:

```bash
src/cli/dsba_cli predict --input /path/to/your/data/file.csv --output /path/to/your/output/file.csv --model-id your_model_id
```

Run an experiment training and testing a model with a dataset

```bash
src/cli/dsba_cli run --input "https://raw.githubusercontent.com/datasciencedojo/datasets/refs/heads/master/titanic.csv" --target "Survived" --model-id "titanic_model"
```

Generate a report that benchmarks trained models on a specified performance metric

```bash
src/cli/dsba_cli report --metric "accuracy"
```

### Notebook

To visualize some results and get an idea of how to use main ML functions

### API

Run API
```bash
uvicorn src.api.api:app --reload
```

Open browser and connect to http://127.0.0.1:8000/docs
Play around!

### Dockerized API

To use the Dockerized API you do not even need prior steps besides cloning the repo and opening Docker

Build the Docker image
```bash
docker build -t dsba-mlops-app -f src/api/Dockerfile.api .
```

Run API
```bash
docker run -p 8000:8000 dsba-mlops-app
```

Open browser and connect to http://localhost:8000/docs
Play around!