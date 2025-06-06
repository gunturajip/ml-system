### Create venv (a new environment) for every ML project
1. python --version
2. python -m venv venv
3. .\venv\Scripts\activate
4. mlflow server --host 127.0.0.1 --port 5000
5. mlflow models generate-dockerfile -m runs:/a56efe1f7db34571a01dc193e1a29267/model -d credit_scoring_docker

### Errors solutions
1. Please define MLFLOW_TRACKING_URI in your system environment variable first for running any mlflow commands related to the mlflow remote local server