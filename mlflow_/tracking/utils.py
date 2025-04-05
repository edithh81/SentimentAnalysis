import os
import mlflow
import mlflow.pytorch


def setup_mlflow(experiment_name="vietnamese-sentiment-analysis", tracking_uri="http://mlflow:5000"):
    """
    Set up MLflow experiment and tracking URI.
    """
    mlflow.set_tracking_uri("http://mlflow:5000")
    try:
        experiment_id = mlflow.create_experiment(experiment_name)
    except:
        experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
    mlflow.set_experiment(experiment_name)
    return experiment_id


def log_params(params):
    mlflow.log_params(params)


def log_metrics(metrics, epoch):
    mlflow.log_metrics(metrics, step=epoch)


def log_model(model, model_name="sentiment_model", model_path="models/"):
    mlflow.pytorch.log_model(model, model_name)
    # log model artifacts
    mlflow.log_artifact(model_path, artifact_path=model_name)
    mlflow.log_artifact(model_path +"vocab.pth", artifact_path=model_name)


