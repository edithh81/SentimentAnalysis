import os
import mlflow


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


def log_model(model, model_name="sentiment_model", model_path="models/", input_shape = None):
    mlflow.pytorch.log_model(model,
                             artifact_path='model',
                             registered_model_name=model_name,
                             input_examples=input_shape)
    vocab_path = os.path.join(model_path, 'vocab')
    vocab_path = os.path.join(vocab_path, 'vocab_textCNN.pth')
    mlflow.log_artifact(vocab_path, artifact_path='vocab')


