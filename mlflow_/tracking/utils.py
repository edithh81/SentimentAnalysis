import os
import mlflow


def setup_mlflow(experiment_name="vietnamese-sentiment-analysis", tracking_uri="http://mlflow:5000"):
    """
    Set up MLflow experiment and tracking URI.
    """
    mlflow.set_tracking_uri(tracking_uri)
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


def log_model(model, model_name="sentiment_model", model_path="models/", input_shape=None):
    """
    Log a PyTorch model to MLflow with proper artifact paths and model registry name.
    
    Args:
        model: PyTorch model to log
        model_name: Name to register the model under
        model_path: Base path where model artifacts are stored
        input_shape: Sample input for model signature
    """
    print(f"Starting model logging with name: {model_name}")
    
    # Ensure vocab path exists before logging
    vocab_path = os.path.join(model_path, 'vocab', 'vocab_textCNN.pth')
    if not os.path.exists(vocab_path):
        print(f"WARNING: Vocab file not found at {vocab_path}")
    else:
        print(f"Logging vocabulary file from {vocab_path}")
        mlflow.log_artifact(vocab_path, artifact_path='vocab')
    
    # Log the saved model state dict
    model_file_path = os.path.join(model_path, 'artifacts', 'textCNN_best_model.pt')
    if not os.path.exists(model_file_path):
        print(f"WARNING: Model file not found at {model_file_path}")
    else:
        print(f"Logging model file from {model_file_path}")
        mlflow.log_artifact(model_file_path, artifact_path='model_state_dict')
    
    # Log the model with MLflow's PyTorch integration
    try:
        if input_shape is not None:
            from mlflow.models.signature import infer_signature
            # Generate predictions for signature
            with torch.no_grad():
                prediction = model(torch.from_numpy(input_shape).cpu())
            signature = infer_signature(input_shape, prediction.detach().numpy())
            
            model_info = mlflow.pytorch.log_model(
                pytorch_model=model,
                artifact_path='pytorch-model',
                signature=signature,
                registered_model_name=model_name
            )
            print(f"Logged PyTorch model with signature, URI: {model_info.model_uri}")
        else:
            model_info = mlflow.pytorch.log_model(
                pytorch_model=model,
                artifact_path='pytorch-model',
                registered_model_name=model_name
            )
            print(f"Logged PyTorch model without signature, URI: {model_info.model_uri}")
    except Exception as e:
        print(f"Error logging PyTorch model: {e}")
        # Fallback to basic logging
        mlflow.pytorch.log_model(
            pytorch_model=model,
            artifact_path='pytorch-model'
        )
        print("Used fallback model logging method")


