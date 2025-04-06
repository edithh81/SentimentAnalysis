import os
import mlflow
import mlflow.pytorch

def register_model(model, model_name="textcnn-sentiment", run_id = "", stage='Staging'):
    model_uri = f"runs:/{run_id}/model"
    result = mlflow.register_model(model_uri, model_name)
    
    client = mlflow.tracking.MlflowClient()
    client.transition_model_version_stage(
        name=model_name,
        version=result.version,
        stages=[stage]
    )
    print(f"Model registered in stage: {stage}")
    return result

    