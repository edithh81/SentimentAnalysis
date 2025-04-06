from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.utils.dates import days_ago
import pandas as pd
import os
import sys
import mlflow
import yaml
import torch
from src.preprocessing.preprocessing import TextPreprocessor
from src.train.train import train_model
from pyvi import ViTokenizer
from torchtext.data.functional import to_map_style_dataset
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
    'start_date': days_ago(1),
}

def load_config(config_name: str):
    config_path = 'config/sentiment_config.yaml'
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file {config_path} not found")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        if config_name:
            config = config.get(config_name, {})
            if not config:
                raise ValueError(f"Config {config_name} not found in {config_path}")
    return config


def extract_data(**kwargs):
    """Extract data from the source."""
    import os
    import shutil
    from langid.langid import LanguageIdentifier, model
    
    os.makedirs('/opt/airflow/data/raw', exist_ok=True)
    os.makedirs('/opt/airflow/data/processed', exist_ok=True)
    os.makedirs('/opt/airflow/models/artifacts', exist_ok=True)
    os.makedirs('/opt/airflow/models/vocab', exist_ok=True)
    
    # Copy raw data files to data/raw directory if needed
    source_dir = os.environ.get('DATA_SOURCE_DIR', '/opt/airflow/ntc-scv/data_train/train')
    dest_dir = os.environ.get('DATA_DEST_DIR', '/opt/airflow/data/raw/train')
    
    val_dir = '/opt/airflow/ntc-scv/data_train/test'
    val_dir_dest = '/opt/airflow/data/raw/val'
    test_dir = '/opt/airflow/ntc-scv/data_test/test'
    test_dir_dest = '/opt/airflow/data/raw/test'
    if not os.path.exists(dest_dir):
        shutil.copytree(source_dir, dest_dir)
    if not os.path.exists(val_dir_dest):
        shutil.copytree(val_dir, val_dir_dest)
    if not os.path.exists(test_dir_dest):
        shutil.copytree(test_dir, test_dir_dest)
    return {'extract_path': dest_dir}

def transform_data(**kwargs):
    """Process the data and prepare it for training."""
    ti = kwargs['ti']
    extract_path = ti.xcom_pull(task_ids='extract_data')['extract_path']
    
    def load_data_from_path(folder_path):
        examples = []
        for label in os.listdir(folder_path):
            full_path = os.path.join(folder_path, label)
            for file_name in os.listdir(full_path):
                file_path = os.path.join(full_path, file_name)
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        lines = f.readlines()
                    sentence = " ".join(lines)
                    if label == "neg":
                        label_value = 0
                    elif label == "pos":
                        label_value = 1
                    else:
                        continue
                    
                    data = {
                        'sentence': sentence,
                        'label': label_value
                    }
                    examples.append(data)
                except:
                    # Skip problematic files
                    continue
        return pd.DataFrame(examples)   
    
    # Load data
    dataconfig = load_config('dataconfig')
    
    train_df = load_data_from_path(dataconfig['data_train_path'])
    valid_df = load_data_from_path(dataconfig['data_val_path'])
    test_df = load_data_from_path(dataconfig['data_test_path'])
    
    # Filter out non-Vietnamese text
    from langid.langid import LanguageIdentifier, model
    
    def identify_VN(df):
        identifier = LanguageIdentifier.from_modelstring(model, norm_probs=True)
        not_vi = set()
        THRESHOLD = 0.9
        for idx, row in df.iterrows():
            lang, score = identifier.classify(row['sentence'])
            if lang != 'vi' or (lang == 'vi' and score < THRESHOLD):
                not_vi.add(idx)
        
        vi_df = df[~df.index.isin(not_vi)]
        return vi_df
    
    train_df = identify_VN(train_df)
    valid_df = identify_VN(valid_df)
    test_df = identify_VN(test_df)
    
    
    # Preprocess text
    preprocessor = TextPreprocessor()
    train_df['sentence'] = train_df['sentence'].apply(preprocessor.preprocess_text)
    valid_df['sentence'] = valid_df['sentence'].apply(preprocessor.preprocess_text)
    test_df['sentence'] = test_df['sentence'].apply(preprocessor.preprocess_text)
    # Save processed data
    train_df.to_csv('/opt/airflow/data/processed/train.csv', index=False)
    valid_df.to_csv('/opt/airflow/data/processed/val.csv', index=False)
    test_df.to_csv('/opt/airflow/data/processed/test.csv', index=False)
    
    return {
    'train_path': '/opt/airflow/data/processed/train.csv', 
    'val_path': '/opt/airflow/data/processed/val.csv',
    'test_path': '/opt/airflow/data/processed/test.csv'
    }

def train(**kwargs):
    """Train the model."""
    ti = kwargs['ti']
    data_paths = ti.xcom_pull(task_ids='transform_data')
    train_path = data_paths['train_path']
    val_path = data_paths['val_path']
    
    # load config for training
    trainconfig = load_config('trainconfig')
    modelconfig = load_config('modelconfig')
    
    # # Parameters for training
    # params = {
    #     'vocab_size': 1000,
    #     'embed_size': 100,
    #     'kernel_size': [2, 3, 4],
    #     'num_filters': 100,
    #     'num_classes': 2,
    #     'batch_size': 128,
    #     'learning_rate': 0.001,
    #     'epochs': 10,
    #     'build_vocab': True,
    #     'min_freq': 2,
    #     'register_model': True
    # }
    for key, value in modelconfig.items():
        trainconfig[key] = value
    # Train the model
    model, preprocessor,vocab_dir, artifacts_dir= train_model(train_path, val_path, trainconfig, output_dir='/opt/airflow/models')
    client = mlflow.tracking.MlflowClient()
    run_id = client.search_runs(
        experiment_ids=[mlflow.get_experiment_by_name("vietnamese-sentiment-analysis").experiment_id],
        filter_string="",
        order_by=["start_time DESC"],
        max_results=1
    )[0].info.run_id
    # log to current run
    artifacts_dir = os.path.join(artifacts_dir, 'textCNN_best_model.pt')
    vocab_dir = os.path.join(vocab_dir, 'vocab_textCNN.pth')
    with mlflow.start_run(run_id=run_id):
        mlflow.log_artifact(artifacts_dir, artifact_path="artifacts")
        mlflow.log_artifact(vocab_dir, artifact_path="vocab")
        model_uri = f"runs:/{run_id}/model"
        result = mlflow.register_model(model_uri, "textcnn-sentiment")

        # Move the model to a specific stage (e.g., Staging)
        client = mlflow.tracking.MlflowClient()
        client.transition_model_version_stage(
            name="textcnn-sentiment",
            version=result.version,
            stage="Staging"
        )
    return {
        'model_path': '/opt/airflow/models/artifacts/textCNN_best_model.pt',
        'vocab_path': '/opt/airflow/models/vocab/vocab_textCNN.pth'
    }

def evaluate(**kwargs):
    """Evaluate the model on test data."""
    ti = kwargs['ti']
    model_paths = ti.xcom_pull(task_ids='train')
    model_path = model_paths['model_path']
    vocab_path = model_paths['vocab_path']
    test_path  = ti.xcom_pull(task_ids='transform_data')['test_path']
    import torch
    from src.model.textcnn import load_model
    modelconfig = load_config('modelconfig')
    # Load the model and vocabulary
    vocab = torch.load(vocab_path)
    model = load_model(model_path, vocab_size=len(vocab), embed_dim=modelconfig['embed_size'],numfilters=modelconfig['num_filters'] ,numclasses=modelconfig['num_classes'], kernel_size=modelconfig['kernel_sizes'])
    
    dataconfig = load_config('dataconfig')
    # Load test data
    test_df = pd.read_csv(test_path)  # Using validation as test for simplicity
    
    # Set up the preprocessor
    preprocessor = TextPreprocessor()
    preprocessor.vocab = vocab
    
    # Create a test dataset and evaluate
    from src.train.train import SentimentDataset, collate_batch, evaluate
    def custom_collate_fn(batch):
        return collate_batch(batch, preprocessor.vocab)
    def prepare_dataset(df, vocabulary):
        vn_tokenizer = ViTokenizer.tokenize
        for idx, row in df.iterrows():
            sentence = row['sentence']
            encoded_sentence = vocabulary(vn_tokenizer(sentence).split())
            label = row['label']
            yield (encoded_sentence, label)
    test_dataset = to_map_style_dataset(
        prepare_dataset(test_df, preprocessor.vocab))
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=128,
        shuffle=False,
        collate_fn=custom_collate_fn,
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = torch.nn.CrossEntropyLoss()
    
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    
    # Log to MLflow
    client = mlflow.tracking.MlflowClient()
    run_id = client.search_runs(
        experiment_ids=[mlflow.get_experiment_by_name("vietnamese-sentiment-analysis").experiment_id],
        filter_string="",
        order_by=["start_time DESC"],
        max_results=1
    )[0].info.run_id
    
    # log to current run
    with mlflow.start_run(run_id=run_id):
        mlflow.log_metric("test_loss", test_loss)
        mlflow.log_metric("test_accuracy", test_acc)
    
    print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")
    
    return {
        'test_loss': test_loss,
        'test_accuracy': test_acc,
        'model_path': model_path,
        'vocab_path': vocab_path
    }

def deploy_model(**kwargs):
    """Deploy the model to production."""
    ti = kwargs['ti']
    model_path = ti.xcom_pull(task_ids='evaluate')['model_path']
    eval_results = ti.xcom_pull(task_ids='evaluate')
    vocab_path = eval_results['vocab_path']
    current_test_loss = eval_results['test_loss']
    current_test_accuracy = eval_results['test_accuracy']
    import torch
    from src.model.textcnn import load_model
    modelconfig = load_config('modelconfig')
    # Load the model and vocabulary
    vocab = torch.load(vocab_path)
    model = load_model(model_path, vocab_size=len(vocab), embed_dim=modelconfig['embed_size'],numfilters=modelconfig['num_filters'] ,numclasses=modelconfig['num_classes'], kernel_size=modelconfig['kernel_sizes'])
    client = mlflow.tracking.MlflowClient()
    deploy_model = True
    
        # get the latest model version
    production_models = client.get_latest_versions(
        name = "textcnn-sentiment",
        stages = ["Production"]
    )
        
    if production_models:
        production_model = production_models[0]
        prod_run_id = production_model.run_id
            
            # get the latest test loss and accuracy
        prod_run = client.get_run(prod_run_id)
        prod_metrics = prod_run.data.metrics
        prod_test_loss = prod_metrics.get('test_loss', float('inf'))
            
        print(f'Production model test loss: {prod_test_loss}')
        print(f'Current model test loss: {current_test_loss}')
        if current_test_loss > prod_test_loss:
            deploy_model = False
            print("Current model is worse than the production model. Not deploying.")
    else: 
        print("No production model found. Deploying current model.")
        latest_version = client.get_latest_versions("textcnn-sentiment")[0].version
        torch.save(model.state_dict(), f"models/artifacts/textCNN_best_model_{latest_version}.pt")
        torch.save(vocab, f"models/vocab/vocab_textCNN_{latest_version}.pth")
        client.transition_model_version_stage(
            name="textcnn-sentiment",
            version=latest_version,
            stage="Production",
            archive_existing_versions=True
        )
        print(f"Model version {latest_version} deployed to production.")
        deploy_model = False
        
    if deploy_model and current_test_accuracy > 0.85:
        latest_version = client.get_latest_versions("textcnn-sentiment", stages=["Production"])[0].version
        torch.save(model.state_dict(), f"models/artifacts/textCNN_best_model_{latest_version}.pt")
        client.transition_model_version_stage(
            name="textcnn-sentiment",
            version=latest_version,
            stage="Production",
            archive_existing_versions=True
        )
        print(f"Model version {latest_version} deployed to production.")
        return {'deployed': True, 'version': latest_version}
    else:
        print("Model not deployed. Either accuracy is too low or the current model is worse than the production model.")
        reason = "Model not deployed. Either accuracy is too low or the current model is worse than the production model."
        return {'deployed': False, 'reason': reason}

        
    


with DAG(
    'sentiment_analysis_training',
    default_args=default_args,
    description='Train Vietnamese sentiment analysis model',
    schedule_interval=timedelta(days=7),
    start_date=days_ago(1),
    catchup=False,
    tags=['nlp', 'sentiment'],
) as dag:
    
    extract_task = PythonOperator(
        task_id='extract_data',
        python_callable=extract_data,
    )
    
    transform_task = PythonOperator(
        task_id='transform_data',
        python_callable=transform_data,
    )
    
    train_task = PythonOperator(
        task_id='train',
        python_callable=train,
    )
    
    evaluate_task = PythonOperator(
        task_id='evaluate',
        python_callable=evaluate,
    )
    
    deploy_task = PythonOperator(
        task_id='deploy_model',
        python_callable=deploy_model,
    )
    
    # Set task dependencies
    extract_task >> transform_task >> train_task >> evaluate_task >> deploy_task
    