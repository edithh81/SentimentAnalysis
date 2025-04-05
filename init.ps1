Write-Host "Waiting for services to start up..."
Start-Sleep -Seconds 30

# Initialize Airflow database
docker-compose exec airflow-webserver airflow db init

# Create Airflow admin user
docker-compose exec airflow-webserver airflow users create `
    --username admin `
    --password admin `
    --firstname Admin `
    --lastname User `
    --role Admin `
    --email admin@example.com

# Install requirements in Airflow containers
docker-compose exec airflow-webserver pip install torch==2.0.1 pyvi==0.1.1 emoji==2.6.0 pandas==2.0.2 mlflow==2.4.1 langid==1.1.6 scikit-learn==1.2.2 pyyaml==6.0

docker-compose exec airflow-scheduler pip install torch==2.0.1 pyvi==0.1.1 emoji==2.6.0 pandas==2.0.2 mlflow==2.4.1 langid==1.1.6 scikit-learn==1.2.2 pyyaml==6.0

# Unpause the DAG
docker-compose exec airflow-webserver airflow dags unpause sentiment_analysis_training

Write-Host "Setup complete!"