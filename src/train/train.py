import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.data.functional import to_map_style_dataset
from torch.nn.utils.rnn import pad_sequence
import pyvi
from pyvi import ViTokenizer
import pandas as pd
import mlflow

from src.model.textcnn import TextCNN
from src.preprocessing.preprocessing import TextPreprocessor

from mlflow_.registry.utils import register_model
from mlflow_.tracking.utils import setup_mlflow, log_params, log_metrics, log_model


class SentimentDataset(Dataset):
    def __init__(self, texts, labels, preprocessor):
        self.texts = texts
        self.labels = labels
        self.preprocessor = preprocessor
    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoded = self.preprocessor.encode_text(text)
        return encoded, label
import traceback 
def prepare_dataset(df, vocabulary):
    vn_tokenizer = ViTokenizer.tokenize
    for idx, row in df.iterrows():
        sentence = row['sentence']
        encoded_sentence = vocabulary(vn_tokenizer(sentence).split())
        label = row['label']
        yield (encoded_sentence, label)
        
def collate_batch(batch, vocabulary):
    encoded_sentences, labels = [], []
    for encoded_sentence, label in batch:
        labels.append(label)
        encoded_sentence = torch.tensor(encoded_sentence, dtype=torch.int64)
        encoded_sentences.append(encoded_sentence)
    labels = torch.tensor(labels, dtype=torch.int64)
    encoded_sentences = pad_sequence(encoded_sentences, padding_value=vocabulary['<pad>'])
    return encoded_sentences, labels


def yield_tokens(texts, preprocessor):
    for text in texts:
        yield preprocessor.tokenize(text)
        
def build_vocabulary(texts, preprocessor, min_freq = 2, max_tokens = 10000):
    vocab = build_vocab_from_iterator(
        yield_tokens(texts, preprocessor),
        min_freq=min_freq,
        specials=["<unk>", "<pad>"],
        special_first=True,
        max_tokens=max_tokens,
    )
    vocab.set_default_index(vocab["<unk>"])
    return vocab

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
    
    return total_loss / len(dataloader), correct / total

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    return total_loss / len(dataloader), correct / total

def train_model(train_path, val_path, params, output_dir='/opt/airflow/models'):
    experiment_id = setup_mlflow()
    
    with mlflow.start_run(experiment_id=experiment_id) as run:
        log_params(params)
        
        train_df = pd.read_csv(train_path)
        val_df = pd.read_csv(val_path)
        assert len(train_df) > 0, "Training data is empty"
        assert len(val_df) > 0, "Validation data is empty"
        preprocessor = TextPreprocessor() # must load vocabl file
        # Create both vocab and model directories
        vocab_dir = os.path.join(output_dir, 'vocab')  
        artifacts_dir = os.path.join(output_dir, 'artifacts')

        # Create directories
        os.makedirs(vocab_dir, exist_ok=True)
        os.makedirs(artifacts_dir, exist_ok=True)
        if params['build_vocab']:
            print("Building vocabulary...")
            vocab = build_vocabulary(
                train_df['sentence'],
                preprocessor,
                min_freq=params['min_freq'],
                max_tokens=params['vocab_size'] if 'vocab_size' in params else params.get('max_tokens', 10000)
            )
            preprocessor.vocab = vocab
            vocab_path = os.path.join(vocab_dir, 'vocab_textCNN.pth')
            torch.save(vocab, vocab_path)
            print(f"Vocabulary saved to {vocab_path}")
        else:
            vocab = torch.load(params['vocab_path'])
            preprocessor.vocab = vocab

        # Create datasets with better error handling
        train_dataset = to_map_style_dataset(prepare_dataset(train_df, preprocessor.vocab))
        val_dataset = to_map_style_dataset(prepare_dataset(val_df, preprocessor.vocab))
        def custom_collate_fn(batch):
            return collate_batch(batch, preprocessor.vocab)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=params['batch_size'],
            shuffle=True,
            collate_fn=custom_collate_fn,
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=params['batch_size'],
            shuffle=False,
            collate_fn=custom_collate_fn,
        )
        
        # init model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = TextCNN(
            vocab_size=len(preprocessor.vocab),
            embed_size=params['embed_size'],
            num_filters=params['num_filters'],
            num_classes=params['num_classes'], 
            kernel_size=params['kernel_sizes']
        ).to(device)

        best_val_loss = float('inf')
        # init parameters
        optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'])
        criterion = nn.CrossEntropyLoss()
        best_model_state_dict = None
        
        # Save the best model path
        best_model_path = os.path.join(artifacts_dir, 'textCNN_best_model.pt')
        
        for epoch in range(params['epochs']):
            try:
                train_loss, train_acc = train_epoch(
                    model,
                    train_loader,
                    criterion,
                    optimizer,
                    device
                )
                
                val_loss, val_acc = evaluate(
                    model,
                    val_loader,
                    criterion,
                    device
                )
                
                metrics = {
                    'train_loss': train_loss,
                    'train_accuracy': train_acc,
                    'val_loss': val_loss,
                    'val_accuracy': val_acc
                }
                log_metrics(metrics, epoch)
                
                print(f"Epoch {epoch+1}/{params['epochs']}")
                print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}")
                print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}")
                
                # save model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_state_dict = model.state_dict()
                    # Save the model immediately to avoid memory issues
                    torch.save(best_model_state_dict, best_model_path)
                    print(f"Model saved at epoch {epoch+1} to {best_model_path}")
            except Exception as e:
                print(f"Error in epoch {epoch+1}: {e}")
                traceback.print_exc()
                continue
        
        # Load the best model
        try:
            model.load_state_dict(torch.load(best_model_path))
            print(f"Loaded best model from {best_model_path}")
        except Exception as e:
            print(f"Error loading best model: {e}")
            traceback.print_exc()
        
        # Log model to MLflow
        try:
            log_model(model, output_dir)
            
            # Register model if requested
            if params.get('register_model'):
                register_model(model, output_dir, run.info.run_id)
        except Exception as e:
            print(f"Error logging model to MLflow: {e}")
            traceback.print_exc()
            
        print("Training complete.")
        return model, preprocessor



