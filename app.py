import os
import re
import string
import torch
import torch.nn as nn
import torch.nn.functional as F
import emoji
import numpy as np
import streamlit as st
import pyvi
from pyvi import ViTokenizer
from torchtext import vocab
from torchtext.vocab import build_vocab_from_iterator, Vocab
from torch.nn.utils.rnn import pad_sequence
def vn_tokenizer(text):
    return ViTokenizer.tokenize(text).split()
idx2label = {0: 'negative', 1:'positive'}
viet_luong_dict = {
    "ko": "không", "k": "không", "hok": "không", "j": "gì",
    "tl": "trả lời", "ntn": "như thế nào", "vkl": "rất",
    "cx": "cũng", "z": "d", "thik": "thích"
}


def replace_teencode(text):
    words = text.split()
    return " ".join([viet_luong_dict[word] if word in viet_luong_dict else word for word in words])

def preprocess_text(text):
    text = text.lower()
    
    text = replace_teencode(text)
    
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    text = url_pattern.sub(r" ", text)
    
    html_pattern = re.compile(r'<.*?>')
    text = html_pattern.sub(r" ", text)
    
    text = emoji.demojize(text)
    
    text = " ".join(text.split())
    return text

class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_size, kernel_size, num_filters, num_classes):
        super(TextCNN, self).__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.num_filters = num_filters
        self.num_classes = num_classes
        self.embedding = nn.Embedding(self.vocab_size, self.embed_size, padding_idx=0)
        self.conv = nn.ModuleList([
            nn.Conv1d(in_channels=embed_size,
                      out_channels=num_filters,
                      kernel_size=k,
                      stride = 1)
            for k in  kernel_size
        ])
        self.fc = nn.Linear(len(kernel_size) * num_filters, num_classes)
    def forward(self, x):
        batch_size, sequence_length = x.shape
        x = self.embedding(x)
        x = x.permute(0, 2, 1)
        x = [F.relu(conv(x)) for conv in self.conv]
        x = [F.max_pool1d(i, i.size(-1)).squeeze(-1) for i in x]
        x = torch.cat(x, dim=1)
        x = self.fc(x)
        return x
    
def load_model(model_path, vocab_size = 1000, embedding_dim = 100, numclasses = 2):
    model = TextCNN(
        vocab_size=vocab_size,
        embed_size=embedding_dim, 
        kernel_size=[2, 3, 4], 
        num_filters=100, 
        num_classes= numclasses)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_model('textCNN_best_model.pt')
vocabulary = torch.load('vocab_textCNN.pth')
def predict_sentiment(sentence, model, vocab, max_length=10):  # Ensure min length
    # Tokenize and convert tokens to indices
    encoded_sentence = [vocab[token] for token in vn_tokenizer(sentence)]
    encoded_sentence = torch.tensor(encoded_sentence, dtype=torch.int64).unsqueeze(0)  # (1, seq_len)

    # Ensure padding to at least max_length
    if encoded_sentence.shape[1] < max_length:
        pad_size = max_length - encoded_sentence.shape[1]
        encoded_sentence = F.pad(encoded_sentence, (0, pad_size), value=vocab['<pad>'])

    with torch.no_grad():
        output = model(encoded_sentence)

    output = nn.Softmax(dim=1)(output)
    p_max, y_hat = torch.max(output.data, 1)
    return round(p_max.item(), 2), idx2label[y_hat.item()]


def main():
    st.title("Sentiment Analysis")
    st.title("Model TextCNN. Dataset: NTC-SCV")
    sentence = st.text_input("Enter your sentence (In Vietnammese):")
    if st.button("Predict"):
        prob, label = predict_sentiment(sentence, model, vocabulary)
        if prob > 0.9:
            st.success("Sentiment: {} with probability: {}".format(label, prob))
        else:
            st.info("Sentiment: {} with probability: {}".format(label, prob))

if __name__ == '__main__':
    main()
