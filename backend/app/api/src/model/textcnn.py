import torch.nn as nn
import torch.nn.functional as F
import torch
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
def load_model(model_path, vocab_size = 10000, embed_dim = 200, numfilters = 100 ,numclasses = 2):
    model = TextCNN(
        vocab_size = vocab_size,
        embed_size = embed_dim,
        kernel_size = [2, 3, 4],
        num_filters = numfilters,
        num_classes = numclasses
    )
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

def predict_sentiment(text, model, vocab, preprocessor = None, max_length=10):
    # Preprocess and encode the text
    preprocessor.vocab = vocab
    encoded_text = preprocessor.encode_text(text, max_length=max_length) # load preprocessor from src/preprocessing/preprocessing.py
    
    # Move the tensor to the same device as the model
    device = next(model.parameters()).device
    encoded_text = encoded_text.to(device)
    
    # Make prediction
    with torch.no_grad():
        output = model(encoded_text)
    output = nn.Softmax(dim=1)(output)
    p_max, y_hat = torch.max(output.data, 1)
    idx2label = {0: 'negative', 1: 'positive'}  # Example mapping, adjust as needed
    
    return round(p_max.item(), 2), idx2label[y_hat.item()]