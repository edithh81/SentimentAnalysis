import re
import emoji
from pyvi import ViTokenizer

class TextPreprocessor:
    def __init__(self, vocab_file=None):
        self.viet_luong_dict = {
            "ko": "không", "k": "không", "hok": "không", "j": "gì",
            "tl": "trả lời", "ntn": "như thế nào", "vkl": "rất",
            "cx": "cũng", "z": "d", "thik": "thích"
        }
        self.vocab = None
        if vocab_file:
            self.load_vocab(vocab_file)
    
    def tokenize(self, text):
        """Tokenize Vietnamese text."""
        return ViTokenizer.tokenize(text).split()
        
    def replace_teencode(self, text):
        """Replace Vietnamese teen code with standard words."""
        words = text.split()
        return " ".join([self.viet_luong_dict[word] if word in self.viet_luong_dict else word for word in words])
    
    def preprocess_text(self, text):
        """Preprocess text for sentiment analysis."""
        text = text.lower()
        
        text = self.replace_teencode(text)
        
        url_pattern = re.compile(r'https?://\S+|www\.\S+')
        text = url_pattern.sub(r" ", text)
        
        html_pattern = re.compile(r'<.*?>')
        text = html_pattern.sub(r" ", text)
        
        text = emoji.demojize(text)
        
        text = " ".join(text.split())
        return text
    
    def encode_text(self, text, max_length=None):
        """Encode text using vocabulary."""
        if not self.vocab:
            raise ValueError("Vocabulary not loaded. Call with vocab_file or load_vocab first.")
            
        import torch
        import torch.nn.functional as F
        
        # Preprocess and tokenize
        processed_text = self.preprocess_text(text)
        tokens = self.tokenize(processed_text)
        
        # Convert tokens to indices with more robust error handling
        try:
            if hasattr(self.vocab, '__call__'):
                # torchtext 0.14+ style vocabulary
                encoded = self.vocab(tokens)
            elif hasattr(self.vocab, 'get_stoi'):
                # torchtext 0.13+ style vocabulary
                stoi = self.vocab.get_stoi()
                unk_idx = stoi.get('<unk>', 0)
                encoded = [stoi.get(token, unk_idx) for token in tokens]
            elif hasattr(self.vocab, '__getitem__'):
                # Dictionary-like vocab
                encoded = [self.vocab[token] for token in tokens]
            else:
                # Fallback case - try direct indexing
                encoded = [self.vocab[token] for token in tokens]
        except Exception as e:
            import traceback
            print(f"Error encoding text: {e}")
            traceback.print_exc()
            # Return empty encoding as fallback
            encoded = [0]  # Just use the unk token
            
        encoded_tensor = torch.tensor(encoded, dtype=torch.int64).unsqueeze(0)  # (1, seq_len)
        
        # Pad if needed
        if max_length and encoded_tensor.shape[1] < max_length:
            pad_value = 0  # Default to 0 (usually <pad> token)
            if hasattr(self.vocab, '__getitem__'):
                try:
                    pad_value = self.vocab['<pad>']
                except:
                    pass
            
            pad_size = max_length - encoded_tensor.shape[1]
            encoded_tensor = F.pad(encoded_tensor, (0, pad_size), value=pad_value)
            
        return encoded_tensor
        
    def load_vocab(self, vocab_file):
        """Load vocabulary from file with better error handling."""
        try:
            import torch
            self.vocab = torch.load(vocab_file)
            # Test if vocab is usable
            if hasattr(self.vocab, '__getitem__'):
                _ = self.vocab['<unk>']  # Try accessing a token
            return self.vocab
        except Exception as e:
            import traceback
            print(f"Error loading vocabulary from {vocab_file}: {e}")
            traceback.print_exc()
            
            # Create a minimal functional vocab as fallback
            print("Creating fallback vocabulary")
            fallback_vocab = {'<unk>': 0, '<pad>': 1}
            self.vocab = type('FallbackVocab', (), {
                '__getitem__': lambda self, token: fallback_vocab.get(token, 0),
                '__len__': lambda self: len(fallback_vocab)
            })()
            return self.vocab