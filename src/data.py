import os
import urllib.request
import torch

class TinyShakespeareDataset:
    def __init__(self, data_dir='.', block_size=128):
        self.data_dir = data_dir
        self.file_path = os.path.join(data_dir, 'input.txt')
        self.block_size = block_size
        self.download_if_not_exists()
        
        with open(self.file_path, 'r', encoding='utf-8') as f:
            text = f.read()
            
        chars = sorted(list(set(text)))
        vocab_size = len(chars)
        assert vocab_size == 65, f"Vocabulary size must be 65, got {vocab_size}"
        
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}
        self.vocab_size = vocab_size
        
        data = torch.tensor(self.encode(text), dtype=torch.long)
        n = int(0.9 * len(data))
        self.train_data = data[:n]
        self.val_data = data[n:]
        
    def download_if_not_exists(self):
        if not os.path.exists(self.file_path):
            url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
            print(f"Downloading {url} to {self.file_path}...")
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req) as response, open(self.file_path, 'wb') as out_file:
                out_file.write(response.read())
            
    def encode(self, s):
        return [self.stoi[c] for c in s]
        
    def decode(self, l):
        return ''.join([self.itos[i] for i in l])
        
    def get_batch(self, split, batch_size):
        data = self.train_data if split == 'train' else self.val_data
        ix = torch.randint(len(data) - self.block_size, (batch_size,))
        x = torch.stack([data[i:i+self.block_size] for i in ix])
        y = torch.stack([data[i+1:i+self.block_size+1] for i in ix])
        return x, y
