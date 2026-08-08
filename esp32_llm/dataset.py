import os
import urllib.request
import torch
from torch.utils.data import Dataset, DataLoader
from training.tokenizer import PoetryCharTokenizer

DEFAULT_POETRY_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

def download_sample_poetry_dataset(output_path: str):
    """Downloads sample poetry corpus (Tiny Shakespeare / Sonnets) if not present."""
    if not os.path.exists(output_path):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        print(f"Downloading sample poetry dataset to {output_path}...")
        urllib.request.urlretrieve(DEFAULT_POETRY_URL, output_path)
        print("Download complete.")

class PoetryCharDataset(Dataset):
    def __init__(self, data_tensor: torch.Tensor, block_size: int):
        self.data = data_tensor
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.block_size]
        y = self.data[idx + 1 : idx + 1 + self.block_size]
        return x, y

def prepare_poetry_data(raw_txt_path: str, processed_dir: str, tokenizer: PoetryCharTokenizer, val_ratio: float = 0.1):
    """Reads raw poetry text, normalizes, tokenizes, splits train/val, and saves as PyTorch tensors."""
    os.makedirs(processed_dir, exist_ok=True)
    with open(raw_txt_path, 'r', encoding='utf-8') as f:
        raw_text = f.read()

    normalized_text = tokenizer.normalize_text(raw_text)
    encoded_tokens = tokenizer.encode(normalized_text)
    tensor_data = torch.tensor(encoded_tokens, dtype=torch.long)

    n_val = int(len(tensor_data) * val_ratio)
    train_data = tensor_data[:-n_val]
    val_data = tensor_data[-n_val:]

    train_path = os.path.join(processed_dir, 'train.pt')
    val_path = os.path.join(processed_dir, 'val.pt')

    torch.save(train_data, train_path)
    torch.save(val_data, val_path)

    print(f"Dataset summary:")
    print(f"  Raw character length: {len(raw_text)}")
    print(f"  Normalized tokens:    {len(tensor_data)}")
    print(f"  Train set tokens:     {len(train_data)}")
    print(f"  Val set tokens:       {len(val_data)}")
    return train_path, val_path
