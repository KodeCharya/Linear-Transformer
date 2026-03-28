import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Optional, Tuple
import os


class TextDataset(Dataset):
    """Simple text dataset for language modeling."""

    def __init__(self, texts: List[str], tokenizer, seq_len: int = 256, stride: int = 1):
        """
        Initialize text dataset.

        Args:
            texts: List of text documents
            tokenizer: Tokenizer object with encode method
            seq_len: Sequence length for context
            stride: Stride for sliding window
        """
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.stride = stride

        # Concatenate and tokenize all texts
        all_text = '\n'.join(texts)
        self.tokens = tokenizer.encode(all_text)

        # Create sequence pairs (context, target)
        self.sequences = []
        for i in range(0, len(self.tokens) - seq_len, stride):
            context = self.tokens[i:i + seq_len]
            target = self.tokens[i + seq_len]
            self.sequences.append((context, target))

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        context, target = self.sequences[idx]
        return (
            torch.tensor(context, dtype=torch.long),
            torch.tensor(target, dtype=torch.long)
        )


class FileDataset(Dataset):
    """Load text data from files."""

    def __init__(self, data_dir: str, tokenizer, seq_len: int = 256,
                 stride: int = 1, max_files: Optional[int] = None):
        """
        Initialize file-based dataset.

        Args:
            data_dir: Directory containing text files
            tokenizer: Tokenizer object
            seq_len: Sequence length
            stride: Sliding window stride
            max_files: Maximum number of files to load
        """
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.stride = stride

        # Load files
        texts = []
        for i, filename in enumerate(sorted(os.listdir(data_dir))):
            if max_files and i >= max_files:
                break
            filepath = os.path.join(data_dir, filename)
            if os.path.isfile(filepath):
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                    texts.append(f.read())

        # Tokenize
        all_text = '\n'.join(texts)
        self.tokens = tokenizer.encode(all_text)

        # Create sequences
        self.sequences = []
        for i in range(0, len(self.tokens) - seq_len, stride):
            context = self.tokens[i:i + seq_len]
            target = self.tokens[i + seq_len]
            self.sequences.append((context, target))

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        context, target = self.sequences[idx]
        return (
            torch.tensor(context, dtype=torch.long),
            torch.tensor(target, dtype=torch.long)
        )


def create_data_loaders(
    texts: List[str],
    tokenizer,
    seq_len: int = 256,
    batch_size: int = 32,
    num_workers: int = 0,
    train_split: float = 0.8
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation data loaders.

    Args:
        texts: List of text documents
        tokenizer: Tokenizer object
        seq_len: Sequence length
        batch_size: Batch size
        num_workers: Number of data loading workers
        train_split: Fraction of data for training

    Returns:
        train_loader, val_loader
    """
    dataset = TextDataset(texts, tokenizer, seq_len)

    # Split dataset
    train_size = int(len(dataset) * train_split)
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    # Create loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader


def collate_variable_length(batch):
    """Collate function for variable length sequences."""
    contexts, targets = zip(*batch)
    contexts = torch.stack(contexts)
    targets = torch.stack(targets)
    return contexts, targets
