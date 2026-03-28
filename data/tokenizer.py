import os
import pickle
from typing import List, Tuple
import torch


class SimpleTokenizer:
    """Simple character-level tokenizer for prototyping."""

    def __init__(self, vocab_size: int = 256):
        self.vocab_size = vocab_size
        self.char_to_id = {chr(i): i for i in range(vocab_size)}
        self.id_to_char = {i: chr(i) for i in range(vocab_size)}

    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs."""
        return [self.char_to_id.get(c, 0) for c in text if ord(c) < self.vocab_size]

    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs to text."""
        return ''.join(self.id_to_char.get(id, '?') for id in token_ids)

    def save(self, path: str):
        """Save tokenizer."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str):
        """Load tokenizer."""
        with open(path, 'rb') as f:
            return pickle.load(f)


class BPETokenizer:
    """
    Byte Pair Encoding tokenizer.
    Starts with character-level tokens and merges frequent pairs.
    """

    def __init__(self, vocab_size: int = 256):
        self.vocab_size = vocab_size
        self.char_to_id = {chr(i): i for i in range(256)}
        self.id_to_char = {i: chr(i) for i in range(256)}
        self.merges = []
        self.token_counter = 256

    def train(self, texts: List[str], num_merges: int = 1000):
        """Train BPE on texts."""
        # Tokenize all texts
        all_tokens = []
        for text in texts:
            tokens = [ord(c) for c in text]
            all_tokens.append(tuple(tokens))

        # Perform merges
        for _ in range(num_merges):
            # Count pairs
            pair_counts = {}
            for tokens in all_tokens:
                for i in range(len(tokens) - 1):
                    pair = (tokens[i], tokens[i + 1])
                    pair_counts[pair] = pair_counts.get(pair, 0) + 1

            if not pair_counts:
                break

            # Find most frequent pair
            best_pair = max(pair_counts, key=pair_counts.get)

            # Merge the pair
            self._merge_pair(best_pair)
            all_tokens = [self._merge_tokens(tokens, best_pair) for tokens in all_tokens]

            self.merges.append(best_pair)

    def _merge_pair(self, pair: Tuple[int, int]):
        """Add merged pair to vocabulary."""
        new_id = self.token_counter
        self.token_counter += 1
        self.char_to_id[pair] = new_id

    @staticmethod
    def _merge_tokens(tokens: tuple, pair: Tuple[int, int]) -> tuple:
        """Merge occurrences of pair in token sequence."""
        new_tokens = []
        i = 0
        while i < len(tokens):
            if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == pair:
                new_id = (pair[0], pair[1])
                new_tokens.append(new_id)
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1
        return tuple(new_tokens)

    def encode(self, text: str) -> List[int]:
        """Encode text using BPE."""
        tokens = [ord(c) for c in text]

        for pair in self.merges:
            new_id = self.char_to_id.get(pair)
            if new_id is None:
                continue

            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == pair:
                    new_tokens.append(new_id)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens

        return tokens

    def save(self, path: str):
        """Save tokenizer."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str):
        """Load tokenizer."""
        with open(path, 'rb') as f:
            return pickle.load(f)
