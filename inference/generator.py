import torch
import torch.nn.functional as F
from typing import Optional, List
from core.transformer import LinearTransformer


class TextGenerator:
    """Generate text using trained Linear Transformer model."""

    def __init__(self, model: LinearTransformer, tokenizer, device: str = 'cuda'):
        self.model = model.to(device)
        self.model.eval()
        self.tokenizer = tokenizer
        self.device = device

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_length: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        repetition_penalty: float = 1.0
    ) -> str:
        """
        Generate text from prompt.

        Args:
            prompt: Starting text
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_k: Top-k filtering
            top_p: Top-p (nucleus) sampling
            repetition_penalty: Penalty for repeated tokens

        Returns:
            Generated text
        """
        # Encode prompt
        input_ids = torch.tensor([self.tokenizer.encode(prompt)], dtype=torch.long, device=self.device)

        generated = input_ids.clone()

        for _ in range(max_length):
            # Get next token logits
            logits = self.model(generated[:, -self.model.max_seq_len:])
            next_logits = logits[:, -1, :]

            # Apply repetition penalty
            if repetition_penalty != 1.0:
                for token_id in set(generated[0].tolist()):
                    next_logits[0, token_id] /= repetition_penalty

            # Apply temperature
            next_logits = next_logits / temperature

            # Apply top-k filtering
            if top_k is not None:
                top_k_values, _ = torch.topk(next_logits, top_k, dim=-1)
                next_logits[next_logits < top_k_values[:, -1:]] = float('-inf')

            # Apply top-p filtering
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                cumsum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumsum_probs > top_p
                sorted_indices_to_remove[..., 0] = False
                next_logits[sorted_indices[sorted_indices_to_remove]] = float('-inf')

            # Sample
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Append to sequence
            generated = torch.cat([generated, next_token], dim=-1)

        # Decode
        generated_text = self.tokenizer.decode(generated[0].tolist())
        return generated_text

    @torch.no_grad()
    def batch_generate(
        self,
        prompts: List[str],
        max_length: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None
    ) -> List[str]:
        """Generate text for batch of prompts."""
        max_prompt_len = max(len(self.tokenizer.encode(p)) for p in prompts)

        # Tokenize and pad prompts
        input_ids_list = [self.tokenizer.encode(p) for p in prompts]
        input_ids = torch.zeros(len(prompts), max_prompt_len, dtype=torch.long, device=self.device)

        for i, ids in enumerate(input_ids_list):
            input_ids[i, -len(ids):] = torch.tensor(ids, dtype=torch.long)

        batch_size = input_ids.shape[0]
        generated = input_ids.clone()

        for _ in range(max_length):
            logits = self.model(generated[:, -self.model.max_seq_len:])
            next_logits = logits[:, -1, :]

            # Batch sampling
            next_logits = next_logits / temperature

            if top_k is not None:
                indices_to_remove = next_logits < torch.topk(next_logits, min(top_k, next_logits.shape[-1]))[0][..., -1, None]
                next_logits[indices_to_remove] = float('-inf')

            probs = F.softmax(next_logits, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1)

            generated = torch.cat([generated, next_tokens], dim=-1)

        # Decode batch
        results = []
        for i in range(batch_size):
            text = self.tokenizer.decode(generated[i].tolist())
            results.append(text)

        return results
