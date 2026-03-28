import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import time
from typing import Callable, Optional, Dict, Any
import json
import os


class LinearTransformerTrainer:
    """Trainer class for Linear Transformer models."""

    def __init__(self, model: nn.Module, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'val_perplexity': [],
            'epoch_times': []
        }

    def train(
        self,
        train_loader,
        val_loader,
        num_epochs: int = 10,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        warmup_epochs: int = 1,
        gradient_clip: float = 1.0,
        log_interval: int = 100,
        checkpoint_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Train the model.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of training epochs
            learning_rate: Initial learning rate
            weight_decay: Weight decay for optimizer
            warmup_epochs: Number of warmup epochs
            gradient_clip: Gradient clipping value
            log_interval: Logging interval in steps
            checkpoint_dir: Directory to save checkpoints

        Returns:
            Training history dictionary
        """
        if checkpoint_dir:
            os.makedirs(checkpoint_dir, exist_ok=True)

        # Optimizer and scheduler
        optimizer = AdamW(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, eta_min=1e-5)

        # Warmup scheduler
        def lr_lambda(current_step: int):
            if current_step < len(train_loader) * warmup_epochs:
                return float(current_step) / float(max(1, len(train_loader) * warmup_epochs))
            return 1.0

        from torch.optim.lr_scheduler import LambdaLR
        warmup_scheduler = LambdaLR(optimizer, lr_lambda)

        criterion = nn.CrossEntropyLoss()

        best_val_loss = float('inf')
        global_step = 0

        for epoch in range(num_epochs):
            epoch_start = time.time()

            # Training phase
            self.model.train()
            total_loss = 0
            num_batches = 0

            for batch_idx, (input_ids, target_ids) in enumerate(train_loader):
                input_ids = input_ids.to(self.device)
                target_ids = target_ids.to(self.device)

                # Forward pass
                logits = self.model(input_ids)
                loss = criterion(logits.view(-1, self.model.vocab_size), target_ids.view(-1))

                # Backward pass
                optimizer.zero_grad()
                loss.backward()

                # Gradient clipping
                if gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), gradient_clip)

                optimizer.step()

                # Warmup
                if epoch < warmup_epochs:
                    warmup_scheduler.step()
                else:
                    scheduler.step()

                total_loss += loss.item()
                num_batches += 1
                global_step += 1

                # Logging
                if (batch_idx + 1) % log_interval == 0:
                    avg_loss = total_loss / num_batches
                    print(f"Epoch {epoch + 1}/{num_epochs} | Batch {batch_idx + 1}/{len(train_loader)} | "
                          f"Loss: {avg_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")

            # Validation phase
            avg_train_loss = total_loss / num_batches
            val_loss, val_perplexity = self.validate(val_loader, criterion)

            epoch_time = time.time() - epoch_start
            self.training_history['train_loss'].append(avg_train_loss)
            self.training_history['val_loss'].append(val_loss)
            self.training_history['val_perplexity'].append(val_perplexity)
            self.training_history['epoch_times'].append(epoch_time)

            print(f"\nEpoch {epoch + 1}/{num_epochs} Summary:")
            print(f"  Train Loss: {avg_train_loss:.4f}")
            print(f"  Val Loss: {val_loss:.4f}")
            print(f"  Val Perplexity: {val_perplexity:.4f}")
            print(f"  Epoch Time: {epoch_time:.2f}s\n")

            # Checkpoint
            if checkpoint_dir and val_loss < best_val_loss:
                best_val_loss = val_loss
                checkpoint_path = os.path.join(checkpoint_dir, f"best_model.pt")
                self.save_checkpoint(checkpoint_path)
                print(f"Saved best model to {checkpoint_path}\n")

        return self.training_history

    def validate(self, val_loader, criterion) -> tuple:
        """Validate the model."""
        self.model.eval()
        total_loss = 0
        num_batches = 0

        with torch.no_grad():
            for input_ids, target_ids in val_loader:
                input_ids = input_ids.to(self.device)
                target_ids = target_ids.to(self.device)

                logits = self.model(input_ids)
                loss = criterion(logits.view(-1, self.model.vocab_size), target_ids.view(-1))

                total_loss += loss.item()
                num_batches += 1

        avg_loss = total_loss / num_batches
        perplexity = torch.exp(torch.tensor(avg_loss)).item()

        return avg_loss, perplexity

    def save_checkpoint(self, path: str):
        """Save model checkpoint."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'training_history': self.training_history
        }
        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.training_history = checkpoint['training_history']

    def save_history(self, path: str):
        """Save training history to JSON."""
        with open(path, 'w') as f:
            json.dump(self.training_history, f, indent=2)
