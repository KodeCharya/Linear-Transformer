"""
Linear Transformer Training Script

Usage:
    python main.py --mode train --epochs 10 --batch_size 32
    python main.py --mode generate --prompt "Hello world" --max_length 100
"""

import os
import argparse
import torch
from dotenv import load_dotenv

from data.tokenizer import SimpleTokenizer
from data.dataset import TextDataset, create_data_loaders
from core.transformer import LinearTransformer
from training.trainer import LinearTransformerTrainer
from inference.generator import TextGenerator
from db.supabase_client import SupabaseClient

load_dotenv()


def create_sample_data():
    """Create sample training data."""
    sample_texts = [
        "The quick brown fox jumps over the lazy dog. " * 100,
        "Linear attention mechanisms enable efficient sequence processing. " * 100,
        "Transformers with linear complexity scale to long sequences. " * 100,
    ]
    return sample_texts


def train_mode(args):
    """Training mode."""
    print("Starting Linear Transformer Training...")

    # Initialize components
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Tokenizer
    tokenizer = SimpleTokenizer(vocab_size=256)

    # Sample data
    texts = create_sample_data()

    # Data loaders
    train_loader, val_loader = create_data_loaders(
        texts,
        tokenizer,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        train_split=0.8
    )

    print(f"Created data loaders with {len(train_loader)} train batches and {len(val_loader)} val batches")

    # Model
    model = LinearTransformer(
        vocab_size=256,
        dim=args.model_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        kernel_type=args.kernel_type,
        use_hybrid=args.use_hybrid,
        window_size=args.window_size
    )

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Created model with {num_params:,} parameters")

    # Save model config to database
    try:
        db = SupabaseClient()
        config = {
            'vocab_size': 256,
            'dim': args.model_dim,
            'num_layers': args.num_layers,
            'num_heads': args.num_heads,
            'kernel_type': args.kernel_type,
            'use_hybrid': args.use_hybrid,
            'window_size': args.window_size
        }
        config_id = db.save_model_config(config, "linear_transformer_v1")
        print(f"Saved model config to database with ID: {config_id}")

        training_config = {
            'learning_rate': args.learning_rate,
            'batch_size': args.batch_size,
            'epochs': args.epochs,
            'seq_len': args.seq_len
        }
        run_id = db.create_training_run("linear_transformer", config_id, training_config)
        print(f"Created training run with ID: {run_id}")
    except Exception as e:
        print(f"Warning: Could not connect to database: {e}")
        run_id = None

    # Trainer
    trainer = LinearTransformerTrainer(model, device)

    # Train
    checkpoint_dir = os.path.join('checkpoints', 'linear_transformer')
    history = trainer.train(
        train_loader,
        val_loader,
        num_epochs=args.epochs,
        learning_rate=args.learning_rate,
        checkpoint_dir=checkpoint_dir
    )

    # Save results
    trainer.save_history(os.path.join(checkpoint_dir, 'training_history.json'))
    print(f"\nTraining complete! Results saved to {checkpoint_dir}")

    # Log final metrics to database
    if run_id:
        try:
            final_val_loss = history['val_loss'][-1]
            final_perplexity = history['val_perplexity'][-1]
            total_time = sum(history['epoch_times'])

            db.update_training_run(run_id, 'completed', None)

            for epoch, (train_loss, val_loss, val_perp, epoch_time) in enumerate(zip(
                history['train_loss'],
                history['val_loss'],
                history['val_perplexity'],
                history['epoch_times']
            )):
                db.save_training_metrics(
                    run_id, epoch, train_loss, val_loss, val_perp,
                    epoch_time, args.learning_rate
                )

            print(f"Logged training results to database")
        except Exception as e:
            print(f"Warning: Could not log to database: {e}")


def generate_mode(args):
    """Generation mode."""
    print("Starting Text Generation...")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load tokenizer
    tokenizer = SimpleTokenizer(vocab_size=256)

    # Load model
    model = LinearTransformer(
        vocab_size=256,
        dim=args.model_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        kernel_type=args.kernel_type,
        use_hybrid=args.use_hybrid,
        window_size=args.window_size
    )

    # Load checkpoint if provided
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded checkpoint from {args.checkpoint}")

    # Generator
    generator = TextGenerator(model, tokenizer, device)

    # Generate
    print(f"\nPrompt: {args.prompt}")
    generated = generator.generate(
        args.prompt,
        max_length=args.max_length,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p
    )

    print(f"\nGenerated:\n{generated[:500]}...")


def main():
    parser = argparse.ArgumentParser(description='Linear Transformer Training and Generation')

    # Mode
    parser.add_argument('--mode', choices=['train', 'generate'], default='train',
                       help='Mode: train or generate')

    # Model parameters
    parser.add_argument('--model_dim', type=int, default=64,
                       help='Model dimension')
    parser.add_argument('--num_layers', type=int, default=3,
                       help='Number of transformer layers')
    parser.add_argument('--num_heads', type=int, default=4,
                       help='Number of attention heads')
    parser.add_argument('--kernel_type', choices=['relu', 'elu', 'identity'], default='elu',
                       help='Kernel type for feature mapping')

    # Attention parameters
    parser.add_argument('--use_hybrid', action='store_true',
                       help='Use hybrid attention with sliding window')
    parser.add_argument('--window_size', type=int, default=64,
                       help='Sliding window size for hybrid attention')

    # Training parameters
    parser.add_argument('--epochs', type=int, default=3,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size')
    parser.add_argument('--seq_len', type=int, default=128,
                       help='Sequence length')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                       help='Learning rate')

    # Generation parameters
    parser.add_argument('--prompt', type=str, default='The quick',
                       help='Prompt for generation')
    parser.add_argument('--max_length', type=int, default=100,
                       help='Maximum generation length')
    parser.add_argument('--temperature', type=float, default=1.0,
                       help='Sampling temperature')
    parser.add_argument('--top_k', type=int, default=None,
                       help='Top-k sampling')
    parser.add_argument('--top_p', type=float, default=None,
                       help='Top-p sampling')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to model checkpoint')

    args = parser.parse_args()

    if args.mode == 'train':
        train_mode(args)
    else:
        generate_mode(args)


if __name__ == '__main__':
    main()