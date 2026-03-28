import os
from supabase import create_client
import json
from typing import Optional, Dict, List, Any
from datetime import datetime
import uuid


class SupabaseClient:
    """Wrapper for Supabase database operations."""

    def __init__(self):
        supabase_url = os.getenv('VITE_SUPABASE_URL')
        supabase_key = os.getenv('VITE_SUPABASE_SUPABASE_ANON_KEY')

        if not supabase_url or not supabase_key:
            raise ValueError("Supabase credentials not found in environment variables")

        self.client = create_client(supabase_url, supabase_key)

    def save_model_config(self, config: Dict[str, Any], name: str) -> str:
        """Save model configuration to database."""
        data = {
            'name': name,
            'vocab_size': config['vocab_size'],
            'dim': config['dim'],
            'num_layers': config['num_layers'],
            'num_heads': config['num_heads'],
            'kernel_type': config.get('kernel_type', 'elu'),
            'use_hybrid': config.get('use_hybrid', False),
            'window_size': config.get('window_size', 64),
            'max_seq_len': config.get('max_seq_len', 4096),
            'config': config
        }

        response = self.client.table('model_configurations').insert(data).execute()
        return response.data[0]['id']

    def get_model_config(self, config_id: str) -> Optional[Dict]:
        """Retrieve model configuration."""
        response = self.client.table('model_configurations').select('*').eq('id', config_id).execute()
        return response.data[0] if response.data else None

    def create_training_run(self, model_name: str, model_config_id: str,
                           training_config: Dict) -> str:
        """Create a new training run record."""
        data = {
            'model_name': model_name,
            'model_config_id': model_config_id,
            'config': training_config,
            'status': 'running',
            'start_time': datetime.utcnow().isoformat()
        }

        response = self.client.table('training_runs').insert(data).execute()
        return response.data[0]['id']

    def update_training_run(self, run_id: str, status: str, end_time: Optional[str] = None):
        """Update training run status."""
        data = {'status': status}
        if end_time:
            data['end_time'] = end_time

        self.client.table('training_runs').update(data).eq('id', run_id).execute()

    def save_training_metrics(self, run_id: str, epoch: int, train_loss: float,
                             val_loss: float, val_perplexity: float,
                             epoch_time: float, learning_rate: float):
        """Save training metrics for an epoch."""
        data = {
            'run_id': run_id,
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_perplexity': val_perplexity,
            'epoch_time': epoch_time,
            'learning_rate': learning_rate
        }

        self.client.table('training_metrics').insert(data).execute()

    def save_checkpoint(self, run_id: str, epoch: int, checkpoint_path: str,
                       val_loss: float, val_perplexity: float,
                       checkpoint_size: Optional[int] = None, is_best: bool = False):
        """Save checkpoint metadata."""
        data = {
            'run_id': run_id,
            'checkpoint_epoch': epoch,
            'checkpoint_path': checkpoint_path,
            'checkpoint_size': checkpoint_size,
            'val_loss': val_loss,
            'val_perplexity': val_perplexity,
            'is_best': is_best
        }

        self.client.table('model_checkpoints').insert(data).execute()

    def get_training_run(self, run_id: str) -> Optional[Dict]:
        """Get training run details."""
        response = self.client.table('training_runs').select('*').eq('id', run_id).execute()
        return response.data[0] if response.data else None

    def get_training_metrics(self, run_id: str) -> List[Dict]:
        """Get all training metrics for a run."""
        response = self.client.table('training_metrics').select('*').eq('run_id', run_id).order('epoch').execute()
        return response.data

    def get_best_checkpoint(self, run_id: str) -> Optional[Dict]:
        """Get the best checkpoint for a run."""
        response = self.client.table('model_checkpoints').select('*').eq('run_id', run_id).eq('is_best', True).execute()
        return response.data[0] if response.data else None

    def list_training_runs(self, limit: int = 10) -> List[Dict]:
        """List recent training runs."""
        response = self.client.table('training_runs').select('*').order('created_at', desc=True).limit(limit).execute()
        return response.data

    def get_run_statistics(self, run_id: str) -> Dict:
        """Get statistics for a training run."""
        metrics = self.get_training_metrics(run_id)

        if not metrics:
            return {}

        return {
            'num_epochs': len(metrics),
            'best_train_loss': min(m['train_loss'] for m in metrics),
            'best_val_loss': min(m['val_loss'] for m in metrics),
            'best_val_perplexity': min(m['val_perplexity'] for m in metrics),
            'total_time': sum(m['epoch_time'] for m in metrics),
            'final_val_loss': metrics[-1]['val_loss'],
            'final_val_perplexity': metrics[-1]['val_perplexity']
        }
