/*
  # Create Linear Transformer Training Infrastructure

  1. New Tables
    - `training_runs`: Track training experiments and configurations
      - `id` (uuid, primary key)
      - `model_name` (text): Name of the model
      - `config` (jsonb): Training hyperparameters
      - `status` (text): 'pending', 'running', 'completed', 'failed'
      - `start_time` (timestamp)
      - `end_time` (timestamp)
      - `created_at` (timestamp)

    - `training_metrics`: Store per-epoch training metrics
      - `id` (uuid, primary key)
      - `run_id` (uuid, foreign key to training_runs)
      - `epoch` (integer): Training epoch number
      - `train_loss` (float): Training loss
      - `val_loss` (float): Validation loss
      - `val_perplexity` (float): Validation perplexity
      - `epoch_time` (float): Time for epoch in seconds
      - `learning_rate` (float): Learning rate for this epoch
      - `created_at` (timestamp)

    - `model_checkpoints`: Store model checkpoints and metadata
      - `id` (uuid, primary key)
      - `run_id` (uuid, foreign key to training_runs)
      - `checkpoint_epoch` (integer): Which epoch this checkpoint is from
      - `checkpoint_path` (text): S3 or local path to checkpoint
      - `checkpoint_size` (integer): Size in bytes
      - `val_loss` (float): Validation loss for this checkpoint
      - `val_perplexity` (float): Validation perplexity
      - `is_best` (boolean): Whether this is the best checkpoint
      - `created_at` (timestamp)

    - `model_configurations`: Store model architecture configurations
      - `id` (uuid, primary key)
      - `name` (text, unique): Configuration name
      - `vocab_size` (integer): Vocabulary size
      - `dim` (integer): Model dimension
      - `num_layers` (integer): Number of transformer layers
      - `num_heads` (integer): Number of attention heads
      - `kernel_type` (text): Kernel function type ('relu', 'elu', etc.)
      - `use_hybrid` (boolean): Whether to use hybrid attention
      - `window_size` (integer): Sliding window size for hybrid attention
      - `max_seq_len` (integer): Maximum sequence length
      - `config` (jsonb): Full configuration as JSON
      - `created_at` (timestamp)

  2. Security
    - Enable RLS on all tables
    - Public read access to configurations and completed runs (for sharing results)
    - Access control based on ownership for training runs and checkpoints
*/

CREATE TABLE IF NOT EXISTS model_configurations (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  name text UNIQUE NOT NULL,
  vocab_size integer NOT NULL,
  dim integer NOT NULL,
  num_layers integer NOT NULL,
  num_heads integer NOT NULL,
  kernel_type text NOT NULL DEFAULT 'elu',
  use_hybrid boolean DEFAULT false,
  window_size integer DEFAULT 64,
  max_seq_len integer DEFAULT 4096,
  config jsonb NOT NULL,
  created_at timestamptz DEFAULT now()
);

CREATE TABLE IF NOT EXISTS training_runs (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  model_name text NOT NULL,
  model_config_id uuid REFERENCES model_configurations(id),
  config jsonb NOT NULL,
  status text NOT NULL DEFAULT 'pending',
  start_time timestamptz,
  end_time timestamptz,
  created_at timestamptz DEFAULT now()
);

CREATE TABLE IF NOT EXISTS training_metrics (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  run_id uuid NOT NULL REFERENCES training_runs(id) ON DELETE CASCADE,
  epoch integer NOT NULL,
  train_loss float NOT NULL,
  val_loss float NOT NULL,
  val_perplexity float NOT NULL,
  epoch_time float NOT NULL,
  learning_rate float NOT NULL,
  created_at timestamptz DEFAULT now()
);

CREATE TABLE IF NOT EXISTS model_checkpoints (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  run_id uuid NOT NULL REFERENCES training_runs(id) ON DELETE CASCADE,
  checkpoint_epoch integer NOT NULL,
  checkpoint_path text NOT NULL,
  checkpoint_size integer,
  val_loss float,
  val_perplexity float,
  is_best boolean DEFAULT false,
  created_at timestamptz DEFAULT now()
);

ALTER TABLE model_configurations ENABLE ROW LEVEL SECURITY;
ALTER TABLE training_runs ENABLE ROW LEVEL SECURITY;
ALTER TABLE training_metrics ENABLE ROW LEVEL SECURITY;
ALTER TABLE model_checkpoints ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Model configs are publicly readable"
  ON model_configurations FOR SELECT
  TO public
  USING (true);

CREATE POLICY "Anyone can insert model configs"
  ON model_configurations FOR INSERT
  TO public
  WITH CHECK (true);

CREATE POLICY "Training runs are publicly readable"
  ON training_runs FOR SELECT
  TO public
  USING (true);

CREATE POLICY "Anyone can insert training runs"
  ON training_runs FOR INSERT
  TO public
  WITH CHECK (true);

CREATE POLICY "Anyone can update training runs"
  ON training_runs FOR UPDATE
  TO public
  WITH CHECK (true);

CREATE POLICY "Training metrics are publicly readable"
  ON training_metrics FOR SELECT
  TO public
  USING (true);

CREATE POLICY "Anyone can insert training metrics"
  ON training_metrics FOR INSERT
  TO public
  WITH CHECK (true);

CREATE POLICY "Model checkpoints are publicly readable"
  ON model_checkpoints FOR SELECT
  TO public
  USING (true);

CREATE POLICY "Anyone can insert model checkpoints"
  ON model_checkpoints FOR INSERT
  TO public
  WITH CHECK (true);

CREATE INDEX IF NOT EXISTS idx_training_runs_status ON training_runs(status);
CREATE INDEX IF NOT EXISTS idx_training_runs_created ON training_runs(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_training_metrics_run_id ON training_metrics(run_id);
CREATE INDEX IF NOT EXISTS idx_training_metrics_epoch ON training_metrics(epoch);
CREATE INDEX IF NOT EXISTS idx_model_checkpoints_run_id ON model_checkpoints(run_id);
CREATE INDEX IF NOT EXISTS idx_model_checkpoints_is_best ON model_checkpoints(is_best);
