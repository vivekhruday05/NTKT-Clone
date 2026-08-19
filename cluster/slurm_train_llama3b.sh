#!/bin/bash
#SBATCH --job-name=ntkt_llama3b
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=logs/ntkt_llama3b_%j.out
#SBATCH --error=logs/ntkt_llama3b_%j.err

echo "=========================================================="
echo "Starting NTKT LLaMA-3.2-3B Training on $(hostname)"
echo "Job ID: $SLURM_JOB_ID | Start Time: $(date)"
echo "GPU Device: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "=========================================================="

# Activate Conda Environment
source ~/.bashrc
conda activate myenv

# Set HuggingFace Cache and Token
export HF_HOME=${HF_HOME:-~/.cache/huggingface}
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Make directories
mkdir -p logs checkpoints/ntkt_llama3b artifacts

# Run NTKT LLaMA-3.2-3B Fine-Tuning
python scripts/run_train_ntkt.py \
    --config configs/default_ntkt_llama3b.yaml \
    --output_dir checkpoints/ntkt_llama3b

echo "Training completed successfully at $(date)!"
