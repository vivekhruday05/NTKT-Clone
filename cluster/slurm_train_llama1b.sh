#!/bin/bash
#SBATCH --job-name=ntkt_llama1b
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/ntkt_llama1b_%j.out
#SBATCH --error=logs/ntkt_llama1b_%j.err

echo "=========================================================="
echo "Starting NTKT LLaMA-3.2-1B Training on $(hostname)"
echo "Job ID: $SLURM_JOB_ID | Start Time: $(date)"
echo "=========================================================="

source ~/.bashrc
conda activate myenv

export TOKENIZERS_PARALLELISM=false
mkdir -p logs checkpoints/ntkt_llama1b artifacts

python scripts/run_train_ntkt.py \
    --config configs/ntkt_llama1b.yaml \
    --output_dir checkpoints/ntkt_llama1b

echo "LLaMA-1B Training completed at $(date)!"
