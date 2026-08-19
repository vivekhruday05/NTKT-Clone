#!/bin/bash
#SBATCH --job-name=ntkt_reproduce_all
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=logs/ntkt_full_pipeline_%j.out
#SBATCH --error=logs/ntkt_full_pipeline_%j.err

echo "=========================================================="
echo "REPRODUCING FULL NTKT PAPER RESULTS"
echo "Job ID: $SLURM_JOB_ID | Start Time: $(date)"
echo "=========================================================="

source ~/.bashrc
conda activate myenv
mkdir -p logs artifacts checkpoints

# ------------------------------------------------------------
# STEP 1: Process Raw Eedi Dataset (if not already processed)
# ------------------------------------------------------------
if [ ! -f "data/processed/processed_eedi.csv" ]; then
    echo "Processing raw Eedi dataset..."
    python data/download_eedi.py --raw_dir data/raw_eedi --output_dir data/processed
fi

# ------------------------------------------------------------
# STEP 2: RQ1 - Train Baselines (DKT, AKT, AKT-text, DTransformer)
# ------------------------------------------------------------
echo "Training DKT baseline..."
python scripts/run_train_baseline.py --config configs/baselines/dkt.yaml

echo "Training AKT baseline..."
python scripts/run_train_baseline.py --config configs/baselines/akt.yaml

echo "Training AKT-text baseline..."
python scripts/run_train_baseline.py --config configs/baselines/akt_text.yaml

echo "Training DTransformer baseline..."
python scripts/run_train_baseline.py --config configs/baselines/dtransformer.yaml

echo "Running Zero-Shot No-FT Control..."
python scripts/run_zero_shot_llama.py --model_name meta-llama/Llama-3.2-3B-Instruct

# ------------------------------------------------------------
# STEP 3: RQ1 - Train NTKT Main Models (LLaMA 1B and LLaMA 3B)
# ------------------------------------------------------------
echo "Training NTKT LLaMA-1B..."
python scripts/run_train_ntkt.py --config configs/ntkt_llama1b.yaml

echo "Training NTKT LLaMA-3B..."
python scripts/run_train_ntkt.py --config configs/default_ntkt_llama3b.yaml

# ------------------------------------------------------------
# STEP 4: RQ2 - Feature Representation Ablations (Table 2)
# ------------------------------------------------------------
echo "Running Feature Representation Ablations..."
python scripts/run_ablation.py \
    --model_name meta-llama/Llama-3.2-3B-Instruct \
    --output_dir checkpoints/ablations

# ------------------------------------------------------------
# STEP 5: RQ3 - Cold-Start Generalization (Figures 2 & 3)
# ------------------------------------------------------------
echo "Running Cold-Start Generalization Experiments..."
python scripts/run_cold_start.py \
    --model_name meta-llama/Llama-3.2-3B-Instruct \
    --output_dir artifacts/cold_start

echo "=========================================================="
echo "ALL EXPERIMENTS COMPLETED AT: $(date)"
echo "Artifacts and figures saved to artifacts/"
echo "=========================================================="
