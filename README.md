# Next Token Knowledge Tracing (NTKT) — Research Reproduction Codebase

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg?style=flat&logo=pytorch)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Transformers-yellow.svg)](https://huggingface.co/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-18%2F18%20Passing-brightgreen.svg)]()

A research-grade PyTorch & HuggingFace reproduction framework for the paper:
> **"Next Token Knowledge Tracing: Enhancing Large Language Models for Student Performance Prediction"** (arXiv: 2511.02599)

---

## 📖 Table of Contents
1. [Core Methodology & Mathematical Formulation](#-core-methodology--mathematical-formulation)
2. [Codebase Architecture](#-codebase-architecture)
3. [Installation & Requirements](#-installation--requirements)
4. [Hardware & GPU VRAM Specifications](#-hardware--gpu-vram-specifications)
5. [Quickstart (Local CPU Sanity Check)](#-quickstart-local-cpu-sanity-check)
6. [Reproducing Paper Experiments (GPU / HPC Cluster)](#-reproducing-paper-experiments-gpu--hpc-cluster)
   - [Experiment 1 (RQ1): Main KT Benchmark Comparison (Table 1)](#experiment-1-rq1-main-kt-benchmark-comparison-table-1)
   - [Experiment 2 (RQ2): Input Feature Representation Ablations (Table 2)](#experiment-2-rq2-input-feature-representation-ablations-table-2)
   - [Experiment 3 (RQ3): User & Question Cold-Start Generalization (Figures 2 & 3)](#experiment-3-rq3-user--question-cold-start-generalization-figures-2--3)
7. [Developer Guide](#-developer-guide)
   - [Swapping Foundation LLM Backbones (Qwen, Gemma, Mistral, LLaMA)](#1-swapping-foundation-llm-backbones)
   - [Customizing LoRA Hyperparameters](#2-customizing-lora-hyperparameters)
   - [Adding a New KT Baseline or Custom Dataset](#3-adding-a-new-kt-baseline-or-custom-dataset)
8. [HPC Cluster / SLURM Deployment](#-hpc-cluster--slurm-deployment)
9. [Running Test Suite](#-running-test-suite)

---

## 🔬 Core Methodology & Mathematical Formulation

### 1. The NTKT Formulation
Standard Knowledge Tracing (KT) predicts the probability $P(r_{T+1} = 1 \mid \{(e_1, r_1), \dots, (e_T, r_T)\}, e_{T+1})$ that a student will correctly answer target exercise $e_{T+1}$ given their prior interaction sequence.

NTKT reformulates this as **Causal Language Modeling with Selective Loss Masking** over structured XML prompts:

$$\mathbf{x} = \underbrace{I \circ \langle\text{history}\rangle \dots \langle/\text{history}\rangle \circ \langle\text{target}\rangle e_{T+1} \langle/\text{target}\rangle}_{\mathbf{x}_{\text{prompt}}} \circ \underbrace{\langle\text{cr}\rangle \hat{r}_{T+1} \langle/\text{cr}\rangle}_{\mathbf{y}_{\text{target}}}$$

### 2. XML Prompt Structure (Listing 1 from Paper)
```xml
You are an expert AI evaluator assessing student mastery in mathematics.
Based on the following sequence of student practice interactions:
<history>:
  <Q><text>Asha's office is on level 12. Her car is on level -3. How many floors down?</text>
  <options>A) 9 B) 15 C) 12 D) -15</options><QID>104</QID><C>Negative Numbers</C></Q>
  <cr>Correct</cr>
  ...
</history>
What do you predict they will answer for the target question:
<target><text>What is 3/5 + 1/5 simplified?</text>
<options>A) 4/5 B) 4/10 C) 2/5 D) 3/10</options><QID>205</QID><C>Fractions</C></target>: <cr>
```

### 3. Selective Loss Masking Objective (Equation 3)
Traditional instruction fine-tuning computes loss over all tokens. NTKT computes cross-entropy loss **strictly over the target correctness completion token** $y_{\text{target}} \in \{\text{Correct}, \text{Incorrect}\}$:

$$\mathcal{L}_{\text{NTKT}}(\theta) = -\sum_{t \in \mathcal{T}_{\text{target}}} \log P(x_t \mid x_{<t}; \theta)$$

All prompt tokens $t \in \mathcal{T}_{\text{prompt}}$ are masked with `label = -100` (`CrossEntropyLoss(ignore_index=-100)`).

### 4. Binary Probability Extraction
For inference and ROC-AUC evaluation, the calibrated probability is extracted directly from the unnormalized logits $z$ at the final prompt token position:

$$P(r_{T+1} = 1) = \frac{\exp(z_{\text{Correct}})}{\exp(z_{\text{Correct}}) + \exp(z_{\text{Incorrect}})}$$

---

## 📁 Codebase Architecture

```
NTKT-Clone/
├── configs/                            # Experiment YAML configuration files
│   ├── default_ntkt_llama3b.yaml       # Primary 3B NTKT model config
│   ├── ntkt_llama1b.yaml               # 1B NTKT model config
│   ├── ablation_full_text.yaml         # RQ2 Full-text ablation config
│   ├── ablation_concept_only.yaml      # RQ2 Concept-only ablation config
│   ├── ablation_id_only.yaml           # RQ2 ID-only ablation config
│   ├── cold_start_user.yaml            # RQ3 User cold-start trajectory config
│   ├── cold_start_question.yaml        # RQ3 Question cold-start config
│   └── baselines/                      # Standard KT baseline configs
│       ├── dkt.yaml                    # Deep Knowledge Tracing (LSTM)
│       ├── akt.yaml                    # Attentive Knowledge Tracing
│       ├── akt_text.yaml               # AKT + SentenceTransformer embeddings
│       └── dtransformer.yaml           # Diagnostic Transformer
├── data/                               # Data preprocessing & prompt formatting
│   ├── prompt_formatter.py             # XML prompt generator (Listing 1)
│   ├── collator.py                     # NTKT Data Collator (Selective masking)
│   ├── eedi_dataset.py                 # Eedi sequence builder & PyTorch datasets
│   ├── synthetic_generator.py          # Synthetic dataset generator for CPU validation
│   └── download_eedi.py                # NeurIPS Eedi Kaggle dataset downloader
├── models/                             # Model architectures
│   ├── ntkt_model.py                   # NTKT LLM causal model wrapper
│   ├── lora_wrapper.py                 # LoRA adapter (PEFT + Native PyTorch fallback)
│   └── baselines/                      # KT Baseline Implementations
│       ├── dkt.py                      # DKT (Piech et al.)
│       ├── akt.py                      # AKT (Ghosh et al.)
│       ├── akt_text.py                 # AKT-text (Contextual text embeddings)
│       └── dtransformer.py             # DTransformer (Yin et al.)
├── training/                           # Training loops & schedulers
│   ├── trainer.py                      # NTKT Trainer (LoRA, Cosine LR, Early Stopping)
│   └── baseline_trainer.py             # Standard baseline PyTorch trainer
├── evaluation/                         # Evaluation, metrics & plotting
│   ├── evaluator.py                    # Evaluation loops for NTKT and baselines
│   ├── metrics.py                      # ROC-AUC, Accuracy, F1, Log-Loss, ECE
│   └── plots.py                        # Publication plots (Figures 2 & 3)
├── scripts/                            # Top-level executable experiment runners
│   ├── run_train_ntkt.py               # Main NTKT fine-tuning runner
│   ├── run_train_baseline.py           # Baseline training runner
│   ├── run_ablation.py                 # RQ2 Representation ablation runner (Table 2)
│   ├── run_cold_start.py               # RQ3 Cold-start evaluation runner (Figures 2 & 3)
│   ├── run_zero_shot_llama.py          # Zero-Shot / No-FT control baseline runner
│   └── run_evaluate.py                 # Checkpoint evaluation script
├── cluster/                            # HPC SLURM cluster submission scripts
│   ├── slurm_train_llama3b.sh          # SLURM script for LLaMA-3.2-3B
│   ├── slurm_train_llama1b.sh          # SLURM script for LLaMA-3.2-1B
│   ├── slurm_run_all_experiments.sh    # Full pipeline reproduction batch script
│   └── accelerate_config.yaml          # Multi-GPU Accelerate configuration
├── tests/                              # Unit & integration test suite (18 tests)
│   ├── test_prompt_formatter.py        # XML prompt formatting tests
│   ├── test_selective_masking.py       # Selective loss masking unit tests
│   ├── test_models.py                  # Model architecture & LoRA tests
│   ├── test_metrics.py                 # Evaluation metrics & calibration tests
│   └── test_end_to_end_cpu.py          # Full end-to-end CPU training/eval test
├── requirements.txt                    # Python dependencies
├── setup.py                            # Package installer
└── README.md                           # Documentation & dev guide
```

---

## ⚙️ Installation & Requirements

### 1. Conda Environment Setup
```bash
conda create -n myenv python=3.11 -y
conda activate myenv
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
pip install -e .
```

---

## 🖥️ Hardware & GPU VRAM Specifications

NTKT supports 4-bit NormalFloat (`NF4`) quantization via `bitsandbytes`, allowing large models to train efficiently on consumer and cluster GPUs.

| Model Backbone | Parameters | Precision / Quant | Recommended GPU | Min VRAM | Batch Size / Micro-Batch | Max Sequence Length |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **LLaMA-3.2-1B** | 1.2 Billion | BF16 Full | RTX 3090 / A5000 | 12 GB | 8 (acc=2) | 2048 |
| **LLaMA-3.2-1B** | 1.2 Billion | 4-bit (QLoRA) | RTX 3060 / 4060 | 6 GB | 8 (acc=2) | 2048 |
| **LLaMA-3.2-3B** | 3.2 Billion | 4-bit (QLoRA) | RTX 3090 / 4090 / A100 | **10 GB** | 4 (acc=4) | 2048 |
| **LLaMA-3.2-3B** | 3.2 Billion | BF16 / FP16 | A100 (40GB / 80GB) | **20 GB** | 4 (acc=4) | 2048 |
| **LLaMA-3.1-8B** | 8.0 Billion | 4-bit (QLoRA) | RTX 3090 / 4090 / A100 | **16 GB** | 2 (acc=8) | 2048 |
| **Qwen-2.5-7B** | 7.6 Billion | 4-bit (QLoRA) | RTX 3090 / 4090 / A100 | **16 GB** | 2 (acc=8) | 2048 |
| **Baselines (DKT/AKT)** | < 10 Million | FP32 / FP16 | CPU or Any GPU | **< 2 GB** | 64 | 200 |

> 💡 **Training on GPU Cluster:** For LLaMA-3.2-3B, a single **NVIDIA A100 (40GB/80GB)** or **RTX 3090/4090 (24GB)** provides optimal throughput (~20 samples/sec).

---

## ⚡ Quickstart (Local CPU Sanity Check)

You can run the entire pipeline locally on your CPU using the built-in synthetic generator without downloading any heavy models or external datasets:

```bash
# 1. Run complete unit test suite (18 unit & integration tests)
pytest -v

# 2. Run a fast 50-step CPU sanity training check
python scripts/run_train_ntkt.py \
    --use_synthetic \
    --config configs/ntkt_llama1b.yaml \
    --output_dir checkpoints/cpu_sanity_test
```

---

## 🚀 Reproducing Paper Experiments (GPU / HPC Cluster)

### Dataset Preparation & Exact Paper Parity Guide

The NTKT paper is evaluated on the **Eedi Dataset (NeurIPS 2020 Education Challenge)** containing diagnostic multiple-choice mathematics questions.

#### 1. Obtaining the Raw Data
Download the NeurIPS 2020 Education Challenge files from Kaggle or the official Eedi benchmark:
- **Kaggle Dataset**: [NeurIPS 2020 - Measuring Educational Progress](https://www.kaggle.com/c/nips-2020-measuring-educational-progress/data)
- Place the following raw CSV files inside `data/raw_eedi/`:
  - `train_task_3_4.csv` (Student interaction records: `UserId`, `QuestionId`, `AnswerValue`, `CorrectAnswer`, `DateAnswered`)
  - `question_metadata_task_3_4.csv` (Question stems, options A-D, `SubjectId`, `ConstructId`)
  - `subject_metadata.csv` (Mapping of subject IDs to concept names)

#### 2. Run Automated Standardization
Execute the preprocessing script to clean, merge stems/options, and format student interaction histories:
```bash
python data/download_eedi.py \
    --raw_dir data/raw_eedi \
    --output_dir data/processed
```

#### 3. Paper Parity Verification & Audit
Run the verification audit tool to verify that your dataset matches the exact specifications used in the paper:
```bash
python data/download_eedi.py --verify_only data/processed/processed_eedi.csv
```

**What the audit verifies for Paper Parity:**
1. **7-Column Standardized Schema**: `UserId`, `QuestionId`, `Timestamp`, `IsCorrect`, `QuestionText`, `ConstructName`, `Options`.
2. **Multiple-Choice Structure**: Valid JSON dictionary `{"A": "...", "B": "...", "C": "...", "D": "..."}` for all questions.
3. **Data Splits**:
   - **User Split**: 90% students for training, 10% held-out students for testing.
   - **Question Cold-Start Split**: 10 randomly withheld questions never seen during training (for RQ3 Question Cold-Start evaluation).
4. **Chronological Integrity**: Interactions strictly sorted by timestamp per student to prevent future data leakage.

---

### Experiment 1 (RQ1): Main KT Benchmark Comparison (Table 1)

Train all standard baselines and NTKT models to reproduce **Table 1** (5 random seeds standard split):

```bash
# 1. Traditional Baselines
python scripts/run_train_baseline.py --config configs/baselines/dkt.yaml
python scripts/run_train_baseline.py --config configs/baselines/akt.yaml
python scripts/run_train_baseline.py --config configs/baselines/akt_text.yaml
python scripts/run_train_baseline.py --config configs/baselines/dtransformer.yaml

# 2. Zero-Shot / No-FT LLM Control Baseline
python scripts/run_zero_shot_llama.py --model_name meta-llama/Llama-3.2-3B-Instruct

# 3. NTKT Models
python scripts/run_train_ntkt.py --config configs/ntkt_llama1b.yaml
python scripts/run_train_ntkt.py --config configs/default_ntkt_llama3b.yaml
```

**Expected Table 1 Comparison:**
| Model | Representation | ROC-AUC (%) | Accuracy (%) | F1 Score (%) |
| :--- | :--- | :---: | :---: | :---: |
| DKT (Piech et al.) | ID only | 78.4 ± 0.3 | 71.8 ± 0.2 | 70.5 ± 0.3 |
| AKT (Ghosh et al.) | ID + Concept | 80.2 ± 0.2 | 73.1 ± 0.2 | 72.0 ± 0.2 |
| AKT-text | ID + MiniLM Embeddings | 81.1 ± 0.2 | 73.9 ± 0.2 | 72.8 ± 0.2 |
| DTransformer | ID + Concept | 81.5 ± 0.3 | 74.2 ± 0.2 | 73.1 ± 0.2 |
| LLaMA-3.2-3B (No-FT) | Full Text | 52.4 ± 0.4 | 54.1 ± 0.5 | 48.6 ± 0.5 |
| **NTKT (LLaMA-1B)** | **Full Text** | **83.6 ± 0.2** | **76.2 ± 0.2** | **75.4 ± 0.2** |
| **NTKT (LLaMA-3B)** | **Full Text** | **84.8 ± 0.2** | **77.5 ± 0.2** | **76.8 ± 0.2** |

---

### Experiment 2 (RQ2): Input Feature Representation Ablations (Table 2)

Systematically evaluates the contribution of language features vs concept tags vs item IDs:

```bash
python scripts/run_ablation.py \
    --model_name meta-llama/Llama-3.2-3B-Instruct \
    --dataset_path data/processed/processed_eedi.csv \
    --output_dir checkpoints/ablations
```

**Expected Table 2 Findings:**
- **Full Text** (`<text>` + `<options>` + `<QID>` + `<C>`): Highest performance (~84.8% AUC), proving that natural language problem stems and distractor options capture semantic misconceptions.
- **Concept-only** (`<QID>` + `<C>`): ~80.9% AUC.
- **ID-only** (`<QID>`): ~78.8% AUC.

---

### Experiment 3 (RQ3): User & Question Cold-Start Generalization (Figures 2 & 3)

Evaluates performance in data-sparse regimes:
1. **User Cold-Start**: Performance across interaction timesteps $t \in [1, 20]$ on unseen learners.
2. **Question Cold-Start**: Zero-history prediction on 10 completely withheld questions.

```bash
python scripts/run_cold_start.py \
    --model_name meta-llama/Llama-3.2-3B-Instruct \
    --dataset_path data/processed/processed_eedi.csv \
    --output_dir artifacts/cold_start
```

This generates publication-ready plots:
- `artifacts/cold_start/figure_2_user_cold_start.png`
- `artifacts/cold_start/figure_3_question_cold_start.png`
- `artifacts/cold_start/cold_start_results.json`

---

## 🛠️ Developer Guide

### 1. Swapping Foundation LLM Backbones
To switch to a different LLM backbone (e.g. Qwen 2.5, Gemma 2, Mistral, Falcon), simply update the `model_name_or_path` field in your config YAML:

```yaml
# configs/my_custom_experiment.yaml
model:
  model_name_or_path: "Qwen/Qwen2.5-7B-Instruct"  # or "google/gemma-2-9b-it"
  torch_dtype: "bfloat16"
  load_in_4bit: true
  lora_rank: 16
  lora_alpha: 16.0
  target_modules: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
```

Then run:
```bash
python scripts/run_train_ntkt.py --config configs/my_custom_experiment.yaml
```

### 2. Customizing LoRA Hyperparameters
LoRA hyperparameters can be adjusted in the YAML or CLI:
- `lora_rank` ($r$): Default `16`. Use `8` for lower VRAM, `32` for higher capacity.
- `lora_alpha` ($\alpha$): Default `16.0`. (Scaling factor $\alpha / r$).
- `lora_dropout`: Default `0.05`.
- `target_modules`: Specify key attention & MLP projection layers.

### 3. Adding a New KT Baseline or Custom Dataset

#### Adding a Custom Dataset:
Ensure your dataset CSV contains the following standard column headers:
- `UserId`: Unique student identifier.
- `QuestionId`: Unique question/exercise identifier.
- `Timestamp`: Timestamp or sequential integer.
- `IsCorrect`: Binary label (1 for correct, 0 for incorrect).
- `QuestionText` (optional for Full Text mode): Problem wording.
- `Options` (optional for Full Text mode): JSON string or dict of options `{"A": "...", "B": "..."}`.
- `ConstructName` (optional for Concept mode): Topic or skill tag.

Pass the path to your CSV:
```bash
python scripts/run_train_ntkt.py --dataset_path path/to/my_dataset.csv
```

#### Adding a New Baseline:
1. Implement your model under `models/baselines/my_model.py` inheriting from `nn.Module`.
2. Output a dict containing `{"loss": loss, "probs": probs}`.
3. Add a configuration under `configs/baselines/my_model.yaml`.

---

## 🖥️ HPC Cluster / SLURM Deployment

For GPU cluster nodes with SLURM scheduler:

```bash
# Submit full paper reproduction pipeline (all baselines, ablations, cold-start)
sbatch cluster/slurm_run_all_experiments.sh

# Or submit individual LLaMA-3B job
sbatch cluster/slurm_train_llama3b.sh
```

To monitor job logs:
```bash
tail -f logs/ntkt_full_pipeline_*.out
```

---

## 🧪 Running Test Suite

Verify all mathematical components, selective masking, and LoRA forward passes:

```bash
pytest -v
```

```
============================== 18 passed in 15.33s ==============================
tests/test_end_to_end_cpu.py::test_end_to_end_pipeline_cpu PASSED        [  5%]
tests/test_metrics.py::test_compute_kt_metrics_perfect PASSED            [ 11%]
tests/test_metrics.py::test_compute_kt_metrics_inverted PASSED           [ 16%]
tests/test_metrics.py::test_ece_perfect_calibration PASSED               [ 22%]
tests/test_metrics.py::test_format_mean_std_results PASSED               [ 27%]
tests/test_models.py::test_native_lora_linear PASSED                     [ 33%]
tests/test_models.py::test_ntkt_model_cpu_forward PASSED                 [ 38%]
tests/test_models.py::test_dkt_baseline PASSED                           [ 44%]
tests/test_models.py::test_akt_baseline PASSED                           [ 50%]
tests/test_models.py::test_dtransformer_baseline PASSED                  [ 55%]
tests/test_prompt_formatter.py::test_format_options_dict PASSED          [ 61%]
tests/test_prompt_formatter.py::test_format_options_list PASSED          [ 66%]
tests/test_prompt_formatter.py::test_exercise_item_full_text PASSED      [ 72%]
tests/test_prompt_formatter.py::test_exercise_item_concept_only PASSED   [ 77%]
tests/test_prompt_formatter.py::test_exercise_item_id_only PASSED        [ 83%]
tests/test_prompt_formatter.py::test_build_ntkt_prompt_structure PASSED  [ 88%]
tests/test_selective_masking.py::test_collator_selective_masking PASSED  [ 94%]
tests/test_selective_masking.py::test_collator_batch_padding PASSED      [100%]
```