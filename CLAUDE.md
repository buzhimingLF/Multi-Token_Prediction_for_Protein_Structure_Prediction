# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**ProteinMTP** applies Multi-Token Prediction (MTP) to protein structure prediction. The core innovation is using MTP to transform LLM output into a fixed-length format where k=sequence_length, enabling direct coordinate regression rather than string generation.

**Why MTP instead of direct coordinate string generation?**
- Direct string generation ("1.23,4.56,7.89") suffers from non-smooth gradients, low task ceiling (indirect mapping), and can only use cross-entropy loss
- MTP enables: direct (x,y,z) output, MSE loss for coordinate optimization, smooth gradients, and fixed-length output

**Key Insight**: MTP uses placeholder tokens to constrain output length. In the reference classification task, `num_of_pl_tokens = num_classes`. In this protein structure prediction task, `num_of_pl_tokens = sequence_length`.

## Commands

### Data Preparation
```bash
# Extract coordinates from PDBbind-Plus dataset (requires P-L/ directory)
python data_preprocessing.py --extract_coords --max_samples 1000

# Create training dataset with normalization and train/val split
python create_coord_data.py --max_seq_len 512 --train_ratio 0.9
```

### Training
```bash
# Train coordinate regression model (recommended)
python train_protein_structure.py \
    --train_data coord_train.json \
    --val_data coord_val.json \
    --model_name Qwen/Qwen2.5-0.5B \
    --max_seq_len 512 \
    --num_epochs 3 \
    --learning_rate 1e-4 \
    --output_dir ./output_structure

# Note: train_protein_mtp.py is deprecated (standard MTP version)
```

### Inference
```bash
# Predict structure for a single sequence (automatic denormalization)
python infer_protein_structure.py \
    --model_path ./output_structure \
    --sequence "MKTAYIAKQRQISFVKTIGDEVQREAPGDSRLAGHFELSC" \
    --output predicted_structure.pdb

# Note: Coordinates are automatically denormalized using global stats from training_config.json
# Output coordinates will be in Ångström units (typical range: 10-50 Å)
```

### Evaluation & Visualization
```bash
# Evaluate predicted structure (requires ground truth)
python evaluate_structure.py \
    --pred predicted_structure.pdb \
    --true true_structure.pdb \
    --all_metrics

# Visualize structure (3D/2D projections)
python visualize_structure.py \
    --pdb predicted_structure.pdb \
    --output structure_viz.png \
    --mode both \
    --stats
```

## Architecture

### MTP Flow
```
Input: Protein sequence "MKTAYIAK..."
  ↓ Tokenize
  [token1, token2, ..., tokenN]
  ↓ Add k placeholder tokens (k=sequence_length)
  [token1, ..., tokenN, <unk>, <unk>, ..., <unk>]
  ↓ Through LLM
  hidden_states: (B, N+k, hidden_size)
  ↓ Split: lm_part + pl_part
  ↓ LabelWiseAttention (pool lm_part, concat with pl_part)
  ↓ coord_head projection
  Output: (B, k, 3) coordinates
```

### Key Components (train_protein_structure.py)

**LabelWiseAttention**: Maps placeholder token hidden states to output positions
- Pools input features: `lm_part.max(dim=1)`
- Concatenates with placeholder features: `cat([lm_part_pooled, pl_part])`
- Projects through linear layers

**ProteinStructureMTP**: Main model class
- `coord_proj`: LabelWiseAttention module
- `coord_head`: Linear projection to 3D coordinates (hidden_size → 3)
- Loss: MSE between predicted and true normalized coordinates
- Metric: RMSD (Root Mean Square Deviation)

**ProteinCoordCollator**: Data collator that adds placeholder tokens
- Critical: Appends `num_of_pl_tokens = seq_len` placeholder tokens (`<unk>`) to input_ids
- Currently only supports batch_size=1 (consistent with reference code)

### Reference Code (SFT_main.py)
The reference code from the original vision-language classification task demonstrates:
- How to add placeholder tokens at input level
- Splitting hidden states into lm_part and pl_part
- LabelWiseAttention for mapping to fixed outputs
- This project adapts these patterns from classification (logits) to coordinate regression (x,y,z)

### Data Pipeline

1. **data_preprocessing.py**: Extract sequences and Cα coordinates from PDB files
   - Reads ATOM records directly
   - Validates sequence length matches coordinate count
   - Outputs: protein_coords_data.json

2. **create_coord_data.py**: Normalize and split data
   - Per-sample center+scale normalization (removes translation and scale differences)
   - Saves per-sample normalization stats in each training sample
   - Outputs: coord_train.json, coord_val.json

3. **train_protein_structure.py**: Train MTP model
   - Uses LoRA for memory efficiency
   - Computes global normalization statistics from training set
   - Saves model + training_config.json (includes global_norm_mean and global_norm_std)

4. **infer_protein_structure.py**: Predict structure
   - Loads model + LoRA weights
   - Automatically applies denormalization using global stats from training_config.json
   - Outputs standard PDB format with coordinates in Ångström units

### Evaluation Metrics (evaluate_structure.py)

- **RMSD**: Root Mean Square Deviation with Kabsch alignment
  - <2.0Å: High quality, 2.0-5.0Å: Medium, >5.0Å: Low
- **TM-score**: Template Modeling score [0,1]
  - >0.5: Same fold, 0.4-0.5: Similar, <0.4: Different
- **GDT-TS**: Global Distance Test [0,100]
  - >50: High quality, 30-50: Medium, <30: Low
- **Contact Map**: Precision/Recall/F1 for long-range interactions

## Important Implementation Details

- **Batch size limitation**: Currently only batch_size=1 is supported (matching reference code). The collator explicitly asserts this.
- **Coordinate normalization**: Training uses per-sample center+scale normalization (`(coords - centroid) / std`). During training, global statistics are computed from the training set and saved to `training_config.json`. Inference automatically applies denormalization using these global stats to convert normalized predictions back to Ångström units.
- **Placeholder token**: Uses `<unk>` token. Must exist in tokenizer vocabulary.
- **Sequence length**: Must match between input sequence and coordinate array. Data preprocessing validates this.
- **LoRA configuration**: Applied to q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj with configurable r and alpha
- **Training config**: Saved to training_config.json in output directory, includes global normalization stats (global_norm_mean, global_norm_std) for inference denormalization

## File Organization

**Core training/inference**:
- train_protein_structure.py: Main training script (recommended)
- infer_protein_structure.py: Structure prediction
- evaluate_structure.py: Metric computation
- visualize_structure.py: 3D/2D structure visualization

**Data preparation**:
- data_preprocessing.py: PDB → sequences + coordinates
- create_coord_data.py: Normalization + train/val split
- create_mtp_data.py: Alternative data format (deprecated)

**Reference code** (do not modify):
- SFT_main.py: Original vision-language classification MTP implementation
- SFT_infer.py: Original inference script

**Documentation**:
- README.md: User-facing documentation
- PROJECT_SUMMARY.md: Complete project summary
- start.md: Requirements document
- DISCUSSION.md: Progress and technical discussions

**Deprecated**:
- train_protein_mtp.py: Old standard MTP version (not recommended)

## Technical Context

**Base Model**: Supports multiple Qwen models:
- Qwen/Qwen2.5-0.5B: Quick validation, CPU/GPU
- Qwen/Qwen3-4B: Medium scale
- Qwen/Qwen3-8B: **Recommended for formal training** (requires GPU + gradient checkpointing)

**8B Model Training** (RTX 3090 24GB):
```bash
python train_protein_structure.py \
    --model_name Qwen/Qwen3-8B \
    --gradient_checkpointing \
    --gradient_accumulation_steps 8 \
    --lora_rank 8 \
    --lora_alpha 16 \
    --learning_rate 5e-5 \
    ...
```

**Memory Optimization Options**:
- `--gradient_checkpointing`: Reduces memory by recomputing activations
- `--use_4bit`: 4-bit quantization for extreme memory savings
- `--gradient_accumulation_steps N`: Increase to reduce per-step memory

**Dataset**: PDBbind-Plus (protein-ligand binding data)
- Download to P-L/ directory
- Contains PDB files with ATOM records (sequence + 3D coordinates)

**Key Libraries**:
- torch, transformers, peft (LoRA), accelerate
- bitsandbytes (for 4bit quantization)
- numpy, matplotlib for data/visualization

**Training Strategy**:
- LoRA fine-tuning (PEFT) for memory efficiency
- MSE loss for coordinate regression
- RMSD as evaluation metric during training
- Max sequence length: 512 (configurable)
- Automatic GPU detection and dtype selection (bf16/fp16 on GPU, fp32 on CPU)

## Development Notes

- When modifying the model architecture, ensure the placeholder token mechanism remains intact
- Coordinate normalization/denormalization must stay synchronized between training and inference
- If changing batch size >1, the collator needs significant refactoring (currently hardcoded for batch_size=1)
- RMSD computation uses Kabsch algorithm for optimal rigid alignment - do not modify without understanding the math
- PDB output format in inference must follow standard conventions (ATOM records, proper spacing)

## Common Issues and Solutions

### 1. PDB File Permission Issues
**Problem**: PDB files marked as executable, causing "command not found" errors when attempting to use file paths.
**Solution**: Remove execute permissions:
```bash
find P-L/ -name "*.pdb" -type f -exec chmod 644 {} +
find P-L/ -name "*.sdf" -o -name "*.mol2" -type f -exec chmod 644 {} +
```

### 2. Missing INDEX File
**Problem**: `data_preprocessing.py` requires `index/INDEX_general_PL.2020R1.lst` which may not exist.
**Solution**: Script now auto-detects missing index and scans P-L directory directly using `scan_data_directory()` function.

### 3. NumPy/scikit-learn Compatibility
**Problem**: `ValueError: numpy.dtype size changed, may indicate binary incompatibility`
**Solution**: Reinstall scikit-learn to match current numpy version:
```bash
pip3 uninstall -y scikit-learn && pip3 install --no-cache-dir scikit-learn
```

### 4. BFloat16 CPU Incompatibility
**Problem**: `RuntimeError: "mse_cpu" not implemented for 'BFloat16'`
**Solution**: Use float32 on CPU instead of bfloat16. Model loading changed to:
```python
AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32, ...)
```

### 6. CUDA Driver Version Incompatibility
**Problem**: `CUDA initialization: The NVIDIA driver on your system is too old`
**Solution**: Install PyTorch version compatible with your CUDA driver:
```bash
# Check CUDA version: nvidia-smi
# For CUDA 11.4 (driver 470.x), use:
pip install torch==2.0.1+cu118 --index-url https://download.pytorch.org/whl/cu118
```

### 7. Out of Memory (OOM) with Large Models
**Problem**: GPU runs out of memory when training 8B model
**Solution**: Enable memory optimizations:
```bash
python train_protein_structure.py \
    --model_name Qwen/Qwen3-8B \
    --gradient_checkpointing \
    --gradient_accumulation_steps 16 \
    --use_4bit \
    ...
```

### 8. Disk Space Insufficient for Large Models
**Problem**: 8B model requires ~17GB disk space for download, root partition may be too small
**Solution**: Move HuggingFace cache to a larger disk:
```bash
# Create cache directory on large disk
mkdir -p /mnt/data/huggingface_cache

# Move existing cache and create symlink
mv ~/.cache/huggingface/* /mnt/data/huggingface_cache/
rm -rf ~/.cache/huggingface
ln -s /mnt/data/huggingface_cache ~/.cache/huggingface

# Optionally, also use large disk for training output
python train_protein_structure.py \
    --output_dir /mnt/data/protein_mtp_output/output_structure_8b \
    ...
```
