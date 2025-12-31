# Verification Using Main Training Scripts

This document describes how to verify the correctness of the three distributed training implementations (DDP, DeepSpeed AutoTP, FSDP+DTensor) using the main training scripts.

## Overview

For a fair comparison, all implementations must start from **identical initial weights**. The training scripts support this via:
- `--init_weights_path`: Path to load/save initial model weights
- `--save_init_weights`: Save initial weights before training (DDP only)

## Prerequisites

- 4 GPUs (for TP=2, DP=2 configuration)
- Model: `Qwen/Qwen2.5-0.5B` (or any compatible model)
- Python environment with PyTorch, DeepSpeed, and Ray Train installed

## Quick Start

```bash
# Create output directories
mkdir -p /tmp/shared_weights /tmp/loss_curves

# Run all three implementations with shared weights
./run_verification_main.sh
```

Or follow the step-by-step instructions below.

## Step-by-Step Verification

### Step 1: Run DDP and Save Initial Weights

DDP serves as the baseline. Run it first to create the shared initial weights:

```bash
python train_ddp.py \
    --model_name Qwen/Qwen2.5-0.5B \
    --num_workers 2 \
    --batch_size 2 \
    --seq_length 1024 \
    --num_layers 2 \
    --num_epochs 1 \
    --learning_rate 1e-5 \
    --debug_steps 100 \
    --log_interval 10 \
    --loss_output_dir /tmp/loss_curves \
    --storage_path /tmp/ray_checkpoints \
    --init_weights_path /tmp/shared_weights/init_weights.pt \
    --save_init_weights
```

This will:
- Create the model with random initialization
- Save the initial weights to `/tmp/shared_weights/init_weights.pt`
- Train for 100 steps
- Save loss history to `/tmp/loss_curves/loss_ddp.json`

### Step 2: Run DeepSpeed AutoTP with Same Weights

```bash
python train_deepspeed.py \
    --model_name Qwen/Qwen2.5-0.5B \
    --tp_size 2 \
    --dp_size 2 \
    --num_workers 4 \
    --batch_size 2 \
    --seq_length 1024 \
    --num_layers 2 \
    --num_epochs 1 \
    --learning_rate 1e-5 \
    --debug_steps 100 \
    --log_interval 10 \
    --loss_output_dir /tmp/loss_curves \
    --storage_path /tmp/ray_checkpoints \
    --init_weights_path /tmp/shared_weights/init_weights.pt
```

This will:
- Load the same initial weights from `/tmp/shared_weights/init_weights.pt`
- Apply tensor parallelism (TP=2) and data parallelism (DP=2)
- Train for 100 steps
- Save loss history to `/tmp/loss_curves/loss_deepspeed.json`

### Step 3: Run FSDP+DTensor with Same Weights

```bash
python train_fsdp.py \
    --model_name Qwen/Qwen2.5-0.5B \
    --tp_size 2 \
    --dp_size 2 \
    --num_workers 4 \
    --batch_size 2 \
    --seq_length 1024 \
    --num_layers 2 \
    --num_epochs 1 \
    --learning_rate 1e-5 \
    --debug_steps 100 \
    --log_interval 10 \
    --loss_output_dir /tmp/loss_curves \
    --storage_path /tmp/ray_checkpoints \
    --init_weights_path /tmp/shared_weights/init_weights.pt \
    --autocast
```

Note: `--autocast` is recommended to match DDP's mixed precision behavior.

This will:
- Load the same initial weights from `/tmp/shared_weights/init_weights.pt`
- Apply DTensor tensor parallelism (TP=2) and FSDP2 data parallelism (DP=2)
- Train for 100 steps
- Save loss history to `/tmp/loss_curves/loss_fsdp.json`

### Step 4: Plot and Compare Loss Curves

```bash
python plot_loss_curves.py \
    --input_dir /tmp/loss_curves \
    --output /tmp/loss_curves/loss_curves.png
```

This will:
- Load loss histories from all three implementations
- Generate a comparison plot
- Print statistics showing differences from DDP baseline

## Expected Results

When using shared initial weights, you should see:

### Initial Loss
All three implementations should have **nearly identical initial loss** (difference < 0.001):
```
DDP:       Initial=10.8732
DeepSpeed: Initial=10.8736
FSDP:      Initial=10.8736
```

### Loss Curve Matching

| Implementation | Max Diff from DDP | Mean Diff | Notes |
|----------------|-------------------|-----------|-------|
| FSDP+DTensor | < 0.02 | < 0.001 | Almost identical to DDP |
| DeepSpeed AutoTP | < 0.5 | < 0.15 | Slightly different due to gradient clipping |

### Example Output
```
============================================================
LOSS CURVE STATISTICS
============================================================

DDP (TP=1, DP=2):
  Initial loss: 10.873175
  Final loss:   4.644855
  Min loss:     0.723590
  Mean loss:    7.341280

DeepSpeed AutoTP (TP=2, DP=2):
  Initial loss: 10.873634
  Final loss:   4.424264
  Min loss:     0.475180
  Mean loss:    7.223806

FSDP+DTensor (TP=2, DP=2):
  Initial loss: 10.873634
  Final loss:   4.644554
  Min loss:     0.723624
  Mean loss:    7.341585

------------------------------------------------------------
Differences from DDP baseline:

DeepSpeed AutoTP (TP=2, DP=2):
  Max abs diff:  0.456453
  Mean abs diff: 0.119506
  Final diff:    -0.220591

FSDP+DTensor (TP=2, DP=2):
  Max abs diff:  0.019137
  Mean abs diff: 0.000863
  Final diff:    -0.000301
```

## Understanding the Differences

### Why FSDP matches DDP almost perfectly
- Both use standard PyTorch optimizer (AdamW)
- Both use the same autocast for mixed precision
- FSDP2's gradient synchronization is mathematically equivalent to DDP's all-reduce

### Why DeepSpeed has slightly larger differences
- DeepSpeed uses its own optimizer wrapper with ZeRO Stage 1
- DeepSpeed has built-in gradient clipping (`max_grad_norm=1.0`)
- DeepSpeed's bf16 handling includes master weights and gradient management
- These differences accumulate over training steps but remain small (~0.1 mean diff)

## Troubleshooting

### Large initial loss differences
If initial losses differ significantly (> 0.01):
- Verify the init weights file exists and was loaded (check for "Loaded initial weights" message)
- Ensure all scripts use the same `--model_name` and `--num_layers`

### FSDP loss diverges from DDP
- Ensure `--autocast` flag is passed to `train_fsdp.py`
- Without autocast, FSDP uses different precision handling

### DeepSpeed loss diverges significantly
- This is expected due to different optimizer/precision handling
- Mean diff < 0.2 is normal; > 0.5 may indicate configuration issues

## Configuration Reference

### Common Parameters (should match across all scripts)
| Parameter | Value | Description |
|-----------|-------|-------------|
| `--model_name` | Qwen/Qwen2.5-0.5B | Model to use |
| `--num_layers` | 2 | Number of layers (for faster testing) |
| `--batch_size` | 2 | Per-worker batch size |
| `--seq_length` | 1024 | Sequence length |
| `--learning_rate` | 1e-5 | Learning rate |
| `--debug_steps` | 100 | Number of training steps |

### Implementation-Specific Parameters
| Script | Workers | TP | DP | Extra Flags |
|--------|---------|----|----|-------------|
| train_ddp.py | 2 | 1 | 2 | `--save_init_weights` (first run only) |
| train_deepspeed.py | 4 | 2 | 2 | - |
| train_fsdp.py | 4 | 2 | 2 | `--autocast` |

## Anyscale Job Submission

For running on Anyscale, use the provided job YAML files:

### Step 1: Submit DDP Job (Creates Initial Weights)
```bash
anyscale job submit -f job_ddp.yaml
```

Wait for the job to complete. This creates:
- `/mnt/cluster_storage/shared_weights/init_weights.pt` - Shared initial weights
- `/mnt/cluster_storage/loss_curves/loss_ddp.json` - DDP loss history

### Step 2: Submit DeepSpeed Job
```bash
anyscale job submit -f job_deepspeed.yaml
```

This loads the shared weights and saves:
- `/mnt/cluster_storage/loss_curves/loss_deepspeed.json` - DeepSpeed loss history

### Step 3: Submit FSDP Job
```bash
anyscale job submit -f job_fsdp.yaml
```

This loads the shared weights and saves:
- `/mnt/cluster_storage/loss_curves/loss_fsdp.json` - FSDP loss history

### Step 4: Plot Results
After all jobs complete, run the plotting script:
```bash
python plot_loss_curves.py \
    --input_dir /mnt/cluster_storage/loss_curves \
    --output /mnt/cluster_storage/loss_curves/loss_curves.png
```

## Files

| File | Description |
|------|-------------|
| `train_ddp.py` | DDP baseline training script |
| `train_deepspeed.py` | DeepSpeed AutoTP training script |
| `train_fsdp.py` | FSDP2+DTensor training script |
| `plot_loss_curves.py` | Loss curve plotting and comparison |
| `ddp_strategy.py` | DDP strategy implementation |
| `autotp_strategy.py` | DeepSpeed AutoTP strategy implementation |
| `fsdp_strategy.py` | FSDP2+DTensor strategy implementation |
| `job_ddp.yaml` | Anyscale job config for DDP (run first) |
| `job_deepspeed.yaml` | Anyscale job config for DeepSpeed AutoTP |
| `job_fsdp.yaml` | Anyscale job config for FSDP+DTensor |
