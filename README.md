# Ray Train + Tensor Parallelism

Training examples combining Ray Train with tensor parallelism using either DeepSpeed AutoTP or PyTorch native FSDP2+DTensor.

## Features

- **2D Parallelism**: Combine tensor parallelism (TP) with data parallelism (DP)
- **Ray Train Integration**: Distributed execution, checkpointing, and fault tolerance
- **Two Implementations**:
  - `train_deepspeed.py`: DeepSpeed AutoTP with ZeRO optimization
  - `train_fsdp.py`: PyTorch native FSDP2 + DTensor

## Quick Start

```bash
# DeepSpeed AutoTP: 4 GPUs with 2-way TP, 2-way DP
python train_deepspeed.py \
    --model_name Qwen/Qwen2-7B \
    --tp_size 2 \
    --dp_size 2 \
    --num_workers 4 \
    --num_epochs 3

# FSDP+DTensor: 4 GPUs with 2-way TP, 2-way DP
python train_fsdp.py \
    --model_name Qwen/Qwen2-7B \
    --tp_size 2 \
    --dp_size 2 \
    --num_workers 4 \
    --num_epochs 3
```

## Common Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--model_name` | HuggingFace model name | `Qwen/Qwen2-7B` |
| `--tp_size` | Tensor parallel degree | Required |
| `--dp_size` | Data parallel degree | `1` |
| `--num_workers` | Total workers (must equal tp_size * dp_size) | Required |
| `--batch_size` | Per-GPU micro batch size | `1` |
| `--seq_length` | Maximum sequence length | `2048` |
| `--num_epochs` | Number of training epochs | `3` |
| `--learning_rate` | Learning rate | `1e-5` |
| `--dataset_name` | HuggingFace dataset | `wikitext` |
| `--storage_path` | Checkpoint storage path | `/mnt/cluster_storage` |
| `--resume_from` | Experiment name to resume from | None |

### DeepSpeed-specific

| Argument | Description | Default |
|----------|-------------|---------|
| `--zero_stage` | ZeRO optimization stage (0-2) | `1` |

### FSDP-specific

| Argument | Description | Default |
|----------|-------------|---------|
| `--autocast` | Enable torch.autocast for mixed precision | `False` |

## File Structure

```
train_tensor_parallel_deepspeed/
├── train_deepspeed.py     # DeepSpeed AutoTP entry point
├── train_fsdp.py          # FSDP+DTensor entry point
├── common.py              # Shared utilities (training loop, checkpointing)
├── autotp_strategy.py     # DeepSpeed AutoTP strategy
├── fsdp_strategy.py       # FSDP2+DTensor strategy
├── vocab_parallel.py      # Vocabulary parallel embedding and loss
├── model_builder.py       # Model creation utilities
└── data.py                # TP-aware data loading
```

## How 2D Parallelism Works

With `tp_size=2` and `dp_size=2` on 4 GPUs:

```
Device Mesh (2x2):
        TP Dim
      [0]  [1]
DP   ┌───┬───┐
Dim  │ 0 │ 1 │  ← TP Group 0 (same data, sharded model)
     ├───┼───┤
     │ 2 │ 3 │  ← TP Group 1 (same data, sharded model)
     └───┴───┘
       ↑   ↑
      DP Groups (different data, gradient sync)
```

- **TP Groups** (rows): GPUs 0,1 and GPUs 2,3 share the same input data but have sharded model weights
- **DP Groups** (columns): GPUs 0,2 and GPUs 1,3 see different data and synchronize gradients

## Key Implementation Details

### TP-Aware Data Loading

Standard data loaders shard by `world_rank`, giving each GPU different data. With tensor parallelism, all GPUs in a TP group must see identical data. The `data.py` module handles this by sharding based on `dp_rank` instead:

```python
# All TP ranks in same DP group get identical batches
sampler = DistributedSampler(
    dataset,
    num_replicas=dp_size,  # NOT world_size
    rank=dp_rank,          # NOT world_rank
)
```

### Vocabulary Parallelism

Both implementations shard the embedding and output projection layers across the vocabulary dimension for memory efficiency. This requires special handling for the cross-entropy loss computation.

### Checkpointing

All workers save their model shards independently. Ray Train aggregates these into a single checkpoint that can be used for resuming training.
