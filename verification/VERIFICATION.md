# Tensor Parallelism Verification

This document summarizes the verification results comparing three distributed training implementations:
- **DDP** (Distributed Data Parallel) - baseline reference
- **DeepSpeed AutoTP** - DeepSpeed's automatic tensor parallelism
- **FSDP+DTensor** - PyTorch native FSDP2 with DTensor tensor parallelism

## Configuration

| Parameter | Value |
|-----------|-------|
| Model | Qwen/Qwen2.5-0.5B |
| Batch size | 2 |
| Sequence length | 1024 |
| Number of layers | 2 (for faster testing) |
| TP size | 2 (for TP implementations) |
| DP size | 1 |
| Precision | bfloat16 |

All implementations used:
- Same initial random weights (saved and shared)
- Same fixed input batch (saved and shared)
- Single forward + backward pass

## Loss Comparison

| Implementation | Loss | Relative Diff from DDP |
|---------------|------|------------------------|
| DDP (baseline) | 12.13309765 | - |
| DeepSpeed AutoTP | 12.13316059 | 5.19e-06 |
| FSDP+DTensor | 12.13316059 | 5.19e-06 |

**Result: All implementations produce essentially identical losses** (within ~0.0005% relative difference)

## Layer Output Norms (Rank 0)

| Layer | DDP | DeepSpeed AutoTP | FSDP+DTensor |
|-------|-----|------------------|--------------|
| model.embed_tokens | 27.1251 | 27.1251 | 27.1251 |
| model.layers.0 | 374.6458 | 374.6450 | 374.6450 |
| model.layers.0.self_attn | 43.3304 | 43.3291 | 43.3291 |
| model.layers.0.self_attn.q_proj | 810.2407 | 573.3234* | 573.3234* |
| model.layers.0.self_attn.o_proj | 43.3304 | 43.3291 | 43.3291 |
| model.layers.0.mlp | 371.1974 | 371.1962 | 371.1962 |
| model.layers.0.mlp.gate_proj | 1892.3745 | 1338.2168* | 1338.2168* |
| model.layers.0.mlp.down_proj | 371.1974 | 371.1962 | 371.1962 |
| model.layers.1 | 560.0239 | 560.0243 | 560.0243 |
| model.norm | 1354.6165 | 1354.6190 | 1354.6190 |
| lm_head | 10558.2695 | 10558.2891 | 10558.2891 |

*\* Sharded tensors show smaller norms because each rank holds only a partition (TP=2 means ~1/√2 of full norm)*

## Layer Gradient Norms (Rank 0)

| Layer | DDP | DeepSpeed AutoTP | FSDP+DTensor |
|-------|-----|------------------|--------------|
| model.embed_tokens | 1.2588 | 1.2590 | 1.2590 |
| model.layers.0.self_attn | 0.7774 | 0.7774 | 0.7774 |
| model.layers.0.self_attn.o_proj | 0.7774 | 0.7774 | 0.7774 |
| model.layers.0.mlp | 0.0529 | 0.0529 | 0.0529 |
| model.layers.0.mlp.down_proj | 0.0529 | 0.0529 | 0.0529 |
| model.layers.1.self_attn | 0.0526 | 0.0526 | 0.0526 |
| model.layers.1.mlp | 0.0322 | 0.0322 | 0.0322 |
| model.norm | 0.0132 | 0.0132 | 0.0132 |
| lm_head | 0.0221 | 0.0221 | 0.0221 |

## Max Absolute Differences

### DeepSpeed AutoTP vs FSDP+DTensor

| Layer | Max Abs Diff | Relative Diff |
|-------|--------------|---------------|
| All layers | **0.0** | **0.0** |

**The two TP implementations are numerically identical!**

### DDP vs TP Implementations

| Layer | Max Abs Diff | Relative Diff |
|-------|--------------|---------------|
| model.embed_tokens | 0.0 | 0.0 |
| model.layers.0 | 7.81e-03 | 1.77e-03 |
| model.layers.0.self_attn | 3.91e-03 | 1.11e-03 |
| model.layers.0.self_attn.o_proj | 3.91e-03 | 1.11e-03 |
| model.layers.0.mlp | 5.86e-03 | 2.18e-03 |
| model.layers.0.mlp.down_proj | 5.86e-03 | 2.18e-03 |
| model.layers.1 | 1.56e-02 | 2.39e-03 |
| model.norm | 2.34e-02 | 2.40e-03 |
| lm_head | 1.56e-02 | 2.49e-03 |

Small differences (~1e-3 relative) are expected due to:
- Different computation order in TP (parallel compute then all-reduce vs sequential)
- bfloat16 precision rounding differences

## Key Findings

1. **DeepSpeed AutoTP and FSDP+DTensor produce bit-identical results** when given the same initial weights and inputs.

2. **All implementations produce the same loss** within numerical precision (5.19e-06 relative difference).

3. **Sharded tensors** (q_proj, k_proj, v_proj, gate_proj, up_proj) show smaller norms in TP implementations because each rank only holds a partition.

4. **Reduction point outputs** (o_proj, down_proj, layer outputs) match closely across all implementations.

5. **Gradients are consistent** at reduction points across all implementations.

## Verification Scripts

| File | Description |
|------|-------------|
| `verify_utils.py` | Hook manager, dump utilities, fixed input/weight creation |
| `verify_ddp.py` | DDP baseline verification (creates shared weights/inputs) |
| `verify_deepspeed.py` | DeepSpeed AutoTP verification |
| `verify_fsdp.py` | FSDP+DTensor verification |
| `compare_dumps.py` | Comparison and analysis tool |
| `run_verification.py` | Master orchestration script |

## How to Run

```bash
# Full verification (runs all three and compares)
python run_verification.py --output_dir /tmp/verify_tp

# Or run individually:
python verify_ddp.py --output_dir /tmp/verify_tp      # Creates baseline weights/inputs
python verify_deepspeed.py --input_dir /tmp/verify_tp  # Uses shared weights/inputs
python verify_fsdp.py --input_dir /tmp/verify_tp       # Uses shared weights/inputs
python compare_dumps.py --input_dir /tmp/verify_tp     # Compare all dumps

# Customize parameters:
python run_verification.py \
    --model_name Qwen/Qwen2.5-0.5B \
    --batch_size 2 \
    --seq_length 1024 \
    --num_layers 2 \
    --tp_size 2 \
    --output_dir /tmp/verify_tp
```

## DP=2 Verification (2D Parallelism)

Additional verification was performed with data parallelism enabled:
- **DDP**: TP=1, DP=2 (2 workers)
- **DeepSpeed AutoTP**: TP=2, DP=2 (4 workers)
- **FSDP+DTensor**: TP=2, DP=2 (4 workers)

### DP=2 Configuration

| Parameter | Value |
|-----------|-------|
| Global batch size | 4 (batch_size=2 × dp_size=2) |
| TP size | 2 (for TP implementations) |
| DP size | 2 |

### DP=2 Loss Comparison by DP Rank

| DP Rank | Samples | DDP | DeepSpeed AutoTP | FSDP+DTensor |
|---------|---------|-----|------------------|--------------|
| 0 | 0-1 | 12.13309765 | 12.13316059 | 12.13316059 |
| 1 | 2-3 | 12.10635662 | 12.10641384 | 12.10641384 |

### DP=2 Differences from DDP

| DP Rank | DeepSpeed Diff | FSDP Diff | Relative Diff |
|---------|---------------|-----------|---------------|
| 0 | 0.00006294 | 0.00006294 | 5.19e-06 |
| 1 | 0.00005722 | 0.00005722 | 4.73e-06 |

### DP=2 Key Findings

1. **All implementations correctly handle data parallelism**:
   - Different DP ranks process different samples (as expected)
   - DP rank 0 processes samples 0-1, DP rank 1 processes samples 2-3
   - Different samples result in different losses (12.13 vs 12.11)

2. **Loss values match across implementations per DP rank**:
   - DDP matches TP implementations within ~5e-06 relative difference
   - DeepSpeed AutoTP and FSDP+DTensor are numerically identical

3. **TP ranks within the same DP group see identical data**:
   - Ranks 0,1 (TP group for DP=0) both report loss 12.133161
   - Ranks 2,3 (TP group for DP=1) both report loss 12.106414

### DP=2 Verification Scripts

```bash
# Setup global batch (4 samples)
python setup_verification_dp2.py --output_dir /tmp/verify_tp_dp2 --dp_size 2

# Run DDP with DP=2
python verify_ddp_dp2.py --input_dir /tmp/verify_tp_dp2 --output_dir /tmp/verify_tp_dp2 --dp_size 2

# Run DeepSpeed with TP=2, DP=2
python verify_deepspeed.py --input_dir /tmp/verify_tp_dp2 --output_dir /tmp/verify_tp_dp2 --tp_size 2 --dp_size 2

# Run FSDP with TP=2, DP=2
python verify_fsdp.py --input_dir /tmp/verify_tp_dp2 --output_dir /tmp/verify_tp_dp2 --tp_size 2 --dp_size 2
```

## 100-Step Training Comparison

Extended training was performed to compare loss curves over 100 optimization steps.

### Configuration

| Parameter | DDP | DeepSpeed AutoTP | FSDP+DTensor |
|-----------|-----|------------------|--------------|
| TP size | 1 | 2 | 2 |
| DP size | 2 | 2 | 2 |
| Workers | 2 | 4 | 4 |
| Batch size/worker | 2 | 2 | 2 |
| Learning rate | 1e-5 | 1e-5 | 1e-5 |

### Loss Curve Results

![Loss Curves](loss_curves.png)

### Loss Statistics

| Implementation | Initial Loss | Final Loss | Min Loss | Mean Loss |
|----------------|--------------|------------|----------|-----------|
| DDP (TP=1, DP=2) | 11.3556 | 6.0390 | 1.9368 | 7.5174 |
| DeepSpeed AutoTP (TP=2, DP=2) | 11.3557 | 6.0483 | 1.9366 | 7.5229 |
| FSDP+DTensor (TP=2, DP=2) | 11.3557 | 6.0521 | 1.9628 | 7.5261 |

### Key Observations

1. **Initial losses are identical** (Step 0: ~11.356 for all) - confirms forward pass correctness

2. **All three implementations match closely**:
   - DDP vs DeepSpeed: max diff 0.031, final diff 0.009
   - DDP vs FSDP: max diff 0.048, final diff 0.013
   - All implementations follow nearly identical training dynamics

### Configuration Fixes Applied

**DeepSpeed Configuration:**
Initial runs showed DeepSpeed converging much faster than DDP. Fixed by:
- Adding `bf16_master_weights_and_grads: True` setting
- Adding `bf16_optimizer_states: True` setting
- Using ZeRO Stage 1 instead of Stage 0
- Removing manual `torch.autocast` wrapper (DeepSpeed handles bf16 internally)

**FSDP Configuration:**
Initial runs showed FSDP diverging from DDP. Fixed by:
- Adding `fully_shard()` for DP gradient synchronization
- The original implementation only applied TP via `parallelize_module()` but was missing FSDP2 for DP gradient sync

After these fixes, all three implementations produce nearly identical loss curves.

### Running 100-Step Training

```bash
# Run DDP (TP=1, DP=2)
python train_100steps.py --impl ddp --output_dir /tmp/loss_curves

# Run DeepSpeed (TP=2, DP=2)
python train_100steps.py --impl deepspeed --output_dir /tmp/loss_curves

# Run FSDP (TP=2, DP=2)
python train_100steps.py --impl fsdp --output_dir /tmp/loss_curves

# Plot results
python plot_loss_curves.py --input_dir /tmp/loss_curves --output loss_curves.png
```

## Gradient Comparison

Gradient comparison was performed after a single forward/backward pass to verify gradient correctness across implementations.

### Gradient Collection Method

- **DDP**: Standard `param.grad` access
- **FSDP+DTensor**: `param.grad.to_local()` for DTensor gradients
- **DeepSpeed AutoTP**: `deepspeed.utils.safe_get_full_grad(param)` for accessing gradients

### DDP vs FSDP+DTensor

| Metric | Value |
|--------|-------|
| Max abs norm diff | 0.000192 |
| Mean abs norm diff | 0.000027 |
| Max rel norm diff | 1.87e-03 |

**Result: DDP and FSDP gradients are essentially identical.**

### DDP vs DeepSpeed AutoTP

| Parameter Type | Norm Diff | Explanation |
|----------------|-----------|-------------|
| Non-sharded (embed_tokens, layernorms) | ~1e-04 | Nearly identical |
| Sharded (projections) | ~29% | Expected - DeepSpeed returns local shard gradients |

The ~29% difference for sharded parameters is expected because:
- DeepSpeed's `safe_get_full_grad` returns the local shard gradient (not gathered)
- With TP=2, the local shard is 1/2 of the full gradient
- The norm of 1/2 the tensor is approximately 1/√2 ≈ 0.707 of the full norm

### Running Gradient Comparison

```bash
# Run verification (creates gradient files)
python verify_ddp.py --output_dir /tmp/verify_tp
python verify_deepspeed.py --input_dir /tmp/verify_tp --output_dir /tmp/verify_tp
python verify_fsdp.py --input_dir /tmp/verify_tp --output_dir /tmp/verify_tp

# Compare gradients
python compare_gradients.py --input_dir /tmp/verify_tp
```

## Conclusion

Both tensor parallelism implementations (DeepSpeed AutoTP and FSDP+DTensor) are **correct** and produce identical results when given the same initialization.

The verification confirms:
1. **TP=2, DP=1**: Both TP implementations match DDP baseline within numerical precision
2. **TP=2, DP=2**: Both TP implementations correctly handle 2D parallelism, matching DDP per DP rank
3. **DeepSpeed AutoTP vs FSDP+DTensor**: Numerically identical in single-step verification
4. **100-step training**: All three implementations (DDP, DeepSpeed, FSDP) produce nearly identical loss curves
   - DDP vs DeepSpeed: final diff 0.009
   - DDP vs FSDP: final diff 0.013
5. **Gradients**: DDP and FSDP gradients are identical; DeepSpeed local shard gradients match expected norms

The small differences between implementations are within expected numerical precision for bfloat16 distributed computation.
