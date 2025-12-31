"""
FSDP+DTensor verification script.

This script:
1. Loads initial weights created by verify_ddp.py
2. Loads the fixed input batch
3. Runs a single forward/backward pass with hooks
4. Dumps outputs and gradients for comparison

Usage:
    python verify_fsdp.py --input_dir /tmp/verify_tp --output_dir /tmp/verify_tp
"""

import argparse
import os
from typing import Any, Dict

os.environ["RAY_TRAIN_V2_ENABLED"] = "1"

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor.parallel import parallelize_module

import ray
import ray.train
import ray.train.torch
from ray.train import RunConfig, ScalingConfig
from ray.train.torch import TorchTrainer

from model_builder import create_model, get_transformer_layers
from verify_utils import (
    HookManager,
    VerificationDump,
    collect_param_grad_norms,
    collect_param_gradients,
    load_fixed_input,
    load_initial_weights,
    save_verification_dump,
)


def train_loop_per_worker(config: Dict[str, Any]) -> None:
    """Training loop for verification."""
    from torch.distributed.tensor import Replicate, Shard
    from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel

    ctx = ray.train.get_context()
    world_rank = ctx.get_world_rank()
    world_size = ctx.get_world_size()
    device = ray.train.torch.get_device()

    tp_size = config["tp_size"]
    dp_size = config["dp_size"]
    input_dir = config["input_dir"]
    output_dir = config["output_dir"]

    print(f"[Rank {world_rank}] FSDP+DTensor verification starting")
    print(f"[Rank {world_rank}] tp_size={tp_size}, dp_size={dp_size}, world_size={world_size}")

    # Calculate TP/DP ranks
    tp_rank = world_rank % tp_size
    dp_rank = world_rank // tp_size

    # Create 2D device mesh: (dp, tp)
    device_mesh = init_device_mesh(
        "cuda", (dp_size, tp_size), mesh_dim_names=("dp", "tp")
    )
    tp_mesh = device_mesh["tp"]

    if world_rank == 0:
        print(f"[Rank 0] Created device mesh: {device_mesh}")

    # Create model
    model = create_model(
        model_name=config["model_name"],
        device=device,
        dtype=torch.bfloat16,
        num_layers=config["num_layers"],
        attn_impl="sdpa",
    )

    # Load shared initial weights BEFORE TP sharding
    weights_path = os.path.join(input_dir, "initial_weights.pt")
    load_initial_weights(model, weights_path)

    # Create verification dump
    dump = VerificationDump(
        implementation="fsdp_dtensor",
        tp_size=tp_size,
        dp_size=dp_size,
        rank=world_rank,
    )

    # Get transformer layers
    layers = get_transformer_layers(model)
    if layers is None:
        raise ValueError("Could not find transformer layers in model")

    # TP mapping for transformer layers (Qwen/Llama-style models)
    tp_mapping = {
        "self_attn.q_proj": ColwiseParallel(),
        "self_attn.k_proj": ColwiseParallel(),
        "self_attn.v_proj": ColwiseParallel(),
        "self_attn.o_proj": RowwiseParallel(),
        "mlp.gate_proj": ColwiseParallel(),
        "mlp.up_proj": ColwiseParallel(),
        "mlp.down_proj": RowwiseParallel(),
    }

    if world_rank == 0:
        print(f"[Rank 0] Applying DTensor TP to {len(layers)} layers")

    # Apply DTensor TP to transformer layers
    for layer in layers:
        parallelize_module(layer, tp_mesh, tp_mapping)

    # Register hooks AFTER TP sharding
    hook_manager = HookManager(dump)
    layer_patterns = [
        "embed_tokens",
        "self_attn",
        "mlp",
        "input_layernorm",
        "post_attention_layernorm",
        "norm",
        "lm_head",
        "layers.0",
        "layers.1",
    ]
    hook_manager.register_hooks(model, layer_patterns)

    # Create optimizer (foreach=False for DTensor compatibility)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, foreach=False)

    # Load fixed input (global batch) and take this DP rank's portion
    global_batch = load_fixed_input(os.path.join(input_dir, "fixed_input.pt"))
    batch_size = config["batch_size"]

    # Each DP rank gets its slice of the global batch
    # All TP ranks within the same DP group see the same data
    start_idx = dp_rank * batch_size
    end_idx = start_idx + batch_size
    batch = {
        k: v[start_idx:end_idx].to(device)
        for k, v in global_batch.items()
    }

    if world_rank == 0:
        print(f"[Rank 0] Global batch size: {global_batch['input_ids'].shape[0]}")
        print(f"[Rank 0] Local batch indices (dp_rank={dp_rank}): [{start_idx}:{end_idx}]")

    # Forward pass
    if world_rank == 0:
        print("[Rank 0] Running forward pass...")

    model.train()
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
            use_cache=False,
        )
        loss = outputs.loss

    dump.loss = loss.item()
    if world_rank == 0:
        print(f"[Rank 0] Loss: {dump.loss:.6f}")

    # Backward pass
    if world_rank == 0:
        print("[Rank 0] Running backward pass...")

    loss.backward()

    # Collect parameter gradient norms
    dump.param_grad_norms = collect_param_grad_norms(model)

    # Also collect full gradients for detailed comparison
    gradients = collect_param_gradients(model)
    if gradients:
        grad_path = os.path.join(output_dir, f"gradients_fsdp_rank{world_rank}.pt")
        torch.save(gradients, grad_path)
        if world_rank == 0:
            print(f"[Rank 0] Saved {len(gradients)} parameter gradients to {grad_path}")

    # Finalize hooks
    hook_manager.finalize()
    hook_manager.remove_hooks()

    # Save verification dump (each rank saves its own)
    dump_path = os.path.join(output_dir, f"dump_fsdp_rank{world_rank}.json")
    save_verification_dump(dump, dump_path)

    # Print summary from rank 0
    if world_rank == 0:
        print("\n" + "=" * 60)
        print("FSDP+DTensor Verification Summary (Rank 0)")
        print("=" * 60)
        print(f"Loss: {dump.loss:.6f}")
        print(f"\nLayer output norms:")
        for name, layer in sorted(dump.layers.items()):
            if layer.output_norm > 0:
                print(f"  {name}: output_norm={layer.output_norm:.6f}, shape={layer.output_shape}")

    # Synchronize before exit
    dist.barrier()
    ray.train.report({"loss": dump.loss, "rank": world_rank})


def main():
    parser = argparse.ArgumentParser(description="FSDP+DTensor Verification")
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen2.5-0.5B",
        help="HuggingFace model name",
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="/tmp/verify_tp",
        help="Input directory with weights and fixed input",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/tmp/verify_tp",
        help="Output directory for dumps",
    )
    parser.add_argument(
        "--tp_size",
        type=int,
        default=2,
        help="Tensor parallel size",
    )
    parser.add_argument(
        "--dp_size",
        type=int,
        default=1,
        help="Data parallel size",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="Batch size",
    )
    parser.add_argument(
        "--seq_length",
        type=int,
        default=1024,
        help="Sequence length",
    )
    parser.add_argument(
        "--num_layers",
        type=int,
        default=2,
        help="Number of transformer layers",
    )

    args = parser.parse_args()

    num_workers = args.tp_size * args.dp_size
    print(f"Configuration: tp_size={args.tp_size}, dp_size={args.dp_size}, num_workers={num_workers}")

    os.makedirs(args.output_dir, exist_ok=True)

    # Check that input files exist
    weights_path = os.path.join(args.input_dir, "initial_weights.pt")
    input_path = os.path.join(args.input_dir, "fixed_input.pt")
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Initial weights not found at {weights_path}. Run verify_ddp.py first.")
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Fixed input not found at {input_path}. Run verify_ddp.py first.")

    train_loop_config = {
        "model_name": args.model_name,
        "input_dir": args.input_dir,
        "output_dir": args.output_dir,
        "tp_size": args.tp_size,
        "dp_size": args.dp_size,
        "batch_size": args.batch_size,
        "seq_length": args.seq_length,
        "num_layers": args.num_layers,
    }

    scaling_config = ScalingConfig(
        num_workers=num_workers,
        use_gpu=True,
    )

    run_config = RunConfig(
        storage_path="/tmp",
        name="verify_fsdp",
    )

    trainer = TorchTrainer(
        train_loop_per_worker=train_loop_per_worker,
        scaling_config=scaling_config,
        train_loop_config=train_loop_config,
        run_config=run_config,
    )

    result = trainer.fit()
    print(f"Verification finished. Result: {result}")


if __name__ == "__main__":
    main()
