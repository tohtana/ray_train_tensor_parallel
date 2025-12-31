"""
DDP verification script with DP=2 support.

This script runs DDP with multiple workers for data parallelism comparison.
The global batch is split across DP ranks.

Usage:
    python verify_ddp_dp2.py --input_dir /tmp/verify_tp_dp2 --output_dir /tmp/verify_tp_dp2 --dp_size 2
"""

import argparse
import os
from typing import Any, Dict

os.environ["RAY_TRAIN_V2_ENABLED"] = "1"

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import ray
import ray.train
import ray.train.torch
from ray.train import RunConfig, ScalingConfig
from ray.train.torch import TorchTrainer

from model_builder import create_model
from verify_utils import (
    HookManager,
    VerificationDump,
    collect_param_grad_norms,
    load_fixed_input,
    load_initial_weights,
    save_verification_dump,
)


def train_loop_per_worker(config: Dict[str, Any]) -> None:
    """Training loop for DDP verification with DP>1."""
    ctx = ray.train.get_context()
    world_rank = ctx.get_world_rank()
    world_size = ctx.get_world_size()
    device = ray.train.torch.get_device()

    input_dir = config["input_dir"]
    output_dir = config["output_dir"]
    dp_size = config["dp_size"]

    print(f"[Rank {world_rank}] DDP verification starting")
    print(f"[Rank {world_rank}] dp_size={dp_size}, world_size={world_size}")

    # For DDP: dp_rank = world_rank, dp_size = world_size
    dp_rank = world_rank

    # Create model
    model = create_model(
        model_name=config["model_name"],
        device=device,
        dtype=torch.bfloat16,
        num_layers=config["num_layers"],
        attn_impl="sdpa",
    )

    # Load shared initial weights
    weights_path = os.path.join(input_dir, "initial_weights.pt")
    load_initial_weights(model, weights_path)

    # Wrap with DDP
    ddp_model = DDP(
        model,
        device_ids=[device.index] if device.type == "cuda" else None,
        output_device=device.index if device.type == "cuda" else None,
        find_unused_parameters=False,
    )

    # Create verification dump
    dump = VerificationDump(
        implementation="ddp",
        tp_size=1,
        dp_size=dp_size,
        rank=world_rank,
    )

    # Register hooks
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

    # Create optimizer
    optimizer = torch.optim.AdamW(ddp_model.parameters(), lr=1e-5)

    # Load fixed input (global batch) and take this rank's portion
    global_batch = load_fixed_input(os.path.join(input_dir, "fixed_input.pt"))
    batch_size = config["batch_size"]

    # Each DP rank gets its slice of the global batch
    start_idx = dp_rank * batch_size
    end_idx = start_idx + batch_size
    batch = {
        k: v[start_idx:end_idx].to(device)
        for k, v in global_batch.items()
    }

    if world_rank == 0:
        print(f"[Rank 0] Global batch size: {global_batch['input_ids'].shape[0]}")
        print(f"[Rank 0] Local batch indices: [{start_idx}:{end_idx}]")

    # Forward pass
    if world_rank == 0:
        print("[Rank 0] Running forward pass...")

    ddp_model.train()
    optimizer.zero_grad()

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        outputs = ddp_model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
            use_cache=False,
        )
        loss = outputs.loss

    dump.loss = loss.item()
    print(f"[Rank {world_rank}] Local loss: {dump.loss:.6f}")

    # Backward pass
    if world_rank == 0:
        print("[Rank 0] Running backward pass...")

    loss.backward()

    # Collect parameter gradient norms (after DDP gradient sync)
    dump.param_grad_norms = collect_param_grad_norms(model)

    # Finalize hooks
    hook_manager.finalize()
    hook_manager.remove_hooks()

    # Save verification dump (each rank saves its own)
    dump_path = os.path.join(output_dir, f"dump_ddp_rank{world_rank}.json")
    save_verification_dump(dump, dump_path)

    # Print summary from rank 0
    if world_rank == 0:
        print("\n" + "=" * 60)
        print("DDP Verification Summary (Rank 0)")
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
    parser = argparse.ArgumentParser(description="DDP Verification with DP>1")
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen2.5-0.5B",
        help="HuggingFace model name",
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="/tmp/verify_tp_dp2",
        help="Input directory with weights and fixed input",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/tmp/verify_tp_dp2",
        help="Output directory for dumps",
    )
    parser.add_argument(
        "--dp_size",
        type=int,
        default=2,
        help="Data parallel size (number of workers)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="Per-worker batch size",
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

    num_workers = args.dp_size
    print(f"Configuration: dp_size={args.dp_size}, num_workers={num_workers}")

    os.makedirs(args.output_dir, exist_ok=True)

    # Check that input files exist
    weights_path = os.path.join(args.input_dir, "initial_weights.pt")
    input_path = os.path.join(args.input_dir, "fixed_input.pt")
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Initial weights not found at {weights_path}.")
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Fixed input not found at {input_path}.")

    train_loop_config = {
        "model_name": args.model_name,
        "input_dir": args.input_dir,
        "output_dir": args.output_dir,
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
        name="verify_ddp_dp2",
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
