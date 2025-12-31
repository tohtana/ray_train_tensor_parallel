"""
Run 100 training steps and save loss history for comparison.

This script runs training for all three implementations and saves loss histories
for plotting comparison.

Usage:
    # Run DDP (TP=1, DP=2)
    python train_100steps.py --impl ddp --output_dir /tmp/loss_curves

    # Run DeepSpeed (TP=2, DP=2)
    python train_100steps.py --impl deepspeed --output_dir /tmp/loss_curves

    # Run FSDP (TP=2, DP=2)
    python train_100steps.py --impl fsdp --output_dir /tmp/loss_curves
"""

import argparse
import json
import os
from typing import Any, Dict, List

os.environ["RAY_TRAIN_V2_ENABLED"] = "1"

import torch
import torch.distributed as dist

import ray
import ray.train
import ray.train.torch
from ray.train import RunConfig, ScalingConfig
from ray.train.torch import TorchTrainer

from model_builder import create_model, get_transformer_layers
from verify_utils import load_initial_weights


def create_dataloader(model_name: str, batch_size: int, seq_length: int, dp_rank: int, dp_size: int):
    """Create dataloader with TP-aware sharding."""
    from data import create_tp_aware_dataloader
    return create_tp_aware_dataloader(
        model_name=model_name,
        dataset_name="wikitext",
        seq_length=seq_length,
        batch_size=batch_size,
        dp_rank=dp_rank,
        dp_size=dp_size,
        seed=42,
        dataset_percentage=10.0,
    )


def train_ddp(config: Dict[str, Any]) -> None:
    """Training loop for DDP."""
    from torch.nn.parallel import DistributedDataParallel as DDP

    ctx = ray.train.get_context()
    world_rank = ctx.get_world_rank()
    world_size = ctx.get_world_size()
    device = ray.train.torch.get_device()

    # Create model
    model = create_model(
        model_name=config["model_name"],
        device=device,
        dtype=torch.bfloat16,
        num_layers=config["num_layers"],
        attn_impl="sdpa",
    )

    # Load shared initial weights
    load_initial_weights(model, config["weights_path"])

    # Wrap with DDP
    ddp_model = DDP(model, device_ids=[device.index])

    # Create optimizer
    optimizer = torch.optim.AdamW(ddp_model.parameters(), lr=config["learning_rate"])

    # Create dataloader
    dataloader = create_dataloader(
        model_name=config["model_name"],
        batch_size=config["batch_size"],
        seq_length=config["seq_length"],
        dp_rank=world_rank,
        dp_size=world_size,
    )

    # Training loop
    ddp_model.train()
    loss_history = []
    step = 0

    for batch in dataloader:
        if step >= config["num_steps"]:
            break

        batch = {k: v.to(device) for k, v in batch.items()}
        optimizer.zero_grad()

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs = ddp_model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
                use_cache=False,
            )
            loss = outputs.loss

        loss.backward()

        # Gradient clipping (same as DeepSpeed default)
        torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), max_norm=1.0)

        optimizer.step()

        # All-reduce loss for logging (average across DP ranks)
        loss_tensor = loss.detach().clone()
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
        loss_history.append(loss_tensor.item())

        if world_rank == 0 and step % 10 == 0:
            print(f"[DDP] Step {step}: loss={loss_tensor.item():.4f}")

        step += 1

    # Save loss history from rank 0
    if world_rank == 0:
        output_path = os.path.join(config["output_dir"], "loss_ddp.json")
        with open(output_path, "w") as f:
            json.dump({"implementation": "ddp", "loss_history": loss_history}, f)
        print(f"Saved loss history to {output_path}")

    dist.barrier()
    ray.train.report({"final_loss": loss_history[-1] if loss_history else 0})


def train_deepspeed(config: Dict[str, Any]) -> None:
    """Training loop for DeepSpeed AutoTP."""
    import deepspeed
    from deepspeed.module_inject.layers import set_autotp_mode

    ctx = ray.train.get_context()
    world_rank = ctx.get_world_rank()
    world_size = ctx.get_world_size()
    device = ray.train.torch.get_device()

    tp_size = config["tp_size"]
    dp_size = config["dp_size"]

    # Initialize DeepSpeed distributed
    deepspeed.init_distributed()

    # Calculate TP/DP ranks
    tp_rank = world_rank % tp_size
    dp_rank = world_rank // tp_size

    # Create TP process group
    tp_group = None
    for dp_idx in range(dp_size):
        tp_group_ranks = list(range(dp_idx * tp_size, (dp_idx + 1) * tp_size))
        group = dist.new_group(tp_group_ranks)
        if world_rank in tp_group_ranks:
            tp_group = group

    # Enable AutoTP
    set_autotp_mode(training=True)

    # Create model
    model = create_model(
        model_name=config["model_name"],
        device=device,
        dtype=torch.bfloat16,
        num_layers=config["num_layers"],
        attn_impl="sdpa",
    )

    # Load shared initial weights
    load_initial_weights(model, config["weights_path"])

    # Sync weights within TP group
    if tp_group is not None and tp_size > 1:
        tp_group_first_rank = dp_rank * tp_size
        with torch.no_grad():
            for param in model.parameters():
                dist.broadcast(param.data, src=tp_group_first_rank, group=tp_group)

    # Apply TP sharding
    model = deepspeed.tp_model_init(model, tp_size=tp_size, dtype=torch.bfloat16, tp_group=tp_group)

    # DeepSpeed config - match original autotp_strategy.py settings
    # Use ZeRO Stage 1 (default in autotp_strategy.py) for bf16 master weights support
    ds_config = {
        "train_batch_size": config["batch_size"] * dp_size,
        "train_micro_batch_size_per_gpu": config["batch_size"],
        "gradient_accumulation_steps": 1,
        "gradient_clipping": 1.0,  # Same as DDP/FSDP
        "zero_optimization": {"stage": 1, "overlap_comm": True},
        "bf16": {
            "enabled": True,
            "bf16_master_weights_and_grads": True,
            "bf16_optimizer_states": True,
        },
        "tensor_parallel": {"autotp_size": tp_size},
    }

    # Create optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"])

    # MPU for DeepSpeed
    class SimpleMPU:
        def __init__(self, tp_group, dp_size, tp_size, tp_rank, dp_rank):
            self._tp_group = tp_group
            self._dp_size = dp_size
            self._tp_size = tp_size
            self._tp_rank = tp_rank
            self._dp_rank = dp_rank

        def get_model_parallel_group(self): return self._tp_group
        def get_model_parallel_world_size(self): return self._tp_size
        def get_model_parallel_rank(self): return self._tp_rank
        def get_data_parallel_world_size(self): return self._dp_size
        def get_data_parallel_rank(self): return self._dp_rank

    mpu = SimpleMPU(tp_group, dp_size, tp_size, tp_rank, dp_rank)
    engine, _, _, _ = deepspeed.initialize(model=model, optimizer=optimizer, config=ds_config, mpu=mpu)

    # Create dataloader (shard by DP rank)
    dataloader = create_dataloader(
        model_name=config["model_name"],
        batch_size=config["batch_size"],
        seq_length=config["seq_length"],
        dp_rank=dp_rank,
        dp_size=dp_size,
    )

    # Training loop
    engine.train()
    loss_history = []
    step = 0

    for batch in dataloader:
        if step >= config["num_steps"]:
            break

        batch = {k: v.to(device) for k, v in batch.items()}

        # DeepSpeed handles bf16 internally via ds_config, no manual autocast needed
        outputs = engine(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
            use_cache=False,
        )
        loss = outputs.loss

        engine.backward(loss)
        engine.step()

        # All-reduce loss across all ranks for consistent logging
        loss_tensor = loss.detach().clone()
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
        loss_history.append(loss_tensor.item())

        if world_rank == 0 and step % 10 == 0:
            print(f"[DeepSpeed] Step {step}: loss={loss_tensor.item():.4f}")

        step += 1

    # Save loss history from rank 0
    if world_rank == 0:
        output_path = os.path.join(config["output_dir"], "loss_deepspeed.json")
        with open(output_path, "w") as f:
            json.dump({"implementation": "deepspeed_autotp", "loss_history": loss_history}, f)
        print(f"Saved loss history to {output_path}")

    dist.barrier()
    ray.train.report({"final_loss": loss_history[-1] if loss_history else 0})


def clip_grad_norm_dtensor(parameters, max_norm: float) -> torch.Tensor:
    """Gradient clipping for DTensor parameters."""
    from torch.distributed.tensor import DTensor

    parameters = list(parameters)
    max_norm = float(max_norm)

    # Compute total grad norm (local computation for DTensor)
    total_norm_sq = torch.tensor(0.0, device="cuda")
    for p in parameters:
        if p.grad is not None:
            if isinstance(p.grad, DTensor):
                # For DTensor, use local tensor and squared sum
                local_grad = p.grad.to_local()
                total_norm_sq += local_grad.float().pow(2).sum()
            else:
                total_norm_sq += p.grad.float().pow(2).sum()

    # All-reduce across all ranks to get true global norm
    dist.all_reduce(total_norm_sq, op=dist.ReduceOp.SUM)
    total_norm = total_norm_sq.sqrt()

    clip_coef = max_norm / (total_norm + 1e-6)
    clip_coef_clamped = torch.clamp(clip_coef, max=1.0)

    # Apply clipping locally
    for p in parameters:
        if p.grad is not None:
            if isinstance(p.grad, DTensor):
                # Modify local tensor directly
                p.grad._local_tensor.mul_(clip_coef_clamped)
            else:
                p.grad.mul_(clip_coef_clamped)

    return total_norm


def train_fsdp(config: Dict[str, Any]) -> None:
    """Training loop for FSDP+DTensor."""
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
    from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel, parallelize_module

    ctx = ray.train.get_context()
    world_rank = ctx.get_world_rank()
    world_size = ctx.get_world_size()
    device = ray.train.torch.get_device()

    tp_size = config["tp_size"]
    dp_size = config["dp_size"]

    # Calculate TP/DP ranks
    tp_rank = world_rank % tp_size
    dp_rank = world_rank // tp_size

    # Create 2D device mesh
    device_mesh = init_device_mesh("cuda", (dp_size, tp_size), mesh_dim_names=("dp", "tp"))
    tp_mesh = device_mesh["tp"]
    dp_mesh = device_mesh["dp"]

    # Create model
    model = create_model(
        model_name=config["model_name"],
        device=device,
        dtype=torch.bfloat16,
        num_layers=config["num_layers"],
        attn_impl="sdpa",
    )

    # Load shared initial weights
    load_initial_weights(model, config["weights_path"])

    # Get transformer layers and apply TP
    layers = get_transformer_layers(model)
    tp_mapping = {
        "self_attn.q_proj": ColwiseParallel(),
        "self_attn.k_proj": ColwiseParallel(),
        "self_attn.v_proj": ColwiseParallel(),
        "self_attn.o_proj": RowwiseParallel(),
        "mlp.gate_proj": ColwiseParallel(),
        "mlp.up_proj": ColwiseParallel(),
        "mlp.down_proj": RowwiseParallel(),
    }
    for layer in layers:
        parallelize_module(layer, tp_mesh, tp_mapping)

    # Apply FSDP2 (fully_shard) for DP gradient synchronization
    mp_policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16)
    if dp_size > 1:
        for layer in layers:
            fully_shard(layer, mesh=dp_mesh, mp_policy=mp_policy)
        fully_shard(model, mesh=dp_mesh, mp_policy=mp_policy)
        if world_rank == 0:
            print("[FSDP] Applied FSDP2 for DP gradient sync")

    # Create optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], foreach=False)

    # Create dataloader (shard by DP rank)
    dataloader = create_dataloader(
        model_name=config["model_name"],
        batch_size=config["batch_size"],
        seq_length=config["seq_length"],
        dp_rank=dp_rank,
        dp_size=dp_size,
    )

    # Training loop
    model.train()
    loss_history = []
    step = 0

    for batch in dataloader:
        if step >= config["num_steps"]:
            break

        batch = {k: v.to(device) for k, v in batch.items()}
        optimizer.zero_grad()

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
                use_cache=False,
            )
            loss = outputs.loss

        loss.backward()

        # Gradient clipping (same as DDP/DeepSpeed)
        # Use custom function for DTensor compatibility
        clip_grad_norm_dtensor(model.parameters(), max_norm=1.0)

        optimizer.step()

        # All-reduce loss across all ranks for consistent logging
        loss_tensor = loss.detach().clone()
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
        loss_history.append(loss_tensor.item())

        if world_rank == 0 and step % 10 == 0:
            print(f"[FSDP] Step {step}: loss={loss_tensor.item():.4f}")

        step += 1

    # Save loss history from rank 0
    if world_rank == 0:
        output_path = os.path.join(config["output_dir"], "loss_fsdp.json")
        with open(output_path, "w") as f:
            json.dump({"implementation": "fsdp_dtensor", "loss_history": loss_history}, f)
        print(f"Saved loss history to {output_path}")

    dist.barrier()
    ray.train.report({"final_loss": loss_history[-1] if loss_history else 0})


def main():
    parser = argparse.ArgumentParser(description="Run 100 training steps")
    parser.add_argument("--impl", type=str, required=True, choices=["ddp", "deepspeed", "fsdp"])
    parser.add_argument("--output_dir", type=str, default="/tmp/loss_curves")
    parser.add_argument("--weights_path", type=str, default="/tmp/verify_tp_dp2/initial_weights.pt")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--seq_length", type=int, default=1024)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--num_steps", type=int, default=100)

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Determine workers and train function based on implementation
    if args.impl == "ddp":
        num_workers = 2  # DP=2
        train_fn = train_ddp
        tp_size, dp_size = 1, 2
    elif args.impl == "deepspeed":
        num_workers = 4  # TP=2, DP=2
        train_fn = train_deepspeed
        tp_size, dp_size = 2, 2
    else:  # fsdp
        num_workers = 4  # TP=2, DP=2
        train_fn = train_fsdp
        tp_size, dp_size = 2, 2

    print(f"Running {args.impl} with {num_workers} workers (TP={tp_size}, DP={dp_size})")

    config = {
        "model_name": args.model_name,
        "num_layers": args.num_layers,
        "batch_size": args.batch_size,
        "seq_length": args.seq_length,
        "learning_rate": args.learning_rate,
        "num_steps": args.num_steps,
        "output_dir": args.output_dir,
        "weights_path": args.weights_path,
        "tp_size": tp_size,
        "dp_size": dp_size,
    }

    trainer = TorchTrainer(
        train_loop_per_worker=train_fn,
        scaling_config=ScalingConfig(num_workers=num_workers, use_gpu=True),
        train_loop_config=config,
        run_config=RunConfig(storage_path="/tmp", name=f"train_100steps_{args.impl}"),
    )

    result = trainer.fit()
    print(f"Training finished: {result}")


if __name__ == "__main__":
    main()
