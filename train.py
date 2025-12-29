"""
Ray Train + DeepSpeed AutoTP Tensor Parallelism Training.

This script demonstrates training with DeepSpeed AutoTP tensor parallelism
using Ray Train's TorchTrainer for distributed execution and checkpointing.

Example usage:
    # 8 GPUs: 4-way tensor parallelism, 2-way data parallelism
    python train.py \
        --model_name Qwen/Qwen2-7B \
        --tp_size 4 \
        --dp_size 2 \
        --num_workers 8 \
        --dataset_name wikitext \
        --batch_size 2 \
        --seq_length 2048 \
        --num_epochs 3

    # 4 GPUs: 4-way tensor parallelism only
    python train.py \
        --model_name Qwen/Qwen2-7B \
        --tp_size 4 \
        --dp_size 1 \
        --num_workers 4 \
        --dataset_name wikitext \
        --num_epochs 3
"""

import argparse
import json
import logging
import os
import tempfile
import uuid
from typing import Any, Dict

os.environ["RAY_TRAIN_V2_ENABLED"] = "1"

import deepspeed
import torch

import ray
import ray.train
import ray.train.torch
from ray.train import Checkpoint, RunConfig, ScalingConfig
from ray.train.torch import TorchTrainer

from autotp_strategy import RayAutoTPStrategy
from data import create_tp_aware_dataloader

logger = logging.getLogger(__name__)


def log_rank0(message: str) -> None:
    """Log message only from rank 0."""
    if ray.train.get_context().get_world_rank() == 0:
        logger.info(message)


def save_checkpoint(
    strategy: RayAutoTPStrategy,
    epoch: int,
    step: int,
    metrics: Dict[str, Any],
) -> None:
    """
    Save checkpoint and report to Ray Train.

    All workers save their checkpoint shards, then report to Ray Train.
    Ray Train aggregates checkpoints from all workers properly.

    Args:
        strategy: The AutoTP strategy containing the DeepSpeed engine
        epoch: Current epoch number
        step: Current step number
        metrics: Training metrics to report
    """
    world_rank = ray.train.get_context().get_world_rank()

    with tempfile.TemporaryDirectory() as tmp_dir:
        checkpoint_dir = os.path.join(tmp_dir, "checkpoint")
        os.makedirs(checkpoint_dir, exist_ok=True)

        # DeepSpeed saves each rank's shard
        strategy.save_checkpoint(checkpoint_dir, tag="model")

        # Save metadata (from world rank 0 only)
        if world_rank == 0:
            metadata = {"epoch": epoch, "step": step}
            with open(os.path.join(checkpoint_dir, "metadata.json"), "w") as f:
                json.dump(metadata, f)

        # All workers must call report() with their checkpoint
        checkpoint = Checkpoint.from_directory(tmp_dir)
        ray.train.report(metrics, checkpoint=checkpoint)

    if world_rank == 0:
        experiment_name = ray.train.get_context().get_experiment_name()
        log_rank0(f"Checkpoint saved for experiment {experiment_name}. Metrics: {metrics}")


def load_checkpoint(
    strategy: RayAutoTPStrategy,
    checkpoint: ray.train.Checkpoint,
) -> Dict[str, Any]:
    """
    Load checkpoint for resuming training.

    Each rank loads its corresponding checkpoint shard.

    Args:
        strategy: The AutoTP strategy containing the DeepSpeed engine
        checkpoint: Ray Train checkpoint to load

    Returns:
        Metadata dict with epoch and step info
    """
    metadata = {"epoch": 0, "step": 0}

    try:
        with checkpoint.as_directory() as checkpoint_dir:
            log_rank0(f"Loading checkpoint from {checkpoint_dir}")
            ckpt_dir = os.path.join(checkpoint_dir, "checkpoint")
            if not os.path.isdir(ckpt_dir):
                ckpt_dir = checkpoint_dir

            # Load DeepSpeed checkpoint (each rank loads its shard)
            strategy.load_checkpoint(ckpt_dir, tag="model")

            # Read metadata
            metadata_file = os.path.join(ckpt_dir, "metadata.json")
            if os.path.isfile(metadata_file):
                with open(metadata_file, "r") as f:
                    metadata = json.load(f)

            # Synchronize across all workers
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                torch.distributed.barrier()

        log_rank0(f"Successfully loaded checkpoint. Epoch: {metadata.get('epoch', 0)}")

    except Exception as e:
        logger.error(f"Failed to load checkpoint: {e}")
        raise RuntimeError(f"Checkpoint loading failed: {e}") from e

    return metadata


def train_loop_per_worker(config: Dict[str, Any]) -> None:
    """
    Main training loop executed by each Ray Train worker.

    Args:
        config: Training configuration dict
    """
    # Get Ray Train context
    ctx = ray.train.get_context()
    world_rank = ctx.get_world_rank()
    world_size = ctx.get_world_size()
    device = ray.train.torch.get_device()

    log_rank0(f"Worker started: world_rank={world_rank}, world_size={world_size}")

    # Initialize DeepSpeed distributed (will detect and use existing process group from Ray Train)
    deepspeed.init_distributed()

    # Create and setup the AutoTP strategy
    strategy = RayAutoTPStrategy(
        tp_size=config["tp_size"],
        dp_size=config["dp_size"],
    )

    strategy.setup(
        model_name=config["model_name"],
        device=device,
        dtype=torch.bfloat16,
        config={
            "batch_size": config["batch_size"],
            "learning_rate": config["learning_rate"],
            "weight_decay": config.get("weight_decay", 0.01),
            "max_grad_norm": config.get("max_grad_norm", 1.0),
            "zero_stage": config["zero_stage"],
            "gradient_accumulation_steps": config.get("gradient_accumulation_steps", 1),
            "num_layers": config.get("num_layers", 0),
            "attn_impl": config.get("attn_impl", "sdpa"),
            "activation_checkpointing": config.get("activation_checkpointing", False),
        },
    )

    # Create TP-aware dataloader
    # Uses dp_rank/dp_size for sharding, not world_rank/world_size
    dataloader = create_tp_aware_dataloader(
        model_name=config["model_name"],
        dataset_name=config["dataset_name"],
        seq_length=config["seq_length"],
        batch_size=config["batch_size"],
        dp_rank=strategy.dp_rank,
        dp_size=strategy.dp_size,
        seed=config.get("seed", 42),
        dataset_percentage=config.get("dataset_percentage", 10.0),
    )

    steps_per_epoch = len(dataloader)
    log_rank0(f"Dataloader created: {steps_per_epoch} steps per epoch")

    # Load checkpoint if resuming
    checkpoint = ray.train.get_checkpoint()
    start_epoch = 0
    if checkpoint:
        metadata = load_checkpoint(strategy, checkpoint)
        start_epoch = metadata.get("epoch", 0) + 1
        log_rank0(f"Resuming training from epoch {start_epoch}")

    # Set model to training mode
    strategy.train()

    # Training loop
    for epoch in range(start_epoch, config["num_epochs"]):
        # Set sampler epoch for different shuffling each epoch
        dataloader.sampler.set_epoch(epoch)

        running_loss = 0.0
        num_batches = 0

        for step, batch in enumerate(dataloader):
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}

            # Forward pass
            loss = strategy.forward(batch)

            # Log progress
            if world_rank == 0 and step % config.get("log_interval", 10) == 0:
                log_rank0(
                    f"Epoch: {epoch} Step: {step + 1}/{steps_per_epoch} Loss: {loss.item():.4f}"
                )

            # Backward pass
            strategy.backward(loss)

            # Optimizer step
            strategy.optimizer_step()

            running_loss += loss.item()
            num_batches += 1

            # Debug mode: stop early for testing
            if config.get("debug_steps", 0) > 0 and step + 1 >= config["debug_steps"]:
                log_rank0(f"Debug steps finished. Stopping epoch {epoch}.")
                break

        # Calculate average loss for epoch
        avg_loss = running_loss / num_batches if num_batches > 0 else 0.0

        # Save checkpoint at end of epoch
        save_checkpoint(
            strategy=strategy,
            epoch=epoch,
            step=step,
            metrics={"loss": avg_loss, "epoch": epoch},
        )

        log_rank0(f"Epoch {epoch} completed. Average loss: {avg_loss:.4f}")


def get_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Ray Train + DeepSpeed AutoTP Tensor Parallelism Training"
    )

    # Model configuration
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen2-7B",
        help="HuggingFace model name or path",
    )
    parser.add_argument(
        "--num_layers",
        type=int,
        default=0,
        help="Override number of layers (0 = use model default)",
    )
    parser.add_argument(
        "--attn_impl",
        type=str,
        default="sdpa",
        choices=["sdpa", "flash_attention_2", "eager"],
        help="Attention implementation",
    )

    # Parallelism configuration
    parser.add_argument(
        "--tp_size",
        type=int,
        required=True,
        help="Tensor parallel degree",
    )
    parser.add_argument(
        "--dp_size",
        type=int,
        default=1,
        help="Data parallel degree",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        required=True,
        help="Total number of workers (must equal tp_size * dp_size)",
    )
    parser.add_argument(
        "--zero_stage",
        type=int,
        default=1,
        choices=[0, 1, 2],
        help="DeepSpeed ZeRO stage (0-2, ZeRO-3 not supported with AutoTP)",
    )

    # Dataset configuration
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="wikitext",
        help="HuggingFace dataset name",
    )
    parser.add_argument(
        "--dataset_percentage",
        type=float,
        default=10.0,
        help="Percentage of dataset to use (0-100)",
    )

    # Training configuration
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Per-GPU micro batch size",
    )
    parser.add_argument(
        "--seq_length",
        type=int,
        default=2048,
        help="Maximum sequence length",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=3,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="Learning rate",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
        help="Weight decay",
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.0,
        help="Gradient clipping norm",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Gradient accumulation steps",
    )
    parser.add_argument(
        "--activation_checkpointing",
        action="store_true",
        help="Enable activation/gradient checkpointing",
    )

    # Checkpointing configuration
    parser.add_argument(
        "--storage_path",
        type=str,
        default="/mnt/cluster_storage",
        help="Storage path for checkpoints",
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default=None,
        help="Experiment name (auto-generated if not provided)",
    )
    parser.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help="Experiment name to resume from",
    )

    # Logging and debugging
    parser.add_argument(
        "--log_interval",
        type=int,
        default=10,
        help="Logging interval (steps)",
    )
    parser.add_argument(
        "--debug_steps",
        type=int,
        default=0,
        help="Stop after this many steps per epoch (0 = run full epoch)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = get_args()

    # Validate parallelism configuration
    if args.tp_size * args.dp_size != args.num_workers:
        raise ValueError(
            f"tp_size ({args.tp_size}) * dp_size ({args.dp_size}) "
            f"must equal num_workers ({args.num_workers})"
        )

    print(f"Configuration: {args}")

    # Configure Ray Train
    scaling_config = ScalingConfig(
        num_workers=args.num_workers,
        use_gpu=True,
    )

    train_loop_config = {
        # Model
        "model_name": args.model_name,
        "num_layers": args.num_layers,
        "attn_impl": args.attn_impl,
        # Parallelism
        "tp_size": args.tp_size,
        "dp_size": args.dp_size,
        "zero_stage": args.zero_stage,
        # Dataset
        "dataset_name": args.dataset_name,
        "dataset_percentage": args.dataset_percentage,
        # Training
        "batch_size": args.batch_size,
        "seq_length": args.seq_length,
        "num_epochs": args.num_epochs,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "max_grad_norm": args.max_grad_norm,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "activation_checkpointing": args.activation_checkpointing,
        # Logging/debugging
        "log_interval": args.log_interval,
        "debug_steps": args.debug_steps,
        "seed": args.seed,
    }

    # Generate experiment name
    name = args.experiment_name
    if name is None:
        if args.resume_from is not None:
            name = args.resume_from
        else:
            name = f"autotp_tp{args.tp_size}_dp{args.dp_size}_{uuid.uuid4().hex[:8]}"

    print(f"Experiment name: {name}")

    run_config = RunConfig(
        storage_path=args.storage_path,
        name=name,
    )

    # Create and run trainer
    trainer = TorchTrainer(
        train_loop_per_worker=train_loop_per_worker,
        scaling_config=scaling_config,
        train_loop_config=train_loop_config,
        run_config=run_config,
    )

    result = trainer.fit()
    print(f"Training finished. Result: {result}")


if __name__ == "__main__":
    main()
