"""
DDP verification script - creates baseline weights/inputs and runs verification.

This script:
1. Creates and saves initial random weights (to be shared with TP implementations)
2. Creates and saves a fixed input batch
3. Runs a single forward/backward pass with hooks
4. Dumps outputs and gradients for comparison

Usage:
    python verify_ddp.py --output_dir /tmp/verify_tp
"""

import argparse
import os

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoTokenizer

from model_builder import create_model
from verify_utils import (
    HookManager,
    VerificationDump,
    collect_param_grad_norms,
    collect_param_gradients,
    create_fixed_input,
    save_fixed_input,
    save_initial_weights,
    save_verification_dump,
)


def run_ddp_verification(
    model_name: str,
    output_dir: str,
    batch_size: int = 2,
    seq_length: int = 1024,
    num_layers: int = 2,
    seed: int = 42,
):
    """
    Run DDP verification (single GPU, no distribution).

    Args:
        model_name: HuggingFace model name
        output_dir: Directory to save outputs
        batch_size: Batch size
        seq_length: Sequence length
        num_layers: Number of transformer layers (for faster testing)
        seed: Random seed
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16

    print(f"Running DDP verification on {device}")
    print(f"Model: {model_name}, batch_size={batch_size}, seq_length={seq_length}")

    # Set random seed for reproducibility
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Create model
    print("Creating model...")
    model = create_model(
        model_name=model_name,
        device=device,
        dtype=dtype,
        num_layers=num_layers,
        attn_impl="sdpa",
    )

    # Save initial weights (before any training)
    weights_path = os.path.join(output_dir, "initial_weights.pt")
    save_initial_weights(model, weights_path)

    # Create tokenizer and fixed input
    print("Creating fixed input...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    fixed_input = create_fixed_input(tokenizer, batch_size, seq_length, seed=seed)
    input_path = os.path.join(output_dir, "fixed_input.pt")
    save_fixed_input(fixed_input, input_path)

    # Move input to device
    batch = {k: v.to(device) for k, v in fixed_input.items()}

    # Create verification dump
    dump = VerificationDump(
        implementation="ddp",
        tp_size=1,
        dp_size=1,
        rank=0,
    )

    # Register hooks
    print("Registering hooks...")
    hook_manager = HookManager(dump)
    # Hook key layers: embeddings, attention, MLP, layer norms, lm_head
    layer_patterns = [
        "embed_tokens",
        "self_attn",
        "mlp",
        "input_layernorm",
        "post_attention_layernorm",
        "norm",
        "lm_head",
        "layers.0",  # First layer for detailed comparison
        "layers.1",  # Second layer if exists
    ]
    hook_manager.register_hooks(model, layer_patterns)

    # Forward pass with autocast
    print("Running forward pass...")
    model.train()
    with torch.autocast(device_type="cuda", dtype=dtype):
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
            use_cache=False,
        )
        loss = outputs.loss

    dump.loss = loss.item()
    print(f"Loss: {dump.loss:.6f}")

    # Backward pass
    print("Running backward pass...")
    loss.backward()

    # Collect parameter gradient norms
    dump.param_grad_norms = collect_param_grad_norms(model)

    # Also collect full gradients for detailed comparison
    gradients = collect_param_gradients(model)
    if gradients:
        grad_path = os.path.join(output_dir, "gradients_ddp_rank0.pt")
        torch.save(gradients, grad_path)
        print(f"Saved {len(gradients)} parameter gradients to {grad_path}")

    # Finalize hooks
    hook_manager.finalize()
    hook_manager.remove_hooks()

    # Save verification dump
    dump_path = os.path.join(output_dir, "dump_ddp.json")
    save_verification_dump(dump, dump_path)

    # Print summary
    print("\n" + "=" * 60)
    print("DDP Verification Summary")
    print("=" * 60)
    print(f"Loss: {dump.loss:.6f}")
    print(f"\nLayer output norms:")
    for name, layer in sorted(dump.layers.items()):
        if layer.output_norm > 0:
            print(f"  {name}: output_norm={layer.output_norm:.6f}, shape={layer.output_shape}")
    print(f"\nLayer gradient norms:")
    for name, layer in sorted(dump.layers.items()):
        if layer.grad_input_norm > 0:
            print(f"  {name}: grad_norm={layer.grad_input_norm:.6f}")
    print(f"\nParameter gradient norms (first 10):")
    for i, (name, norm) in enumerate(sorted(dump.param_grad_norms.items())):
        if i >= 10:
            print(f"  ... and {len(dump.param_grad_norms) - 10} more")
            break
        print(f"  {name}: {norm:.6f}")

    print(f"\nOutputs saved to {output_dir}")
    return dump


def main():
    parser = argparse.ArgumentParser(description="DDP Verification")
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen2.5-0.5B",
        help="HuggingFace model name",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/tmp/verify_tp",
        help="Output directory for dumps",
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
        help="Number of transformer layers (for faster testing)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    run_ddp_verification(
        model_name=args.model_name,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        seq_length=args.seq_length,
        num_layers=args.num_layers,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
