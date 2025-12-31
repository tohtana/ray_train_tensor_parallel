"""
Setup script for DP=2 verification.

Creates initial weights and fixed input batch with global_batch_size = batch_size * dp_size.

Usage:
    python setup_verification_dp2.py --output_dir /tmp/verify_tp_dp2
"""

import argparse
import os

import torch
from transformers import AutoTokenizer

from model_builder import create_model
from verify_utils import (
    create_fixed_input,
    save_fixed_input,
    save_initial_weights,
)


def main():
    parser = argparse.ArgumentParser(description="Setup verification for DP=2")
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen2.5-0.5B",
        help="HuggingFace model name",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/tmp/verify_tp_dp2",
        help="Output directory",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="Per-worker batch size",
    )
    parser.add_argument(
        "--dp_size",
        type=int,
        default=2,
        help="Data parallel size",
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
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16

    print(f"Setting up DP=2 verification")
    print(f"Model: {args.model_name}")
    print(f"Batch size per worker: {args.batch_size}")
    print(f"DP size: {args.dp_size}")
    print(f"Global batch size: {args.batch_size * args.dp_size}")
    print(f"Sequence length: {args.seq_length}")
    print(f"Number of layers: {args.num_layers}")

    # Set random seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # Create model and save initial weights
    print("\nCreating model...")
    model = create_model(
        model_name=args.model_name,
        device=device,
        dtype=dtype,
        num_layers=args.num_layers,
        attn_impl="sdpa",
    )

    weights_path = os.path.join(args.output_dir, "initial_weights.pt")
    save_initial_weights(model, weights_path)

    # Create tokenizer
    print("\nCreating fixed input...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Create fixed input with global batch size
    global_batch_size = args.batch_size * args.dp_size
    fixed_input = create_fixed_input(
        tokenizer,
        batch_size=global_batch_size,
        seq_length=args.seq_length,
        seed=args.seed,
    )

    print(f"Fixed input shape: {fixed_input['input_ids'].shape}")

    input_path = os.path.join(args.output_dir, "fixed_input.pt")
    save_fixed_input(fixed_input, input_path)

    print(f"\nSetup complete. Files saved to {args.output_dir}")
    print(f"  - Initial weights: {weights_path}")
    print(f"  - Fixed input: {input_path}")


if __name__ == "__main__":
    main()
