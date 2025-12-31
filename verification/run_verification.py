#!/usr/bin/env python
"""
Master script to run full verification of DDP, DeepSpeed AutoTP, and FSDP+DTensor.

This script:
1. Runs DDP verification (creates baseline weights and fixed input)
2. Runs DeepSpeed AutoTP verification (TP=2, DP=1)
3. Runs FSDP+DTensor verification (TP=2, DP=1)
4. Compares all dumps and prints summary

Usage:
    python run_verification.py [--output_dir /tmp/verify_tp]
"""

import argparse
import os
import subprocess
import sys


def run_command(cmd: list, description: str) -> bool:
    """Run a command and return success status."""
    print("\n" + "=" * 80)
    print(f"RUNNING: {description}")
    print("=" * 80)
    print(f"Command: {' '.join(cmd)}\n")

    result = subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__)))
    if result.returncode != 0:
        print(f"\n✗ FAILED: {description}")
        return False

    print(f"\n✓ COMPLETED: {description}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Run full TP verification")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/tmp/verify_tp",
        help="Output directory for all verification artifacts",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen2.5-0.5B",
        help="HuggingFace model name",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="Batch size (same for all implementations)",
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
        "--tp_size",
        type=int,
        default=2,
        help="Tensor parallel size for TP implementations",
    )
    parser.add_argument(
        "--skip_ddp",
        action="store_true",
        help="Skip DDP verification (use existing weights/input)",
    )
    parser.add_argument(
        "--skip_deepspeed",
        action="store_true",
        help="Skip DeepSpeed verification",
    )
    parser.add_argument(
        "--skip_fsdp",
        action="store_true",
        help="Skip FSDP verification",
    )
    parser.add_argument(
        "--compare_only",
        action="store_true",
        help="Only run comparison (skip all training)",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("TENSOR PARALLELISM VERIFICATION")
    print("=" * 80)
    print(f"Model: {args.model_name}")
    print(f"Batch size: {args.batch_size}")
    print(f"Sequence length: {args.seq_length}")
    print(f"Number of layers: {args.num_layers}")
    print(f"TP size: {args.tp_size}")
    print(f"Output directory: {args.output_dir}")

    os.makedirs(args.output_dir, exist_ok=True)

    if not args.compare_only:
        # Step 1: Run DDP verification (creates baseline weights and input)
        if not args.skip_ddp:
            success = run_command(
                [
                    sys.executable,
                    "verify_ddp.py",
                    "--model_name", args.model_name,
                    "--output_dir", args.output_dir,
                    "--batch_size", str(args.batch_size),
                    "--seq_length", str(args.seq_length),
                    "--num_layers", str(args.num_layers),
                ],
                "DDP Verification (baseline)",
            )
            if not success:
                print("\nDDP verification failed! Aborting.")
                return 1
        else:
            print("\nSkipping DDP verification (using existing weights/input)")

        # Check that baseline files exist
        weights_path = os.path.join(args.output_dir, "initial_weights.pt")
        input_path = os.path.join(args.output_dir, "fixed_input.pt")
        if not os.path.exists(weights_path) or not os.path.exists(input_path):
            print(f"\n✗ ERROR: Baseline files not found!")
            print(f"  Expected: {weights_path}")
            print(f"  Expected: {input_path}")
            print("  Run without --skip_ddp to create them.")
            return 1

        # Step 2: Run DeepSpeed AutoTP verification
        if not args.skip_deepspeed:
            success = run_command(
                [
                    sys.executable,
                    "verify_deepspeed.py",
                    "--model_name", args.model_name,
                    "--input_dir", args.output_dir,
                    "--output_dir", args.output_dir,
                    "--tp_size", str(args.tp_size),
                    "--dp_size", "1",
                    "--batch_size", str(args.batch_size),
                    "--seq_length", str(args.seq_length),
                    "--num_layers", str(args.num_layers),
                ],
                f"DeepSpeed AutoTP Verification (TP={args.tp_size}, DP=1)",
            )
            if not success:
                print("\nDeepSpeed verification failed!")
        else:
            print("\nSkipping DeepSpeed verification")

        # Step 3: Run FSDP+DTensor verification
        if not args.skip_fsdp:
            success = run_command(
                [
                    sys.executable,
                    "verify_fsdp.py",
                    "--model_name", args.model_name,
                    "--input_dir", args.output_dir,
                    "--output_dir", args.output_dir,
                    "--tp_size", str(args.tp_size),
                    "--dp_size", "1",
                    "--batch_size", str(args.batch_size),
                    "--seq_length", str(args.seq_length),
                    "--num_layers", str(args.num_layers),
                ],
                f"FSDP+DTensor Verification (TP={args.tp_size}, DP=1)",
            )
            if not success:
                print("\nFSDP verification failed!")
        else:
            print("\nSkipping FSDP verification")

    # Step 4: Compare all dumps
    run_command(
        [
            sys.executable,
            "compare_dumps.py",
            "--input_dir", args.output_dir,
        ],
        "Comparing verification dumps",
    )

    print("\n" + "=" * 80)
    print("VERIFICATION COMPLETE")
    print("=" * 80)
    print(f"All artifacts saved to: {args.output_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
