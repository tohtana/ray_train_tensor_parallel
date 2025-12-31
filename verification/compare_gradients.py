"""
Compare gradients across DDP, DeepSpeed, and FSDP implementations.

This script loads saved gradient files and compares them across implementations.

Usage:
    python compare_gradients.py --input_dir /tmp/verify_tp
"""

import argparse
import os
from typing import Dict, List, Tuple

import torch


def load_gradients(path: str) -> Dict[str, torch.Tensor]:
    """Load gradients from a file."""
    if not os.path.exists(path):
        print(f"Warning: {path} not found")
        return {}
    gradients = torch.load(path, map_location="cpu", weights_only=True)
    print(f"Loaded {len(gradients)} gradients from {path}")
    return gradients


def compare_gradient_norms(
    grads1: Dict[str, torch.Tensor],
    grads2: Dict[str, torch.Tensor],
    name1: str,
    name2: str,
) -> List[Tuple[str, float, float, float, float]]:
    """
    Compare gradient norms between two implementations.

    Returns:
        List of (param_name, norm1, norm2, abs_diff, rel_diff)
    """
    results = []
    common_keys = set(grads1.keys()) & set(grads2.keys())

    for key in sorted(common_keys):
        g1 = grads1[key]
        g2 = grads2[key]

        norm1 = g1.float().norm().item()
        norm2 = g2.float().norm().item()

        abs_diff = abs(norm1 - norm2)
        rel_diff = abs_diff / max(abs(norm1), abs(norm2), 1e-10)

        results.append((key, norm1, norm2, abs_diff, rel_diff))

    return results


def compare_gradient_values(
    grads1: Dict[str, torch.Tensor],
    grads2: Dict[str, torch.Tensor],
    name1: str,
    name2: str,
) -> List[Tuple[str, float, float]]:
    """
    Compare actual gradient values between two implementations.

    Returns:
        List of (param_name, max_abs_diff, mean_abs_diff)
    """
    results = []
    common_keys = set(grads1.keys()) & set(grads2.keys())

    for key in sorted(common_keys):
        g1 = grads1[key].float()
        g2 = grads2[key].float()

        # Handle shape mismatches (TP sharding)
        if g1.shape != g2.shape:
            results.append((key, float('nan'), float('nan')))
            continue

        diff = (g1 - g2).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()

        results.append((key, max_diff, mean_diff))

    return results


def main():
    parser = argparse.ArgumentParser(description="Compare gradients")
    parser.add_argument("--input_dir", type=str, default="/tmp/verify_tp")
    args = parser.parse_args()

    # Load gradients from all implementations
    ddp_grads = load_gradients(os.path.join(args.input_dir, "gradients_ddp_rank0.pt"))
    ds_grads = load_gradients(os.path.join(args.input_dir, "gradients_deepspeed_rank0.pt"))
    fsdp_grads = load_gradients(os.path.join(args.input_dir, "gradients_fsdp_rank0.pt"))

    print("\n" + "=" * 80)
    print("GRADIENT COMPARISON")
    print("=" * 80)

    # Compare DDP vs DeepSpeed
    if ddp_grads and ds_grads:
        print("\n" + "-" * 80)
        print("DDP vs DeepSpeed AutoTP - Gradient Norm Comparison")
        print("-" * 80)
        print(f"{'Parameter':<50} {'DDP Norm':>12} {'DS Norm':>12} {'Abs Diff':>12} {'Rel Diff':>12}")
        print("-" * 98)

        norm_results = compare_gradient_norms(ddp_grads, ds_grads, "DDP", "DeepSpeed")
        for param, n1, n2, abs_diff, rel_diff in norm_results[:20]:  # Show first 20
            short_name = param[-45:] if len(param) > 45 else param
            print(f"{short_name:<50} {n1:>12.6f} {n2:>12.6f} {abs_diff:>12.6f} {rel_diff:>12.2e}")

        if len(norm_results) > 20:
            print(f"... and {len(norm_results) - 20} more parameters")

        # Summary statistics
        abs_diffs = [r[3] for r in norm_results]
        rel_diffs = [r[4] for r in norm_results if r[4] < float('inf')]
        print(f"\nMax abs norm diff: {max(abs_diffs):.6f}")
        print(f"Mean abs norm diff: {sum(abs_diffs)/len(abs_diffs):.6f}")
        print(f"Max rel norm diff: {max(rel_diffs):.2e}")

        # Value comparison
        print("\n" + "-" * 80)
        print("DDP vs DeepSpeed AutoTP - Gradient Value Comparison")
        print("-" * 80)
        value_results = compare_gradient_values(ddp_grads, ds_grads, "DDP", "DeepSpeed")
        print(f"{'Parameter':<50} {'Max Abs Diff':>15} {'Mean Abs Diff':>15}")
        print("-" * 80)
        for param, max_diff, mean_diff in value_results[:20]:
            short_name = param[-45:] if len(param) > 45 else param
            if max_diff == max_diff:  # not NaN
                print(f"{short_name:<50} {max_diff:>15.6e} {mean_diff:>15.6e}")
            else:
                print(f"{short_name:<50} {'Shape mismatch':>15} {'(TP sharded)':>15}")

    # Compare DDP vs FSDP
    if ddp_grads and fsdp_grads:
        print("\n" + "-" * 80)
        print("DDP vs FSDP+DTensor - Gradient Norm Comparison")
        print("-" * 80)
        print(f"{'Parameter':<50} {'DDP Norm':>12} {'FSDP Norm':>12} {'Abs Diff':>12} {'Rel Diff':>12}")
        print("-" * 98)

        norm_results = compare_gradient_norms(ddp_grads, fsdp_grads, "DDP", "FSDP")
        for param, n1, n2, abs_diff, rel_diff in norm_results[:20]:
            short_name = param[-45:] if len(param) > 45 else param
            print(f"{short_name:<50} {n1:>12.6f} {n2:>12.6f} {abs_diff:>12.6f} {rel_diff:>12.2e}")

        if len(norm_results) > 20:
            print(f"... and {len(norm_results) - 20} more parameters")

        abs_diffs = [r[3] for r in norm_results]
        rel_diffs = [r[4] for r in norm_results if r[4] < float('inf')]
        print(f"\nMax abs norm diff: {max(abs_diffs):.6f}")
        print(f"Mean abs norm diff: {sum(abs_diffs)/len(abs_diffs):.6f}")
        print(f"Max rel norm diff: {max(rel_diffs):.2e}")

    # Compare DeepSpeed vs FSDP
    if ds_grads and fsdp_grads:
        print("\n" + "-" * 80)
        print("DeepSpeed AutoTP vs FSDP+DTensor - Gradient Norm Comparison")
        print("-" * 80)
        print(f"{'Parameter':<50} {'DS Norm':>12} {'FSDP Norm':>12} {'Abs Diff':>12} {'Rel Diff':>12}")
        print("-" * 98)

        norm_results = compare_gradient_norms(ds_grads, fsdp_grads, "DeepSpeed", "FSDP")
        for param, n1, n2, abs_diff, rel_diff in norm_results[:20]:
            short_name = param[-45:] if len(param) > 45 else param
            print(f"{short_name:<50} {n1:>12.6f} {n2:>12.6f} {abs_diff:>12.6f} {rel_diff:>12.2e}")

        if len(norm_results) > 20:
            print(f"... and {len(norm_results) - 20} more parameters")

        abs_diffs = [r[3] for r in norm_results]
        rel_diffs = [r[4] for r in norm_results if r[4] < float('inf')]
        print(f"\nMax abs norm diff: {max(abs_diffs):.6f}")
        print(f"Mean abs norm diff: {sum(abs_diffs)/len(abs_diffs):.6f}")
        print(f"Max rel norm diff: {max(rel_diffs):.2e}")

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    if ddp_grads:
        print(f"DDP gradients: {len(ddp_grads)} parameters")
    if ds_grads:
        print(f"DeepSpeed gradients: {len(ds_grads)} parameters")
    if fsdp_grads:
        print(f"FSDP gradients: {len(fsdp_grads)} parameters")


if __name__ == "__main__":
    main()
