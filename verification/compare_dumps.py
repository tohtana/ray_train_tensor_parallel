"""
Compare verification dumps from DDP, DeepSpeed AutoTP, and FSDP+DTensor.

This script loads all dumps and computes:
1. Loss comparison
2. Output norms per layer
3. Gradient norms per layer
4. Max absolute differences between implementations

Usage:
    python compare_dumps.py --input_dir /tmp/verify_tp
"""

import argparse
import glob
import os
from typing import Dict, List, Optional

import torch

from verify_utils import VerificationDump, load_verification_dump


def find_dumps(input_dir: str) -> Dict[str, List[VerificationDump]]:
    """Find and load all verification dumps in the input directory."""
    dumps = {
        "ddp": [],
        "deepspeed_autotp": [],
        "fsdp_dtensor": [],
    }

    # Load DDP dump
    ddp_path = os.path.join(input_dir, "dump_ddp.json")
    if os.path.exists(ddp_path):
        dumps["ddp"].append(load_verification_dump(ddp_path))

    # Load DeepSpeed dumps (multiple ranks)
    for path in sorted(glob.glob(os.path.join(input_dir, "dump_deepspeed_rank*.json"))):
        dumps["deepspeed_autotp"].append(load_verification_dump(path))

    # Load FSDP dumps (multiple ranks)
    for path in sorted(glob.glob(os.path.join(input_dir, "dump_fsdp_rank*.json"))):
        dumps["fsdp_dtensor"].append(load_verification_dump(path))

    return dumps


def print_loss_comparison(dumps: Dict[str, List[VerificationDump]]):
    """Print loss comparison across implementations."""
    print("\n" + "=" * 80)
    print("LOSS COMPARISON")
    print("=" * 80)

    losses = {}
    for impl, impl_dumps in dumps.items():
        if impl_dumps:
            # For TP implementations, all ranks should have the same loss
            losses[impl] = impl_dumps[0].loss

    if not losses:
        print("No dumps found!")
        return

    # Print losses
    print(f"\n{'Implementation':<25} {'Loss':>15}")
    print("-" * 42)
    for impl, loss in losses.items():
        print(f"{impl:<25} {loss:>15.8f}")

    # Print differences
    if "ddp" in losses:
        print(f"\n{'Implementation':<25} {'Diff from DDP':>15} {'Rel. Diff':>15}")
        print("-" * 57)
        ddp_loss = losses["ddp"]
        for impl, loss in losses.items():
            diff = loss - ddp_loss
            rel_diff = diff / ddp_loss if ddp_loss != 0 else 0
            print(f"{impl:<25} {diff:>15.8f} {rel_diff:>14.2e}")


def print_layer_output_norms(dumps: Dict[str, List[VerificationDump]]):
    """Print layer output norm comparison."""
    print("\n" + "=" * 80)
    print("LAYER OUTPUT NORMS (from rank 0)")
    print("=" * 80)

    # Get rank 0 dumps for each implementation
    rank0_dumps = {}
    for impl, impl_dumps in dumps.items():
        for dump in impl_dumps:
            if dump.rank == 0:
                rank0_dumps[impl] = dump
                break

    if not rank0_dumps:
        print("No rank 0 dumps found!")
        return

    # Collect all layer names
    all_layers = set()
    for dump in rank0_dumps.values():
        all_layers.update(dump.layers.keys())

    # Sort layers (try to sort by layer index if possible)
    def layer_sort_key(name):
        parts = name.split(".")
        # Extract layer index if present
        for i, part in enumerate(parts):
            if part == "layers" and i + 1 < len(parts):
                try:
                    return (int(parts[i + 1]), name)
                except ValueError:
                    pass
        return (999, name)

    sorted_layers = sorted(all_layers, key=layer_sort_key)

    # Print header
    impls = list(rank0_dumps.keys())
    header = f"{'Layer':<50}"
    for impl in impls:
        header += f" {impl[:15]:>15}"
    print(f"\n{header}")
    print("-" * (50 + 16 * len(impls)))

    # Print output norms
    for layer_name in sorted_layers:
        # Skip layers that are too deep (for readability)
        if layer_name.count(".") > 4:
            continue

        row = f"{layer_name:<50}"
        norms = []
        for impl in impls:
            dump = rank0_dumps[impl]
            if layer_name in dump.layers and dump.layers[layer_name].output_norm > 0:
                norm = dump.layers[layer_name].output_norm
                norms.append(norm)
                row += f" {norm:>15.4f}"
            else:
                norms.append(None)
                row += f" {'N/A':>15}"

        # Only print if at least one implementation has data
        if any(n is not None for n in norms):
            print(row)


def print_layer_gradient_norms(dumps: Dict[str, List[VerificationDump]]):
    """Print layer gradient norm comparison."""
    print("\n" + "=" * 80)
    print("LAYER GRADIENT NORMS (backward pass, from rank 0)")
    print("=" * 80)

    # Get rank 0 dumps for each implementation
    rank0_dumps = {}
    for impl, impl_dumps in dumps.items():
        for dump in impl_dumps:
            if dump.rank == 0:
                rank0_dumps[impl] = dump
                break

    if not rank0_dumps:
        print("No rank 0 dumps found!")
        return

    # Collect all layer names
    all_layers = set()
    for dump in rank0_dumps.values():
        all_layers.update(dump.layers.keys())

    # Sort layers
    def layer_sort_key(name):
        parts = name.split(".")
        for i, part in enumerate(parts):
            if part == "layers" and i + 1 < len(parts):
                try:
                    return (int(parts[i + 1]), name)
                except ValueError:
                    pass
        return (999, name)

    sorted_layers = sorted(all_layers, key=layer_sort_key)

    # Print header
    impls = list(rank0_dumps.keys())
    header = f"{'Layer':<50}"
    for impl in impls:
        header += f" {impl[:15]:>15}"
    print(f"\n{header}")
    print("-" * (50 + 16 * len(impls)))

    # Print gradient norms
    for layer_name in sorted_layers:
        if layer_name.count(".") > 4:
            continue

        row = f"{layer_name:<50}"
        norms = []
        for impl in impls:
            dump = rank0_dumps[impl]
            if layer_name in dump.layers and dump.layers[layer_name].grad_input_norm > 0:
                norm = dump.layers[layer_name].grad_input_norm
                norms.append(norm)
                row += f" {norm:>15.4f}"
            else:
                norms.append(None)
                row += f" {'N/A':>15}"

        if any(n is not None for n in norms):
            print(row)


def print_max_abs_diff(dumps: Dict[str, List[VerificationDump]]):
    """Print maximum absolute differences between implementations."""
    print("\n" + "=" * 80)
    print("MAX ABSOLUTE DIFFERENCES (output samples, rank 0)")
    print("=" * 80)

    # Get rank 0 dumps for each implementation
    rank0_dumps = {}
    for impl, impl_dumps in dumps.items():
        for dump in impl_dumps:
            if dump.rank == 0:
                rank0_dumps[impl] = dump
                break

    if len(rank0_dumps) < 2:
        print("Need at least 2 implementations to compare!")
        return

    # Collect all layer names that have output samples
    all_layers = set()
    for dump in rank0_dumps.values():
        for name, layer in dump.layers.items():
            if layer.output_sample is not None:
                all_layers.add(name)

    # Sort layers
    def layer_sort_key(name):
        parts = name.split(".")
        for i, part in enumerate(parts):
            if part == "layers" and i + 1 < len(parts):
                try:
                    return (int(parts[i + 1]), name)
                except ValueError:
                    pass
        return (999, name)

    sorted_layers = sorted(all_layers, key=layer_sort_key)

    # Print comparison pairs
    impls = list(rank0_dumps.keys())
    for i, impl1 in enumerate(impls):
        for impl2 in impls[i + 1 :]:
            print(f"\n{impl1} vs {impl2}:")
            print(f"{'Layer':<50} {'Max Abs Diff':>15} {'Rel Diff':>15}")
            print("-" * 82)

            for layer_name in sorted_layers:
                if layer_name.count(".") > 4:
                    continue

                dump1 = rank0_dumps[impl1]
                dump2 = rank0_dumps[impl2]

                if layer_name in dump1.layers and layer_name in dump2.layers:
                    sample1 = dump1.layers[layer_name].output_sample
                    sample2 = dump2.layers[layer_name].output_sample

                    if sample1 is not None and sample2 is not None:
                        # Align sizes
                        min_len = min(len(sample1), len(sample2))
                        s1 = sample1[:min_len]
                        s2 = sample2[:min_len]

                        max_diff = (s1 - s2).abs().max().item()
                        norm1 = s1.norm().item()
                        rel_diff = max_diff / norm1 if norm1 > 0 else 0

                        print(f"{layer_name:<50} {max_diff:>15.6e} {rel_diff:>15.6e}")


def print_param_grad_comparison(dumps: Dict[str, List[VerificationDump]]):
    """Print parameter gradient norm comparison."""
    print("\n" + "=" * 80)
    print("PARAMETER GRADIENT NORMS (first 20 parameters, rank 0)")
    print("=" * 80)

    # Get rank 0 dumps for each implementation
    rank0_dumps = {}
    for impl, impl_dumps in dumps.items():
        for dump in impl_dumps:
            if dump.rank == 0:
                rank0_dumps[impl] = dump
                break

    if not rank0_dumps:
        print("No rank 0 dumps found!")
        return

    # Collect all parameter names
    all_params = set()
    for dump in rank0_dumps.values():
        all_params.update(dump.param_grad_norms.keys())

    sorted_params = sorted(all_params)[:20]

    # Print header
    impls = list(rank0_dumps.keys())
    header = f"{'Parameter':<60}"
    for impl in impls:
        header += f" {impl[:12]:>12}"
    print(f"\n{header}")
    print("-" * (60 + 13 * len(impls)))

    # Print gradient norms
    for param_name in sorted_params:
        # Shorten very long names
        short_name = param_name
        if len(short_name) > 58:
            short_name = "..." + short_name[-55:]

        row = f"{short_name:<60}"
        for impl in impls:
            dump = rank0_dumps[impl]
            if param_name in dump.param_grad_norms:
                norm = dump.param_grad_norms[param_name]
                row += f" {norm:>12.4f}"
            else:
                row += f" {'N/A':>12}"
        print(row)

    if len(all_params) > 20:
        print(f"\n... and {len(all_params) - 20} more parameters")


def print_summary(dumps: Dict[str, List[VerificationDump]]):
    """Print overall summary."""
    print("\n" + "=" * 80)
    print("VERIFICATION SUMMARY")
    print("=" * 80)

    # Get rank 0 dumps
    rank0_dumps = {}
    for impl, impl_dumps in dumps.items():
        for dump in impl_dumps:
            if dump.rank == 0:
                rank0_dumps[impl] = dump
                break

    if not rank0_dumps:
        print("No dumps found!")
        return

    print(f"\nImplementations verified: {list(rank0_dumps.keys())}")

    # Check loss consistency
    losses = {impl: dump.loss for impl, dump in rank0_dumps.items()}
    if "ddp" in losses:
        ddp_loss = losses["ddp"]
        print(f"\nDDP baseline loss: {ddp_loss:.8f}")

        max_diff = 0
        for impl, loss in losses.items():
            if impl != "ddp":
                diff = abs(loss - ddp_loss)
                rel_diff = diff / ddp_loss if ddp_loss != 0 else 0
                max_diff = max(max_diff, rel_diff)
                status = "✓ OK" if rel_diff < 0.01 else "✗ MISMATCH"
                print(f"  {impl}: loss={loss:.8f}, rel_diff={rel_diff:.2e} {status}")

        if max_diff < 0.01:
            print("\n✓ All implementations produce similar loss values!")
        else:
            print("\n✗ Warning: Some implementations have different loss values.")
            print("  This may indicate implementation differences or numerical precision issues.")
    else:
        print("\nNo DDP baseline found for comparison.")
        for impl, loss in losses.items():
            print(f"  {impl}: loss={loss:.8f}")


def main():
    parser = argparse.ArgumentParser(description="Compare verification dumps")
    parser.add_argument(
        "--input_dir",
        type=str,
        default="/tmp/verify_tp",
        help="Input directory with verification dumps",
    )

    args = parser.parse_args()

    print(f"Loading dumps from {args.input_dir}...")

    dumps = find_dumps(args.input_dir)

    # Print what we found
    print(f"\nFound dumps:")
    for impl, impl_dumps in dumps.items():
        if impl_dumps:
            ranks = [d.rank for d in impl_dumps]
            print(f"  {impl}: {len(impl_dumps)} dump(s) from rank(s) {ranks}")

    # Run comparisons
    print_loss_comparison(dumps)
    print_layer_output_norms(dumps)
    print_layer_gradient_norms(dumps)
    print_max_abs_diff(dumps)
    print_param_grad_comparison(dumps)
    print_summary(dumps)


if __name__ == "__main__":
    main()
