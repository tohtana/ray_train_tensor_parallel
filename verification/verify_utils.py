"""Utilities for verification of TP/DDP implementations."""

import json
import os
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.distributed as dist


@dataclass
class LayerDump:
    """Container for a single layer's forward/backward data."""
    name: str
    output_norm: float = 0.0
    output_shape: Tuple[int, ...] = ()
    grad_input_norm: float = 0.0
    grad_input_shape: Tuple[int, ...] = ()
    # For detailed comparison
    output_sample: Optional[torch.Tensor] = None  # First few elements
    grad_input_sample: Optional[torch.Tensor] = None


@dataclass
class VerificationDump:
    """Container for all verification data from a run."""
    implementation: str
    tp_size: int
    dp_size: int
    rank: int
    loss: float = 0.0
    layers: Dict[str, LayerDump] = field(default_factory=dict)
    # Parameter gradient norms
    param_grad_norms: Dict[str, float] = field(default_factory=dict)


class HookManager:
    """Manager for forward/backward hooks to capture layer outputs and gradients."""

    def __init__(self, dump: VerificationDump):
        self.dump = dump
        self.hooks = []
        self._forward_outputs = {}
        self._backward_grads = {}

    def register_hooks(self, model: nn.Module, layer_patterns: Optional[List[str]] = None):
        """
        Register forward and backward hooks on model layers.

        Args:
            model: The model to hook
            layer_patterns: Optional list of layer name patterns to hook.
                           If None, hooks all layers.
        """
        for name, module in model.named_modules():
            # Skip the top-level model
            if name == "":
                continue

            # Filter by patterns if provided
            if layer_patterns is not None:
                if not any(pattern in name for pattern in layer_patterns):
                    continue

            # Register forward hook
            forward_hook = module.register_forward_hook(
                self._make_forward_hook(name)
            )
            self.hooks.append(forward_hook)

            # Register backward hook (full backward hook for gradients)
            backward_hook = module.register_full_backward_hook(
                self._make_backward_hook(name)
            )
            self.hooks.append(backward_hook)

    def _make_forward_hook(self, layer_name: str):
        """Create a forward hook for capturing output."""
        def hook(module, input, output):
            # Handle tuple outputs
            if isinstance(output, tuple):
                out = output[0]
            else:
                out = output

            # Handle DTensor
            if hasattr(out, 'full_tensor'):
                out = out.full_tensor()
            elif hasattr(out, 'to_local'):
                out = out.to_local()

            if isinstance(out, torch.Tensor):
                self._forward_outputs[layer_name] = out.detach().clone()
        return hook

    def _make_backward_hook(self, layer_name: str):
        """Create a backward hook for capturing gradients."""
        def hook(module, grad_input, grad_output):
            # grad_output is the gradient w.r.t. the output
            if grad_output is not None and len(grad_output) > 0:
                grad = grad_output[0]
                if grad is not None:
                    # Handle DTensor
                    if hasattr(grad, 'full_tensor'):
                        grad = grad.full_tensor()
                    elif hasattr(grad, 'to_local'):
                        grad = grad.to_local()

                    if isinstance(grad, torch.Tensor):
                        self._backward_grads[layer_name] = grad.detach().clone()
        return hook

    def finalize(self):
        """Finalize the dump with collected data."""
        for name, output in self._forward_outputs.items():
            if name not in self.dump.layers:
                self.dump.layers[name] = LayerDump(name=name)

            self.dump.layers[name].output_norm = output.float().norm().item()
            self.dump.layers[name].output_shape = tuple(output.shape)
            # Save a small sample for detailed comparison
            flat = output.flatten()[:100].float().cpu()
            self.dump.layers[name].output_sample = flat

        for name, grad in self._backward_grads.items():
            if name not in self.dump.layers:
                self.dump.layers[name] = LayerDump(name=name)

            self.dump.layers[name].grad_input_norm = grad.float().norm().item()
            self.dump.layers[name].grad_input_shape = tuple(grad.shape)
            # Save a small sample for detailed comparison
            flat = grad.flatten()[:100].float().cpu()
            self.dump.layers[name].grad_input_sample = flat

    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()


def create_fixed_input(
    tokenizer,
    batch_size: int,
    seq_length: int,
    seed: int = 42,
) -> Dict[str, torch.Tensor]:
    """
    Create a fixed input batch for reproducible testing.

    Args:
        tokenizer: The tokenizer to use
        batch_size: Batch size
        seq_length: Sequence length
        seed: Random seed

    Returns:
        Dict with input_ids, attention_mask, labels
    """
    torch.manual_seed(seed)

    vocab_size = tokenizer.vocab_size

    # Generate random token IDs (avoiding special tokens at the extremes)
    input_ids = torch.randint(100, vocab_size - 100, (batch_size, seq_length))

    # Full attention mask (no padding)
    attention_mask = torch.ones(batch_size, seq_length, dtype=torch.long)

    # Labels are same as input_ids for causal LM
    labels = input_ids.clone()

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def save_initial_weights(model: nn.Module, path: str):
    """
    Save model weights for sharing across implementations.

    Args:
        model: The model
        path: Path to save weights
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"Saved initial weights to {path}")


def load_initial_weights(model: nn.Module, path: str, strict: bool = True):
    """
    Load shared initial weights into model.

    Args:
        model: The model
        path: Path to load weights from
        strict: Whether to strictly enforce matching keys
    """
    state_dict = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict, strict=strict)
    print(f"Loaded initial weights from {path}")


def save_fixed_input(batch: Dict[str, torch.Tensor], path: str):
    """Save fixed input batch."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(batch, path)
    print(f"Saved fixed input to {path}")


def load_fixed_input(path: str) -> Dict[str, torch.Tensor]:
    """Load fixed input batch."""
    batch = torch.load(path, map_location="cpu", weights_only=True)
    print(f"Loaded fixed input from {path}")
    return batch


def save_verification_dump(dump: VerificationDump, path: str):
    """
    Save verification dump to file.

    Args:
        dump: The verification dump
        path: Path to save
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # Convert to serializable format
    data = {
        "implementation": dump.implementation,
        "tp_size": dump.tp_size,
        "dp_size": dump.dp_size,
        "rank": dump.rank,
        "loss": dump.loss,
        "param_grad_norms": dump.param_grad_norms,
        "layers": {}
    }

    for name, layer in dump.layers.items():
        layer_data = {
            "name": layer.name,
            "output_norm": layer.output_norm,
            "output_shape": list(layer.output_shape),
            "grad_input_norm": layer.grad_input_norm,
            "grad_input_shape": list(layer.grad_input_shape),
        }
        if layer.output_sample is not None:
            layer_data["output_sample"] = layer.output_sample.tolist()
        if layer.grad_input_sample is not None:
            layer_data["grad_input_sample"] = layer.grad_input_sample.tolist()
        data["layers"][name] = layer_data

    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved verification dump to {path}")


def load_verification_dump(path: str) -> VerificationDump:
    """Load verification dump from file."""
    with open(path, "r") as f:
        data = json.load(f)

    dump = VerificationDump(
        implementation=data["implementation"],
        tp_size=data["tp_size"],
        dp_size=data["dp_size"],
        rank=data["rank"],
        loss=data["loss"],
        param_grad_norms=data.get("param_grad_norms", {}),
    )

    for name, layer_data in data["layers"].items():
        layer = LayerDump(
            name=layer_data["name"],
            output_norm=layer_data["output_norm"],
            output_shape=tuple(layer_data["output_shape"]),
            grad_input_norm=layer_data["grad_input_norm"],
            grad_input_shape=tuple(layer_data["grad_input_shape"]),
        )
        if "output_sample" in layer_data:
            layer.output_sample = torch.tensor(layer_data["output_sample"])
        if "grad_input_sample" in layer_data:
            layer.grad_input_sample = torch.tensor(layer_data["grad_input_sample"])
        dump.layers[name] = layer

    return dump


def collect_param_grad_norms(model: nn.Module, use_deepspeed: bool = False) -> Dict[str, float]:
    """
    Collect gradient norms for all parameters.

    Args:
        model: The model to collect gradients from
        use_deepspeed: If True, use DeepSpeed's safe_get_full_grad
    """
    grad_norms = {}

    if use_deepspeed:
        try:
            from deepspeed.utils import safe_get_full_grad
            for name, param in model.named_parameters():
                grad = safe_get_full_grad(param)
                if grad is not None:
                    grad_norms[name] = grad.float().norm().item()
        except ImportError:
            print("Warning: deepspeed.utils.safe_get_full_grad not available")
            # Fall back to standard method
            use_deepspeed = False

    if not use_deepspeed:
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad = param.grad
                # Handle DTensor
                if hasattr(grad, 'full_tensor'):
                    grad = grad.full_tensor()
                elif hasattr(grad, 'to_local'):
                    grad = grad.to_local()
                grad_norms[name] = grad.float().norm().item()

    return grad_norms


def collect_param_gradients(model: nn.Module, use_deepspeed: bool = False) -> Dict[str, torch.Tensor]:
    """
    Collect full gradients for all parameters.

    Args:
        model: The model to collect gradients from
        use_deepspeed: If True, use DeepSpeed's safe_get_full_grad

    Returns:
        Dict mapping parameter name to gradient tensor (detached, on CPU)
    """
    gradients = {}

    if use_deepspeed:
        try:
            from deepspeed.utils import safe_get_full_grad
            for name, param in model.named_parameters():
                grad = safe_get_full_grad(param)
                if grad is not None:
                    gradients[name] = grad.detach().float().cpu()
        except ImportError:
            print("Warning: deepspeed.utils.safe_get_full_grad not available")
            use_deepspeed = False

    if not use_deepspeed:
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad = param.grad
                # Handle DTensor
                if hasattr(grad, 'full_tensor'):
                    grad = grad.full_tensor()
                elif hasattr(grad, 'to_local'):
                    grad = grad.to_local()
                gradients[name] = grad.detach().float().cpu()

    return gradients


def sync_and_broadcast_weights(model: nn.Module, src_rank: int = 0):
    """
    Broadcast model weights from src_rank to all other ranks.

    Args:
        model: The model
        src_rank: Source rank for broadcast
    """
    if not dist.is_initialized():
        return

    for param in model.parameters():
        dist.broadcast(param.data, src=src_rank)

    print(f"Broadcasted weights from rank {src_rank}")
