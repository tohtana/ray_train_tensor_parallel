"""
Vocabulary-parallel layers for tensor parallelism.

Based on Megatron-LM implementation, adapted from
multimodal-training/python/tensor_parallel/embedding.py
and multimodal-training/python/tensor_parallel/cross_entropy.py
"""

from typing import Optional, Sequence, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F


def vocab_range_from_global_vocab_size(
    global_vocab_size: int, rank: int, world_size: int
) -> Tuple[int, int]:
    """
    Calculate the vocabulary range [start, end) for a given rank.

    Args:
        global_vocab_size: Total vocabulary size
        rank: Rank of the current process
        world_size: Total number of processes in the TP group

    Returns:
        Tuple of (vocab_start_index, vocab_end_index)
    """
    assert global_vocab_size % world_size == 0, (
        f"Vocabulary size ({global_vocab_size}) must be divisible by "
        f"tensor parallel world size ({world_size})"
    )

    vocab_size_per_partition = global_vocab_size // world_size
    vocab_start_index = rank * vocab_size_per_partition
    vocab_end_index = vocab_start_index + vocab_size_per_partition

    return vocab_start_index, vocab_end_index


class VocabParallelEmbedding(nn.Module):
    """
    Embedding layer parallelized in the vocabulary dimension.

    The vocabulary is partitioned across tensor parallel ranks, with each
    rank holding a contiguous slice of the vocabulary.

    Args:
        num_embeddings: Total vocabulary size (will be partitioned)
        embedding_dim: Size of embedding vectors
        padding_idx: Index for padding token (default: None)
        tp_group: Tensor parallel process group
        tp_mesh: Device mesh for tensor parallelism (optional)
        dtype: Data type for the embedding weights
        device: Device to place the embedding weights
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: Optional[int] = None,
        tp_group: Optional[dist.ProcessGroup] = None,
        tp_mesh=None,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ):
        super().__init__()

        # Store configuration
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.tp_group = tp_group
        self.tp_mesh = tp_mesh

        # Get tensor parallel info
        if tp_group is not None and dist.is_initialized():
            self.tp_world_size = dist.get_world_size(tp_group)
            self.tp_rank = dist.get_rank(tp_group)
        else:
            self.tp_world_size = 1
            self.tp_rank = 0

        # Calculate vocabulary partition for this rank
        self.vocab_start_index, self.vocab_end_index = vocab_range_from_global_vocab_size(
            self.num_embeddings,
            self.tp_rank,
            self.tp_world_size,
        )
        self.num_embeddings_per_partition = self.vocab_end_index - self.vocab_start_index

        # Create the embedding weight parameter (partitioned vocabulary)
        self.weight = nn.Parameter(
            torch.empty(
                self.num_embeddings_per_partition,
                self.embedding_dim,
                dtype=dtype,
                device=device,
            )
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with vocabulary parallelism.

        Args:
            input_ids: Input token IDs [batch_size, seq_length]

        Returns:
            Embeddings [batch_size, seq_length, embedding_dim]
        """
        if self.tp_world_size > 1:
            # Build mask for tokens outside this rank's vocabulary range
            input_mask = (input_ids < self.vocab_start_index) | (
                input_ids >= self.vocab_end_index
            )

            # Adjust input IDs to be relative to this partition's vocabulary
            masked_input = input_ids.clone() - self.vocab_start_index

            # Set out-of-range indices to 0 to avoid index errors
            masked_input[input_mask] = 0
        else:
            masked_input = input_ids
            input_mask = None

        # Perform embedding lookup using the local weight partition
        output_parallel = F.embedding(
            masked_input,
            self.weight,
            padding_idx=None,  # We handle padding through the mask
        )

        # Zero out embeddings for tokens outside this rank's vocabulary range
        if self.tp_world_size > 1:
            output_parallel = output_parallel.clone()
            output_parallel[input_mask] = 0.0

            # All-reduce across tensor parallel group to sum contributions
            # Only the rank that owns a token ID will have non-zero embeddings
            dist.all_reduce(output_parallel, op=dist.ReduceOp.SUM, group=self.tp_group)

        return output_parallel

    def extra_repr(self) -> str:
        """String representation of this module."""
        s = f"{self.num_embeddings}, {self.embedding_dim}"
        if self.padding_idx is not None:
            s += f", padding_idx={self.padding_idx}"
        s += f", vocab_range=[{self.vocab_start_index}, {self.vocab_end_index})"
        s += f", tp_rank={self.tp_rank}/{self.tp_world_size}"
        return s


def create_vocab_parallel_embedding(
    original_embedding: nn.Embedding,
    tp_group: Optional[dist.ProcessGroup] = None,
    tp_mesh=None,
    device: Optional[torch.device] = None,
) -> VocabParallelEmbedding:
    """
    Create a VocabParallelEmbedding from an existing nn.Embedding,
    copying and sharding the weights appropriately.

    Args:
        original_embedding: The original nn.Embedding to convert
        tp_group: Tensor parallel process group
        tp_mesh: Device mesh for tensor parallelism (optional)
        device: Device to place the new embedding

    Returns:
        VocabParallelEmbedding with weights copied from original
    """
    # Create new vocab parallel embedding
    vocab_parallel_emb = VocabParallelEmbedding(
        num_embeddings=original_embedding.num_embeddings,
        embedding_dim=original_embedding.embedding_dim,
        padding_idx=original_embedding.padding_idx,
        tp_group=tp_group,
        tp_mesh=tp_mesh,
        dtype=original_embedding.weight.dtype,
        device=device or original_embedding.weight.device,
    )

    # Copy the appropriate partition of weights
    with torch.no_grad():
        # Extract the vocabulary partition for this rank
        start_idx = vocab_parallel_emb.vocab_start_index
        end_idx = vocab_parallel_emb.vocab_end_index

        # Copy weights to the new embedding
        original_weight = original_embedding.weight.data
        if device is not None and original_weight.device != device:
            original_weight = original_weight.to(device)

        vocab_parallel_emb.weight.data.copy_(original_weight[start_idx:end_idx])

    return vocab_parallel_emb


class VocabUtility:
    """
    Split the vocabulary into `world_size` chunks and return the first
    and last index of the vocabulary belonging to the `rank`
    partition: Note that indices in [first, last)
    """

    @staticmethod
    def vocab_range_from_per_partition_vocab_size(
        per_partition_vocab_size: int, rank: int, world_size: int
    ) -> Sequence[int]:
        """Vocab range from per partition vocab size."""
        index_f = rank * per_partition_vocab_size
        index_l = index_f + per_partition_vocab_size
        return index_f, index_l

    @staticmethod
    def vocab_range_from_global_vocab_size(
        global_vocab_size: int, rank: int, world_size: int
    ) -> Sequence[int]:
        """Vocab range from global vocab size."""
        assert global_vocab_size % world_size == 0
        per_partition_vocab_size = global_vocab_size // world_size
        return VocabUtility.vocab_range_from_per_partition_vocab_size(
            per_partition_vocab_size, rank, world_size
        )


class VocabParallelCrossEntropy:
    """
    Computes the Cross Entropy Loss splitting the Vocab size across tensor parallel
    ranks. This implementation is used in both fused and unfused cross entropy implementations.
    """

    @staticmethod
    def calculate_logits_max(
        vocab_parallel_logits: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calculates logits_max."""
        vocab_parallel_logits = vocab_parallel_logits.float()
        # Maximum value along vocab dimension across all GPUs.
        logits_max = torch.max(vocab_parallel_logits, dim=-1)[0]
        return vocab_parallel_logits, logits_max

    @staticmethod
    def calculate_predicted_logits(
        vocab_parallel_logits: torch.Tensor,
        target: torch.Tensor,
        logits_max: torch.Tensor,
        vocab_start_index: int,
        vocab_end_index: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Calculates predicted logits."""
        # In-place subtraction reduces memory pressure.
        vocab_parallel_logits -= logits_max.unsqueeze(dim=-1)

        # Create a mask of valid vocab ids (1 means it needs to be masked).
        target_mask = (target < vocab_start_index) | (target >= vocab_end_index)
        masked_target = target.clone() - vocab_start_index
        masked_target[target_mask] = 0

        # Get predicted-logits = logits[target].
        partition_vocab_size = vocab_parallel_logits.size()[-1]
        logits_2d = vocab_parallel_logits.view(-1, partition_vocab_size)
        masked_target_1d = masked_target.view(-1)
        arange_1d = torch.arange(start=0, end=logits_2d.size()[0], device=logits_2d.device)
        predicted_logits_1d = logits_2d[arange_1d, masked_target_1d]
        predicted_logits_1d = predicted_logits_1d.clone().contiguous()
        predicted_logits = predicted_logits_1d.view_as(target)
        predicted_logits[target_mask] = 0.0

        exp_logits = vocab_parallel_logits
        torch.exp(vocab_parallel_logits, out=exp_logits)
        sum_exp_logits = exp_logits.sum(dim=-1)

        return target_mask, masked_target_1d, predicted_logits, sum_exp_logits, exp_logits

    @staticmethod
    def calculate_cross_entropy_loss(
        exp_logits: torch.Tensor, predicted_logits: torch.Tensor, sum_exp_logits: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calculates cross entropy loss."""
        # Loss = log(sum(exp(logits))) - predicted-logit.
        loss = torch.log(sum_exp_logits) - predicted_logits

        # Normalize logits to get softmax probabilities
        exp_logits.div_(sum_exp_logits.unsqueeze(dim=-1))

        return exp_logits, loss

    @staticmethod
    def prepare_gradient_calculation_operands(
        softmax: torch.Tensor, target_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Prepare gradient calculation operands."""
        # All the inputs have softmax as their gradient.
        grad_input = softmax
        partition_vocab_size = softmax.size()[-1]
        grad_2d = grad_input.view(-1, partition_vocab_size)

        # Add the gradient from matching classes.
        arange_1d = torch.arange(start=0, end=grad_2d.size()[0], device=grad_2d.device)
        softmax_update = 1.0 - target_mask.view(-1).float()

        return grad_2d, arange_1d, softmax_update, grad_input

    @staticmethod
    def calculate_gradients(
        grad_2d: torch.Tensor,
        arange_1d: torch.Tensor,
        masked_target_1d: torch.Tensor,
        softmax_update: torch.Tensor,
        grad_input: torch.Tensor,
        grad_output: torch.Tensor,
    ) -> torch.Tensor:
        """Calculates gradients."""
        grad_2d[arange_1d, masked_target_1d] -= softmax_update

        # Finally elementwise multiplication with the output gradients.
        grad_input.mul_(grad_output.unsqueeze(dim=-1))

        return grad_input


class _VocabParallelCrossEntropy(torch.autograd.Function):
    """Autograd function for vocab-parallel cross entropy."""

    @staticmethod
    def forward(ctx, vocab_parallel_logits, target, tp_group, tp_rank, tp_world_size):
        """Vocab parallel cross entropy forward function."""
        vocab_parallel_logits, logits_max = VocabParallelCrossEntropy.calculate_logits_max(
            vocab_parallel_logits
        )
        dist.all_reduce(logits_max, op=dist.ReduceOp.MAX, group=tp_group)

        # Get the partition's vocab indices
        get_vocab_range = VocabUtility.vocab_range_from_per_partition_vocab_size
        partition_vocab_size = vocab_parallel_logits.size()[-1]
        vocab_start_index, vocab_end_index = get_vocab_range(
            partition_vocab_size, tp_rank, tp_world_size
        )

        (target_mask, masked_target_1d, predicted_logits, sum_exp_logits, exp_logits) = (
            VocabParallelCrossEntropy.calculate_predicted_logits(
                vocab_parallel_logits, target, logits_max, vocab_start_index, vocab_end_index
            )
        )

        # All reduce to get the chunks from other GPUs.
        dist.all_reduce(predicted_logits, op=dist.ReduceOp.SUM, group=tp_group)
        dist.all_reduce(sum_exp_logits, op=dist.ReduceOp.SUM, group=tp_group)

        exp_logits, loss = VocabParallelCrossEntropy.calculate_cross_entropy_loss(
            exp_logits, predicted_logits, sum_exp_logits
        )

        # Store softmax, target-mask and masked-target for backward pass.
        ctx.save_for_backward(exp_logits, target_mask, masked_target_1d)

        return loss

    @staticmethod
    def backward(ctx, grad_output):
        """Vocab parallel cross entropy backward function."""
        # Retrieve tensors from the forward path.
        softmax, target_mask, masked_target_1d = ctx.saved_tensors

        (grad_2d, arange_1d, softmax_update, grad_input) = (
            VocabParallelCrossEntropy.prepare_gradient_calculation_operands(softmax, target_mask)
        )

        grad_input = VocabParallelCrossEntropy.calculate_gradients(
            grad_2d, arange_1d, masked_target_1d, softmax_update, grad_input, grad_output
        )

        return grad_input, None, None, None, None


def vocab_parallel_cross_entropy(
    vocab_parallel_logits: torch.Tensor,
    target: torch.Tensor,
    tp_group: dist.ProcessGroup,
    tp_rank: int,
    tp_world_size: int,
) -> torch.Tensor:
    """
    Performs cross entropy loss when logits are split across tensor parallel ranks.

    Args:
        vocab_parallel_logits: logits split across tensor parallel ranks
            dimension is [batch_size, sequence_length, vocab_size/num_parallel_ranks]
        target: correct vocab ids of dimension [batch_size, sequence_length]
        tp_group: tensor parallel process group
        tp_rank: rank within tensor parallel group
        tp_world_size: world size of tensor parallel group

    Returns:
        Per-token loss tensor of shape [batch_size, sequence_length]
    """
    return _VocabParallelCrossEntropy.apply(
        vocab_parallel_logits, target, tp_group, tp_rank, tp_world_size
    )


def vocab_parallel_causal_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    tp_group: dist.ProcessGroup,
    tp_rank: int,
    tp_world_size: int,
    ignore_index: int = -100,
) -> torch.Tensor:
    """
    Performs causal (next-token prediction) cross entropy loss when logits are
    split across tensor parallel ranks.

    This function handles:
    - Causal shifting: predicting token t+1 from tokens 0:t
    - Masking of ignore_index tokens (e.g., padding tokens)
    - Proper averaging over valid (non-ignored) tokens

    Args:
        logits: logits split across tensor parallel ranks
            dimension is [batch_size, sequence_length, vocab_size/num_parallel_ranks]
        labels: correct vocab ids of dimension [batch_size, sequence_length]
        tp_group: tensor parallel process group
        tp_rank: rank within tensor parallel group
        tp_world_size: world size of tensor parallel group
        ignore_index: token id to ignore in loss computation (default: -100)

    Returns:
        Scalar loss averaged over all valid (non-ignored) tokens
    """
    # Causal shifting: predict token at position t+1 from tokens at positions 0:t
    shift_logits = logits[:, :-1, :].contiguous()  # [B, S-1, V_local]
    shift_labels = labels[:, 1:].contiguous()  # [B, S-1]

    # Compute per-token loss
    loss_per_token = _VocabParallelCrossEntropy.apply(
        shift_logits, shift_labels, tp_group, tp_rank, tp_world_size
    )  # [B, S-1]

    # Mask out ignore_index tokens
    valid_mask = shift_labels.ne(ignore_index)  # [B, S-1]

    # Compute masked loss: zero out ignored positions
    masked_loss = loss_per_token * valid_mask.float()

    # Sum over all tokens and count valid tokens
    loss_sum = masked_loss.sum()
    num_valid = valid_mask.float().sum()

    # Average over valid tokens
    # NOTE: No all_reduce needed here because loss_per_token is already synchronized
    return loss_sum / num_valid.clamp(min=1.0)
