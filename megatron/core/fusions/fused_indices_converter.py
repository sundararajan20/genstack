# Copyright (c) 2025, The Board of Trustees of the Leland Stanford Junior University.
# All rights reserved.

"""
Mock implementation of fusion indices converter for simulation purposes.
"""


def fused_indices_to_multihot(
    indices,
    num_of_local_experts,
    num_of_tokens,
    capacity,
    num_bits_for_bucket_offset=16,
):
    """A mock implementation that returns a dummy tensor of the right shape."""
    import torch

    # Create and return a zero tensor of the expected shape
    multihot = torch.zeros((num_of_tokens, num_of_local_experts), dtype=torch.float32)
    return multihot
