# Copyright (c) 2025, The Board of Trustees of the Leland Stanford Junior University.

# This source code is licensed under the Apache license found in the
# LICENSE file in the root directory of this source tree.

"""
Trace a single iteration of Megatron-LM training.
This module sets up the model and runs a single forward and backward pass
to capture communication patterns.
"""

from megatron.core.enums import ModelType
from megatron.training.training import pretrain


def run_one_iteration(args):
    """Run a single iteration of training to trace communication patterns."""
    # Import locally to avoid errors when this module is imported
    from pretrain_gpt import (
        forward_step,
        model_provider,
        train_valid_test_datasets_provider,
    )

    # Set mock data flags to avoid loading real datasets
    args.mock_data = True
    args.consume_next_microbatch = True

    # Set flag to use meta tensors
    args.use_cpu_initialization = True
    args.init_model_with_meta_device = True

    # Run only one iteration for tracing
    args.train_iters = 1
    args.eval_interval = 1000  # Avoid evaluation
    args.save_interval = 1000  # Avoid checkpointing

    # Run pretrain with minimal operations
    pretrain(
        train_valid_test_datasets_provider,
        model_provider,
        ModelType.encoder_or_decoder,
        forward_step,
        args_defaults={"tokenizer_type": "GPT2BPETokenizer"},
        parsed_args=args,
    )
