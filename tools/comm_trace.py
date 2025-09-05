# Copyright (c) 2025, The Board of Trustees of the Leland Stanford Junior University.

# This source code is licensed under the Apache license found in the
# LICENSE file in the root directory of this source tree.

#!/usr/bin/env python3
"""
Command-line tool for tracing communication patterns in Megatron-LM.
This tool simulates distributed training with various parallelism strategies
and logs all communication operations without requiring actual GPU resources.
"""

import argparse
import importlib.util
import os
import sys
import types
from pathlib import Path

# Add Megatron-LM to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# Set up a custom importer to handle missing modules automatically
class _MockImporterClass:
    """
    A custom module importer that automatically mocks missing modules.
    This handles nested imports like 'triton.language' automatically.
    """

    def __init__(self):
        self._mocked_modules = {}

    def create_mock(self, fullname):
        """Create a mock module for the given name."""
        if fullname in self._mocked_modules:
            return self._mocked_modules[fullname]

        # Handle nested modules (e.g., triton.language)
        parent_name = fullname.rsplit(".", 1)[0] if "." in fullname else None
        if parent_name and parent_name not in sys.modules:
            # Create the parent module first
            self.create_mock(parent_name)

        # Create a mock module
        mock_module = types.ModuleType(fullname)
        mock_module.__path__ = []
        mock_module.__file__ = f"/mock/{fullname.replace('.', '/')}.py"
        mock_module.__loader__ = None
        mock_module.__spec__ = types.SimpleNamespace(
            name=fullname, loader=None, origin="mock", submodule_search_locations=[]
        )

        # Special handling for known modules
        if fullname == "triton.language":
            # Add common triton language attributes
            for attr_name in [
                "int32",
                "float32",
                "abs",
                "where",
                "sum",
                "atomic_add",
                "atomic_max",
                "constexpr",
            ]:
                setattr(mock_module, attr_name, lambda *args, **kwargs: None)
        elif fullname == "triton":
            # Add jit decorator
            def jit_decorator(*args, **kwargs):
                # Return a decorator that returns the original function
                def decorator(func):
                    return func

                # Handle both @triton.jit and @triton.jit()
                if len(args) == 1 and callable(args[0]):
                    return args[0]  # directly decorated function
                return decorator

            mock_module.jit = jit_decorator

            # Also add other common triton attributes
            for attr_name in ["cdiv", "launch", "Config"]:
                setattr(mock_module, attr_name, lambda *args, **kwargs: None)

        # Add the mock to the cache
        self._mocked_modules[fullname] = mock_module
        sys.modules[fullname] = mock_module

        # If this is a submodule, add it to the parent
        if parent_name and parent_name in sys.modules:
            child_name = fullname.split(".")[-1]
            setattr(sys.modules[parent_name], child_name, mock_module)

        return mock_module

    def find_module(self, fullname, path=None):
        """Find the module with the given name."""
        # Only mock specific modules
        if fullname.startswith(("triton", "apex", "transformer_engine", "flash_attn")):
            # Try to import the module normally first
            try:
                # Check if it's already in sys.modules
                if fullname in sys.modules:
                    return None

                # Check if the module exists on the system
                spec = importlib.util.find_spec(fullname)
                if spec is not None:
                    return None
            except (ImportError, AttributeError):
                # Module not found, we'll mock it
                pass

            return self
        return None

    def load_module(self, fullname):
        """Load the module with the given name."""
        if fullname in sys.modules:
            return sys.modules[fullname]

        # Create and return a mock module
        mock_module = self.create_mock(fullname)
        print(f"Mocked module '{fullname}'")
        return mock_module


# Create an instance of the _MockImporterClass
MockImporter = _MockImporterClass()

# Register the mock importer
sys.meta_path.insert(0, MockImporter)


# Mock required dependencies that might be missing
def mock_missing_modules():
    """Mock any required external modules that might be missing."""

    # List of modules to check and mock if missing
    modules_to_mock = [
        "triton",  # GPU optimization library
        "apex",  # NVIDIA's PyTorch extension
        "transformer_engine",  # NVIDIA's Transformer Engine
        "flash_attn",  # Flash attention implementation
    ]

    for module_name in modules_to_mock:
        try:
            # Only import if not already in sys.modules
            if module_name not in sys.modules:
                import_module = importlib.import_module(module_name)
        except (ImportError, ModuleNotFoundError):
            # Create a mock module
            MockImporter.create_mock(module_name)
            print(f"Module '{module_name}' not found, using mock implementation")


# Apply mocks before importing Megatron modules
mock_missing_modules()

from megatron.instrumentation.dist_spoofer import (
    GPT_MODEL_CONFIGS,
    run_megatron_distributed,
)


def parse_cli_args():
    parser = argparse.ArgumentParser(
        description="Communication tracer for Megatron-LM",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Parallelism dimensions
    parser.add_argument("--dp", type=int, default=1, help="Data parallel size")
    parser.add_argument("--tp", type=int, default=1, help="Tensor parallel size")
    parser.add_argument("--pp", type=int, default=1, help="Pipeline parallel size")

    # Model configuration
    parser.add_argument(
        "--model-size",
        type=str,
        default="small",
        choices=list(GPT_MODEL_CONFIGS.keys()),
        help="GPT model size to simulate",
    )

    # Training parameters
    # microbatch_size = global_batch_size / data_parallel_size / num_micro_batches
    parser.add_argument("--micro-batches", type=int, default=4, help="Number of micro-batches")
    parser.add_argument("--global-batch-size", type=int, default=1024, help="Global batch size")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed")

    # Output options
    parser.add_argument(
        "--output-dir",
        type=str,
        default="comm_logs",
        help="Directory to save communication logs",
    )

    return parser.parse_args()


def validate_args(args):
    """Validate command-line arguments."""
    # Validate parallelism dimensions are valid
    if args.dp < 1 or args.tp < 1 or args.pp < 1:
        raise ValueError("Parallelism dimensions must be at least 1")

    # Validate total world size
    world_size = args.dp * args.tp * args.pp
    if world_size > 128:
        print(f"Warning: Large world size ({world_size}) might cause performance issues")

    # Validate micro-batch configuration
    if args.global_batch_size % args.micro_batches != 0:
        raise ValueError(
            f"Global batch size ({args.global_batch_size}) must be divisible by number of micro-batches ({args.micro_batches})"
        )

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    return True


def main():
    """Main entry point for the communication tracer CLI."""
    args = parse_cli_args()

    if validate_args(args):
        print(f"Starting communication trace for GPT-{args.model_size}")
        print(f"Parallelism: DP={args.dp}, TP={args.tp}, PP={args.pp}")
        print(f"Total world size: {args.dp * args.tp * args.pp}")

        # Run the distributed simulation
        output_path = run_megatron_distributed(
            dp_size=args.dp,
            tp_size=args.tp,
            pp_size=args.pp,
            model_size=args.model_size,
            seed=args.seed,
            mock_data=True,
            output_dir=args.output_dir,
            num_micro_batches=args.micro_batches,
            global_batch_size=args.global_batch_size,
        )

        print(f"Communication trace completed successfully.")
        print(f"Logs saved to: {output_path}")


if __name__ == "__main__":
    main()
