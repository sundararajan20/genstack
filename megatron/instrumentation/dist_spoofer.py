# Copyright (c) 2025, The Board of Trustees of the Leland Stanford Junior University.

# This source code is licensed under the Apache license found in the
# LICENSE file in the root directory of this source tree.

"""
Process spoofing to simulate multiple ranks for distributed communication.
This module handles spawning multiple processes and simulating distributed
training environment for communication pattern analysis.
"""

import contextlib
import importlib.util
import json
import os
import sys
import time
import types
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from packaging.version import Version as PkgVersion


# Mock required dependencies that might be missing
def mock_missing_modules():
    """Mock any required external modules that might be missing."""

    # Create a proper mock module for missing dependencies
    class MockModule(types.ModuleType):
        """
        A more robust mock module that can be used as a substitute for missing dependencies.
        This implementation properly supports iteration and other module protocols.
        """

        def __init__(self, name):
            super().__init__(name)
            self._name = name
            self._submodules = {}

            # Add a spec to avoid errors with find_spec
            self.__spec__ = types.SimpleNamespace(
                name=name, loader=None, origin="mock", submodule_search_locations=[]
            )

        def __getattr__(self, key):
            # Create submodules on demand
            if key not in self._submodules:
                submodule_name = f"{self._name}.{key}"
                self._submodules[key] = MockModule(submodule_name)
            return self._submodules[key]

        def __call__(self, *args, **kwargs):
            # Return a dummy callable that can be used as a function or class
            class MockClass:
                def __init__(self, *args, **kwargs):
                    pass

                def forward(self, *args, **kwargs):
                    return args[0] if args else None

                @classmethod
                def __mro_entries__(cls, bases):
                    return (torch.nn.Module,)

            return MockClass

        def __dir__(self):
            # Support dir() calls
            return list(self._submodules.keys()) + list(super().__dir__())

        def __iter__(self):
            # Support iteration protocol
            return iter([])

    # List of modules to check and mock if missing
    modules_to_mock = [
        "triton",  # GPU optimization library
        "apex",  # NVIDIA's PyTorch extension
        "flash_attn",  # Flash attention implementation
    ]

    for module_name in modules_to_mock:
        try:
            # Only import if not already in sys.modules
            if module_name not in sys.modules:
                import_module = importlib.import_module(module_name)
        except (ImportError, ModuleNotFoundError):
            print(f"Module '{module_name}' not found, using mock implementation")
            mock_module = MockModule(module_name)
            sys.modules[module_name] = mock_module

    # Special handling for transformer_engine
    try:
        if "transformer_engine" not in sys.modules:
            import_module = importlib.import_module("transformer_engine")
    except (ImportError, ModuleNotFoundError):
        print("Module 'transformer_engine' not found, using specialized mock implementation")

        # Create transformer_engine mock
        te_mock = MockModule("transformer_engine")

        # Add version
        te_mock.__version__ = "999.999.999"

        # Create pytorch submodule
        pytorch_mock = MockModule("transformer_engine.pytorch")

        # Create proper class templates for inheritance
        class MockTELayer(torch.nn.Module):
            def __init__(self, *args, **kwargs):
                pass

            def forward(self, *args, **kwargs):
                # Just return the first input if available
                return args[0] if args else None

        # Add specific layer types
        pytorch_mock.Linear = type("Linear", (MockTELayer,), {})
        pytorch_mock.LayerNorm = type("LayerNorm", (MockTELayer,), {})
        pytorch_mock.Embedding = type("Embedding", (MockTELayer,), {})
        pytorch_mock.ColumnParallelLinear = type("ColumnParallelLinear", (MockTELayer,), {})
        pytorch_mock.RowParallelLinear = type("RowParallelLinear", (MockTELayer,), {})

        # Add to pytorch mock
        te_mock.pytorch = pytorch_mock

        # Register the mocked module
        sys.modules["transformer_engine"] = te_mock
        sys.modules["transformer_engine.pytorch"] = pytorch_mock


# Apply mocks before importing Megatron modules
mock_missing_modules()


# --- Early Patch for CudaRNGStatesTracker ---
class MockRNGTracker:
    """Mock RNG Tracker that satisfies required methods."""

    def __init__(self, *args, **kwargs):
        self._is_initialized = False  # Add internal state for checking
        pass

    def is_initialized(self):
        return self._is_initialized

    def add(self, name, seed):
        pass  # No-op for the mock

    def get_states(self):
        """Return empty dict to satisfy TE calls."""
        return {}

    @contextlib.contextmanager
    def fork(self, name=None):
        """Mock fork context manager."""
        try:
            yield
        finally:
            pass  # No cleanup needed for mock


_mock_rng_tracker_instance = MockRNGTracker()  # Create instance once

# --- Attempt Early Patches ---

# Patch CudaRNGStatesTracker class definition in parallel_state (might still be useful)
try:
    import megatron.core.parallel_state

    megatron.core.parallel_state.CudaRNGStatesTracker = MockRNGTracker
    print("Early patched CudaRNGStatesTracker class in megatron.core.parallel_state")
except ImportError:
    print("megatron.core.parallel_state not found for class patch.")
except Exception as e:
    print(f"Error patching parallel_state CudaRNGStatesTracker class: {e}")

# Patch the initializer function in tensor_parallel.random to *always* use the mock
try:
    import megatron.core.tensor_parallel.random

    def mock_initialize_rng_tracker(*args, **kwargs):
        """Initialize the global CUDA RNG tracker with a mock instance.

        This function overrides Megatron's initializer to ensure that the
        global tracker is always a lightweight mock suitable for CPU/meta
        tensor execution in tracing.
        """
        # Directly set the global in that module to our mock instance
        megatron.core.tensor_parallel.random._CUDA_RNG_STATE_TRACKER = _mock_rng_tracker_instance
        # Set the internal _is_initialized flag if the mock tracker has it (optional)
        if hasattr(_mock_rng_tracker_instance, "_is_initialized"):
            _mock_rng_tracker_instance._is_initialized = True

    megatron.core.tensor_parallel.random.initialize_rng_tracker = mock_initialize_rng_tracker
    print("Early patched initialize_rng_tracker in megatron.core.tensor_parallel.random")

    # Call the patched initializer immediately to ensure the global is set
    # Use force_reset=True to guarantee it runs even if it was somehow called before
    mock_initialize_rng_tracker(force_reset=True)
    print("Called patched initialize_rng_tracker to ensure mock instance is set.")

except ImportError:
    print("megatron.core.tensor_parallel.random not found for initializer patch.")
except Exception as e:
    print(f"Error patching tensor_parallel.random initialize_rng_tracker: {e}")

# --- End Early Patch ---

# Apply patch for missing triton module - use a safer check that doesn't rely on find_spec
if "triton" in sys.modules and isinstance(sys.modules["triton"], types.ModuleType):
    # Triton is already mocked or properly imported
    triton_module = sys.modules["triton"]
    # Ensure the language submodule exists
    if not hasattr(triton_module, "language"):
        triton_module.language = types.ModuleType("triton.language")
        # Add commonly used attributes to the language module
        for attr_name in [
            "int32",
            "float32",
            "abs",
            "where",
            "sum",
            "atomic_add",
            "atomic_max",
        ]:
            setattr(triton_module.language, attr_name, lambda *args, **kwargs: None)
        sys.modules["triton.language"] = triton_module.language
else:
    # Import the MockModule from our missing module function
    from megatron.instrumentation.dist_spoofer import MockModule

    print("Triton not found or improperly mocked, using mock implementation")
    triton_mock = MockModule("triton")

    # Create a language submodule with common attributes
    language_mock = MockModule("triton.language")
    # Add common triton language constructs used in the codebase
    for attr_name in [
        "int32",
        "float32",
        "abs",
        "where",
        "sum",
        "atomic_add",
        "atomic_max",
    ]:
        setattr(language_mock, attr_name, lambda *args, **kwargs: None)

    # Assign language to triton and register both
    triton_mock.language = language_mock
    sys.modules["triton"] = triton_mock
    sys.modules["triton.language"] = language_mock

# Import our custom patches
from megatron.instrumentation.patches import patch_all

patch_all()


# Patch transformer_engine module specifically to handle inheritance
def patch_transformer_engine_classes():
    """Add a specific patch for transformer_engine to handle class inheritance."""
    try:
        if "transformer_engine" in sys.modules:
            import torch

            te = sys.modules["transformer_engine"]

            # Create a base class for TE layers
            class MockTELayer(torch.nn.Module):
                def __init__(self, *args, **kwargs):
                    super().__init__()
                    # Store some basic attributes
                    self.in_features = kwargs.get("in_features", 0)
                    self.out_features = kwargs.get("out_features", 0)
                    self.bias = kwargs.get("bias", False)
                    self.use_bias = self.bias
                    self.parallel_mode = kwargs.get("parallel_mode", None)
                    self.tp_size = kwargs.get("tp_size", 1)

                    # Create basic parameters if needed
                    if (
                        hasattr(self, "in_features")
                        and hasattr(self, "out_features")
                        and self.in_features > 0
                        and self.out_features > 0
                    ):
                        self.weight = torch.nn.Parameter(
                            torch.zeros(self.out_features, self.in_features)
                        )
                        if self.bias:
                            self.bias = torch.nn.Parameter(torch.zeros(self.out_features))

                def forward(self, x, *args, **kwargs):
                    """Simple passthrough forward function."""
                    # Always return tensor and None for bias to match TE's interface
                    return x, None

                @classmethod
                def __mro_entries__(cls, bases):
                    """Support proper inheritance."""
                    return (torch.nn.Module,)

            # Ensure pytorch submodule exists
            if not hasattr(te, "pytorch"):
                te.pytorch = types.ModuleType("transformer_engine.pytorch")
                sys.modules["transformer_engine.pytorch"] = te.pytorch

            # Add proper layer types
            te.pytorch.Linear = type("Linear", (MockTELayer,), {})
            te.pytorch.LayerNorm = type("LayerNorm", (MockTELayer,), {})
            te.pytorch.LayerNormLinear = type("LayerNormLinear", (MockTELayer,), {})
            te.pytorch.DotProductAttention = type("DotProductAttention", (MockTELayer,), {})

            # Add tensor submodule and quantized tensor support
            tensor_module = types.ModuleType("transformer_engine.pytorch.tensor")

            # Create a basic QuantizedTensor class
            class QuantizedTensor:
                def __init__(self, *args, **kwargs):
                    pass

                def __class_getitem__(cls, item):
                    return cls

                def cast_to_fp8(self, *args, **kwargs):
                    return self

                @classmethod
                def from_float(cls, *args, **kwargs):
                    return cls()

            # Add to tensor module
            tensor_module.QuantizedTensor = QuantizedTensor

            # Register the module
            te.pytorch.tensor = tensor_module
            sys.modules["transformer_engine.pytorch.tensor"] = tensor_module

            # Ensure common module exists
            if not hasattr(te, "common"):
                te.common = types.ModuleType("transformer_engine.common")
                te.common.recipe = types.ModuleType("transformer_engine.common.recipe")
                sys.modules["transformer_engine.common"] = te.common
                sys.modules["transformer_engine.common.recipe"] = te.common.recipe

            # Add delayed scaling class
            te.common.recipe.DelayedScaling = type(
                "DelayedScaling",
                (object,),
                {
                    "__init__": lambda self, *args, **kwargs: None,
                },
            )

            # Add distributed module
            if not hasattr(te.pytorch, "distributed"):
                te.pytorch.distributed = types.ModuleType("transformer_engine.pytorch.distributed")
                sys.modules["transformer_engine.pytorch.distributed"] = te.pytorch.distributed

            # Add CUDA RNG tracker
            class MockRNGTracker:
                def __init__(self, *args, **kwargs):
                    pass

                def is_initialized(self):
                    return False

            te.pytorch.distributed.CudaRNGStatesTracker = MockRNGTracker

            print("Patched transformer_engine classes for inheritance")

            # Mock multi_tensor_applier to avoid errors with meta tensors
            try:
                # Ensure the submodule path exists
                if not hasattr(te.pytorch, "optimizers"):
                    te.pytorch.optimizers = types.ModuleType(
                        "transformer_engine.pytorch.optimizers"
                    )
                    sys.modules["transformer_engine.pytorch.optimizers"] = te.pytorch.optimizers
                if not hasattr(te.pytorch.optimizers, "multi_tensor_apply"):
                    te.pytorch.optimizers.multi_tensor_apply = types.ModuleType(
                        "transformer_engine.pytorch.optimizers.multi_tensor_apply"
                    )
                    sys.modules["transformer_engine.pytorch.optimizers.multi_tensor_apply"] = (
                        te.pytorch.optimizers.multi_tensor_apply
                    )

                # Define the mock function
                def mock_multi_tensor_applier(*args, **kwargs):
                    # Return dummy values expected by get_grad_norm_fp32
                    # Typically returns [grad_norm_tensor, None]
                    return [torch.tensor(0.0, device="meta"), None]

                # Apply the mock
                te.pytorch.optimizers.multi_tensor_apply.multi_tensor_applier = (
                    mock_multi_tensor_applier
                )
                print("Mocked multi_tensor_applier for meta tensors")
            except Exception as e:
                print(f"Error mocking multi_tensor_applier: {e}")

            # Patch importlib.metadata.version to handle transformer-engine
            try:
                import importlib.metadata

                original_version = importlib.metadata.version

                def patched_version(distribution_name):
                    if distribution_name == "transformer-engine":
                        return "999.999.999"  # Use a fixed version
                    return original_version(distribution_name)

                importlib.metadata.version = patched_version
                print("Patched importlib.metadata.version for transformer-engine")
            except Exception as e:
                print(f"Error patching importlib.metadata.version: {e}")

    except Exception as e:
        print(f"Error patching transformer_engine: {e}")


# Apply the TE patch
patch_transformer_engine_classes()


# Direct monkey patching instead of complex version comparison
def patch_version_checks():
    """Direct patch for version checks to avoid comparison issues."""
    try:
        if "megatron.core.utils" in sys.modules:
            utils_module = sys.modules["megatron.core.utils"]

            # Directly patch the is_te_min_version function
            if hasattr(utils_module, "is_te_min_version"):
                # Just always return True for version checks
                def patched_is_te_min_version(version, check_equality=True):
                    return True

                # Apply the patch
                utils_module.is_te_min_version = patched_is_te_min_version
                print("Patched is_te_min_version to always return True")

            # Also patch get_te_version if necessary
            if hasattr(utils_module, "get_te_version"):
                # Return a Version object
                utils_module.get_te_version = lambda: PkgVersion("999.999.999")
                print("Patched get_te_version to return a Version object")

            # And get_te_version_str
            if hasattr(utils_module, "get_te_version_str"):
                utils_module.get_te_version_str = lambda: "999.999.999"
    except Exception as e:
        print(f"Error patching version checks: {e}")


# Call the patching function
patch_version_checks()

# Define direct imports to avoid relying on megatron.core.utils version checking
from megatron.instrumentation.dist_comm_logger import save_comm_logs

# Model configuration templates for different sizes
GPT_MODEL_CONFIGS = {
    "small": {
        "hidden_size": 768,
        "num_layers": 12,
        "num_attention_heads": 12,
        "seq_length": 1024,
    },
    "medium": {
        "hidden_size": 1024,
        "num_layers": 24,
        "num_attention_heads": 16,
        "seq_length": 1024,
    },
    "large": {
        "hidden_size": 1280,
        "num_layers": 36,
        "num_attention_heads": 20,
        "seq_length": 1024,
    },
    "xl": {
        "hidden_size": 1600,
        "num_layers": 48,
        "num_attention_heads": 25,
        "seq_length": 1024,
    },
    "2b": {
        "hidden_size": 2048,
        "num_layers": 36,
        "num_attention_heads": 32,
        "seq_length": 1024,
    },
    "6b": {
        "hidden_size": 4096,
        "num_layers": 32,
        "num_attention_heads": 32,
        "seq_length": 2048,
    },
    "13b": {
        "hidden_size": 5120,
        "num_layers": 40,
        "num_attention_heads": 40,
        "seq_length": 2048,
    },
    "30b": {
        "hidden_size": 6656,
        "num_layers": 32,
        "num_attention_heads": 52,
        "seq_length": 2048,
    },
    "175b": {
        "hidden_size": 12288,
        "num_layers": 96,
        "num_attention_heads": 96,
        "seq_length": 2048,
    },
}


def initialize_megatron_for_logging(rank, world_size, args_list, port):
    """Initialize Megatron-LM with patched tensor and comm operations."""
    try:
        # Set environment variables for distributed setup
        os.environ["RANK"] = str(rank)
        os.environ["LOCAL_RANK"] = str(
            rank % torch.cuda.device_count() if torch.cuda.is_available() else rank
        )
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(port)

        # Force reload of arguments module in spawned process
        import importlib

        import megatron.training.arguments

        importlib.reload(megatron.training.arguments)
        from megatron.training.arguments import parse_args  # Re-import after reload

        # Ensure we add the allow_no_cuda flag to arguments
        if "--allow-no-cuda" not in args_list:
            args_list.extend(["--allow-no-cuda", "True"])

        # Parse arguments
        # Ignore unknown args like --output-dir which are used by the spoofer itself
        args = parse_args(args_list=args_list, ignore_unknown_args=True)

        # Initialize distributed
        if not dist.is_initialized():
            dist.init_process_group(
                backend="gloo",  # Use gloo as it works on CPU
                init_method=f"tcp://127.0.0.1:{port}",
                world_size=world_size,
                rank=rank,
            )

        # Import and run one iteration
        from megatron.instrumentation.trace_iteration import run_one_iteration

        run_one_iteration(args)

        # Save logs
        # Extract output-dir from args_list
        output_dir = None
        for i, arg in enumerate(args_list):
            if arg == "--output-dir" and i + 1 < len(args_list):
                output_dir = args_list[i + 1]
                break

        if output_dir is None:
            output_dir = args.output_dir

        log_path = save_comm_logs(output_dir)
        print(f"Rank {rank}: Saved logs to {log_path}")

    except Exception as e:
        print(f"Rank {rank} encountered error: {str(e)}")
        import traceback

        traceback.print_exc()
    finally:
        # Cleanup
        if dist.is_initialized():
            dist.destroy_process_group()


def run_megatron_distributed(
    dp_size=1,
    tp_size=1,
    pp_size=1,
    model_size="small",
    seed=1234,
    mock_data=True,
    output_dir="comm_logs",
    num_micro_batches=4,
    global_batch_size=1024,
):
    """Run Megatron-LM in a distributed setting with the given parallelism parameters."""
    # Mock dependencies before starting processes
    mock_missing_modules()

    # Calculate world size
    world_size = dp_size * tp_size * pp_size

    # Select model config based on size
    if model_size not in GPT_MODEL_CONFIGS:
        print(f"Model size {model_size} not found. Using 'small' instead.")
        model_size = "small"

    model_config = GPT_MODEL_CONFIGS[model_size]

    # Generate output directory with timestamp
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_path = Path(output_dir) / f"{model_size}_dp{dp_size}_tp{tp_size}_pp{pp_size}_{timestamp}"
    output_path.mkdir(parents=True, exist_ok=True)

    # Save configuration
    config = {
        "model_size": model_size,
        "model_config": model_config,
        "data_parallel_size": dp_size,
        "tensor_model_parallel_size": tp_size,
        "pipeline_model_parallel_size": pp_size,
        "world_size": world_size,
        "seed": seed,
        "timestamp": timestamp,
        "num_micro_batches": num_micro_batches,
        "global_batch_size": global_batch_size,
    }

    with open(output_path / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Prepare arguments
    port = 29500 + (os.getpid() % 10000) % 10000  # Generate a random port

    args_list = [
        "--tensor-model-parallel-size",
        str(tp_size),
        "--pipeline-model-parallel-size",
        str(pp_size),
        "--num-layers",
        str(model_config["num_layers"]),
        "--hidden-size",
        str(model_config["hidden_size"]),
        "--num-attention-heads",
        str(model_config["num_attention_heads"]),
        "--seq-length",
        str(model_config["seq_length"]),
        "--max-position-embeddings",
        str(model_config["seq_length"]),
        "--micro-batch-size",
        str(global_batch_size // (num_micro_batches * dp_size)),
        "--global-batch-size",
        str(global_batch_size),
        "--train-iters",
        "1",  # Just one iteration
        "--log-interval",
        "1",
        "--eval-interval",
        "100",  # Set high to avoid evaluation
        "--seed",
        str(seed),
        "--output-dir",
        str(output_path),
        "--no-load-optim",
        "--no-load-rng",
        "--use-cpu-initialization",
        "--distributed-backend",
        "gloo",  # Use gloo backend
        "--expert-model-parallel-size",
        "1",  # Disable expert parallelism
        # Add tokenizer args required by Megatron
        "--tokenizer-type",
        "GPT2BPETokenizer",
        "--vocab-file",
        "dummy_vocab.json",  # Placeholder, might not be read with mock_data
        "--merge-file",
        "dummy_merges.txt",  # Placeholder, might not be read with mock_data
        "--lr",
        "0.0001",
        "--lr-decay-iters",
        "1000",
        "--lr-decay-style",
        "cosine",
        "--min-lr",
        "0.00001",
        "--transformer-impl",
        "local",
        "--make-vocab-size-divisible-by",
        "50257",
    ]

    # Conditionally add boolean flags
    if mock_data:
        args_list.append("--mock-data")

    # Launch processes
    print(f"Starting {world_size} processes for distributed logging")
    mp.spawn(
        initialize_megatron_for_logging,
        args=(world_size, args_list, port),
        nprocs=world_size,
        join=True,
    )

    # Generate a summary file
    summary = {
        "config": config,
        "log_files": [f"comm_log_rank_{i}.json" for i in range(world_size)],
    }

    with open(output_path / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Distributed logging completed. Results saved to {output_path}")
    return output_path
