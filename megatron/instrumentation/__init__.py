# Copyright (c) 2025, The Board of Trustees of the Leland Stanford Junior University.

# This source code is licensed under the Apache license found in the
# LICENSE file in the root directory of this source tree.

"""
Megatron-LM Instrumentation Package
===================================

This package provides instrumentation tools for Megatron-LM that allow
analysis of model execution without requiring actual GPU hardware.

Main components:
- Meta tensor implementation (mock GPU tensors)
- Distributed communication logging
- Patching mechanisms for GPU-dependent functionality

Usage:
    from megatron.instrumentation import initialize
    initialize()
"""

import importlib
import os
import sys
import types


# Create a proper mock module to handle Triton imports
class MockModule(types.ModuleType):
    """
    A more robust mock module that can be used as a substitute for missing dependencies.
    This implementation properly supports iteration and other module protocols.
    """

    def __init__(self, name):
        super().__init__(name)
        self._name = name
        self._submodules = {}
        self._attributes = {}

        # Add a spec to avoid errors with find_spec
        self.__spec__ = types.SimpleNamespace(
            name=name, loader=None, origin="mock", submodule_search_locations=[]
        )

    def __getattr__(self, key):
        # Return from attributes if it exists
        if key in self._attributes:
            return self._attributes[key]

        # Create submodules on demand
        if key not in self._submodules:
            submodule_name = f"{self._name}.{key}"
            self._submodules[key] = MockModule(submodule_name)
        return self._submodules[key]

    def __setattr__(self, key, value):
        # Special handling for certain attributes
        if key.startswith("_"):
            # For internal attributes like _name, use normal assignment
            super().__setattr__(key, value)
        else:
            # Store in our custom attributes dict
            self._attributes[key] = value

    def __call__(self, *args, **kwargs):
        # Return a mock class that can be subclassed
        class MockClass:
            def __init__(self, *a, **kw):
                pass

            # Support inheritance in Python classes
            @classmethod
            def __mro_entries__(cls, bases):
                return (object,)

        return MockClass

    def __dir__(self):
        # Support dir() calls
        return (
            list(self._submodules.keys()) + list(self._attributes.keys()) + list(super().__dir__())
        )

    def __iter__(self):
        # Support iteration protocol
        return iter([])

    # Add string representation for better debugging
    def __repr__(self):
        return f"<MockModule '{self._name}'>"

    def __str__(self):
        return f"<MockModule '{self._name}'>"


# Patch os.path functions to handle MockModule instances
orig_splitext = os.path.splitext


def patched_splitext(p):
    if isinstance(p, MockModule):
        return ("mock_module", ".py")
    return orig_splitext(p)


os.path.splitext = patched_splitext

orig_basename = os.path.basename


def patched_basename(p):
    if isinstance(p, MockModule):
        return "mock_module.py"
    return orig_basename(p)


os.path.basename = patched_basename

orig_dirname = os.path.dirname


def patched_dirname(p):
    if isinstance(p, MockModule):
        return "/mock/path"
    return orig_dirname(p)


os.path.dirname = patched_dirname

orig_abspath = os.path.abspath


def patched_abspath(p):
    if isinstance(p, MockModule):
        return "/mock/path/mock_module.py"
    return orig_abspath(p)


os.path.abspath = patched_abspath

orig_normcase = os.path.normcase


def patched_normcase(p):
    if isinstance(p, MockModule):
        return "/mock/path/mock_module.py"
    return orig_normcase(p)


os.path.normcase = patched_normcase

# Patch inspect module to handle MockModule
try:
    import inspect

    orig_getmodule = inspect.getmodule

    def patched_getmodule(object, _filename=None):
        if isinstance(_filename, MockModule):
            return None
        return orig_getmodule(object, _filename)

    inspect.getmodule = patched_getmodule

    orig_getabsfile = inspect.getabsfile

    def patched_getabsfile(object):
        if isinstance(object, MockModule):
            return "/mock/path/mock_module.py"
        try:
            return orig_getabsfile(object)
        except (TypeError, ValueError):
            # Fallback for objects that orig_getabsfile can't handle
            return "/unknown/path"

    # Use a try-except block to avoid type errors in strict type checking
    try:
        inspect.getabsfile = patched_getabsfile
    except TypeError:
        print("Warning: Could not patch inspect.getabsfile due to type restrictions")
except ImportError:
    pass


# For transformer_engine mock, we need more specific classes
def _create_transformer_engine_mock():
    """Create a specialized mock for transformer_engine."""
    te_mock = MockModule("transformer_engine")

    # Add version information directly
    te_mock.__version__ = "0.0.0"

    # Create pytorch submodule
    pytorch_mock = MockModule("transformer_engine.pytorch")

    # Create dummy base classes for TE layers
    class LinearBase:
        def __init__(self, *args, **kwargs):
            pass

        def forward(self, x):
            return x

        @classmethod
        def __mro_entries__(cls, bases):
            return (torch.nn.Module,)

    class LayerNormBase:
        def __init__(self, *args, **kwargs):
            pass

        def forward(self, x):
            return x

        @classmethod
        def __mro_entries__(cls, bases):
            return (torch.nn.Module,)

    class RMSNormBase:
        def __init__(self, *args, **kwargs):
            pass

        def forward(self, x):
            return x

        @classmethod
        def __mro_entries__(cls, bases):
            return (torch.nn.Module,)

    class DotProductAttentionBase:
        def __init__(self, *args, **kwargs):
            pass

        def forward(self, q, k, v, mask=None):
            return q

        @classmethod
        def __mro_entries__(cls, bases):
            return (torch.nn.Module,)

    class GroupedLinearBase:
        def __init__(self, *args, **kwargs):
            pass

        def forward(self, x, splits):
            return x, None

        @classmethod
        def __mro_entries__(cls, bases):
            return (torch.nn.Module,)

    class LayerNormLinearBase:
        def __init__(self, *args, **kwargs):
            pass

        def forward(self, x):
            # Return tensor and None for bias
            return x, None

        @classmethod
        def __mro_entries__(cls, bases):
            return (torch.nn.Module,)

    # Assign the base classes
    pytorch_mock.Linear = LinearBase
    pytorch_mock.LayerNorm = LayerNormBase
    pytorch_mock.RMSNorm = RMSNormBase
    pytorch_mock.DotProductAttention = DotProductAttentionBase
    pytorch_mock.GroupedLinear = GroupedLinearBase
    pytorch_mock.LayerNormLinear = LayerNormLinearBase

    # Add functions
    def dummy_apply_rotary_pos_emb(*args, **kwargs):
        # Return the input tensor unchanged
        return args[0]

    def get_cpu_offload_context(*args, **kwargs):
        # Return a dummy context manager
        class DummyContext:
            def __enter__(self):
                pass

            def __exit__(self, *args):
                pass

        return DummyContext()

    def te_checkpoint(*args, **kwargs):
        # Return the input tensor unchanged
        return args[0] if args else None

    # Create classes for fp8 support
    class Fp8Padding:
        def __init__(self, *args, **kwargs):
            pass

        def __call__(self, x):
            return x

    class Fp8Unpadding:
        def __init__(self, *args, **kwargs):
            pass

        def __call__(self, x):
            return x

    # Add all the objects to the mock
    pytorch_mock.attention = MockModule("transformer_engine.pytorch.attention")
    pytorch_mock.attention.apply_rotary_pos_emb = dummy_apply_rotary_pos_emb
    pytorch_mock.Fp8Padding = Fp8Padding
    pytorch_mock.Fp8Unpadding = Fp8Unpadding
    pytorch_mock.get_cpu_offload_context = get_cpu_offload_context
    pytorch_mock.te_checkpoint = te_checkpoint

    # Add the pytorch submodule to the main mock
    te_mock.pytorch = pytorch_mock

    return te_mock


# Mock external dependencies that might be missing
for module_name in ["triton", "apex", "flash_attn"]:
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
    # Only import if not already in sys.modules
    if "transformer_engine" not in sys.modules:
        import_module = importlib.import_module("transformer_engine")
except (ImportError, ModuleNotFoundError):
    print(f"Module 'transformer_engine' not found, using specialized mock implementation")
    te_mock = _create_transformer_engine_mock()
    sys.modules["transformer_engine"] = te_mock

# Completely disable torch.compile and torch.jit functionality
# since they don't work with meta tensors
try:
    import torch

    # Define a simple passthrough function for decorators
    def passthrough_decorator(fn):
        return fn

    # Patch torch.jit methods
    if hasattr(torch, "jit"):
        # Save original
        orig_jit = torch.jit

        # Create a mock jit module that passes through decorators
        jit_mock = types.ModuleType("torch.jit")

        # Add attributes more safely
        for attr_name in [
            "script",
            "trace",
            "script_if_tracing",
            "ignore",
            "unused",
            "export",
        ]:
            try:
                setattr(jit_mock, attr_name, passthrough_decorator)
            except (AttributeError, TypeError):
                # If we can't set the attribute, print a warning but continue
                print(f"Warning: Could not set {attr_name} on jit_mock")

        # Transfer any attributes we didn't explicitly mock
        for key in dir(orig_jit):
            if not hasattr(jit_mock, key):
                try:
                    setattr(jit_mock, key, getattr(orig_jit, key))
                except (AttributeError, TypeError):
                    pass

        # Replace torch.jit
        try:
            torch.jit = jit_mock
            print("Patched torch.jit to be passthrough decorators")
        except (AttributeError, TypeError):
            print("Warning: Could not patch torch.jit due to type restrictions")

    # Also patch torch.compile
    if hasattr(torch, "compile"):
        # Save the original compile function
        orig_compile = torch.compile

        # Create a passthrough decorator that preserves function signatures
        def patched_compile(*args, **kwargs):
            # If called with a function as first argument, return the function unchanged
            if args and callable(args[0]):
                return args[0]
            # Otherwise return a decorator that just returns the function
            return passthrough_decorator

        # Replace torch.compile with our patched version
        torch.compile = patched_compile
        print("Patched torch.compile to be a no-op")

    # Disable dynamo
    if hasattr(torch, "_dynamo"):
        # Try to disable dynamo errors
        if hasattr(torch._dynamo, "config"):
            torch._dynamo.config.suppress_errors = True
            torch._dynamo.config.verbose = False

        # Make optimize a no-op
        if hasattr(torch._dynamo, "optimize"):
            torch._dynamo.optimize = lambda *args, **kwargs: passthrough_decorator

except ImportError:
    print("Could not import torch, skipping torch patches")

# Import our modules after mocking dependencies
from .patches import patch_all

# Apply patches by default when the module is imported
patch_all()

# Import the main components
# Using import with try-except to handle potential missing functions
from .dist_comm_logger import (
    CommLogger,
    clear_comm_stats,
    get_comm_stats,
    log_communication,
)
from .meta_tensor import (
    MetaTensor,
    clear_meta_stats,
    get_meta_stats,
    initialize_meta_device,
    is_meta_tensor,
    wrap_module,
)
from .patches import patch_all, remove_all_patches


# Main entry point
def initialize(verbose=True, with_module_patching=True):
    """
    Initialize the instrumentation package.

    This function:
    1. Initializes the meta device
    2. Applies required patches for GPU functions
    3. Sets up distributed communication logging

    Args:
        verbose: If True, print progress information
        with_module_patching: If True, apply patches to GPU-dependent modules
    """
    if verbose:
        print("Initializing Megatron-LM instrumentation...")

    # Initialize meta device
    initialize_meta_device(verbose=verbose)

    # Apply patches if requested
    if with_module_patching:
        patch_all(verbose=verbose)

    if verbose:
        print("Megatron-LM instrumentation initialized successfully")


# Export main functions
__all__ = [
    # Main entry point
    "initialize",
    # Meta tensor functionality
    "initialize_meta_device",
    "wrap_module",
    "MetaTensor",
    "is_meta_tensor",
    "get_meta_stats",
    "clear_meta_stats",
    # Distributed communication
    "get_comm_stats",
    "clear_comm_stats",
    "log_communication",
    "CommLogger",
    # Module graph
    "build_module_graph",
    "visualize_module_graph",
    # Patching
    "patch_all",
    "remove_all_patches",
]


# Patch Megatron's version checking to avoid recursion with our mocks
# IMPORTANT: We can't directly import megatron.core.utils here as it would cause a circular import
# Instead, patch the module only after it's loaded elsewhere
def _patch_te_version_check():
    """
    Patch transformer_engine version checking functions to return fixed versions.
    This function is designed to be called later, after all imports are done.
    """
    # Don't use importlib here to avoid recursive imports
    try:
        if "megatron.core.utils" in sys.modules:
            utils_module = sys.modules["megatron.core.utils"]

            # Only patch if the module exists and has the attributes we need
            if hasattr(utils_module, "get_te_version"):
                # Create a fixed version using the version class from the module
                if hasattr(utils_module, "PkgVersion"):
                    _patched_version = utils_module.PkgVersion("0.0.0")

                    # Define patched function that returns our fixed version
                    def patched_get_te_version():
                        return _patched_version

                    # Apply patch
                    utils_module.get_te_version = patched_get_te_version

                    # Also patch get_te_version_str if it exists
                    if hasattr(utils_module, "get_te_version_str"):
                        utils_module.get_te_version_str = lambda: "0.0.0"

                    print("Patched Megatron's transformer_engine version checks")
    except Exception as e:
        print(f"Could not patch Megatron's transformer_engine version checks: {e}")


# Register our patching function to run after importing
# We'll attempt it now, but it will likely run again when fully loaded
_patch_te_version_check()

# Also register it to run at the end of module initialization
import atexit

atexit.register(_patch_te_version_check)
