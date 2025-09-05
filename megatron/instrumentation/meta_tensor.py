# Copyright (c) 2025, The Board of Trustees of the Leland Stanford Junior University.

# This source code is licensed under the Apache license found in the
# LICENSE file in the root directory of this source tree.

"""
Meta tensor functionality for Megatron-LM instrumentation.
This module provides patches to use PyTorch's built-in meta device tensors,
allowing tensor shape tracking without requiring actual GPU hardware.
"""

import functools
import types
from typing import Any, Dict, List, Optional, Tuple, Union

import torch

# Global statistics collection
_meta_tensor_stats = {
    "created": 0,
    "operations": {},
}

# Store original tensor creation functions
_ORIGINAL_TENSOR_FUNCS = {}


def _create_meta_tensor(shape, dtype=None, **kwargs):
    """
    Create a proper PyTorch meta tensor.

    Args:
        shape: Tensor shape
        dtype: Tensor dtype (defaults to float32)
        **kwargs: Additional arguments

    Returns:
        A PyTorch tensor on the meta device
    """
    if dtype is None:
        dtype = torch.float32

    # Track stats
    global _meta_tensor_stats
    _meta_tensor_stats["created"] += 1

    # Use the original torch.empty function to create the meta tensor
    # to avoid recursion
    orig_empty = _ORIGINAL_TENSOR_FUNCS.get("empty", torch.empty)
    return orig_empty(shape, dtype=dtype, device="meta")


def _patch_tensor_func(name, force_meta=False):
    """Patch a tensor creation function to return meta tensors when requested."""
    if name in _ORIGINAL_TENSOR_FUNCS:
        # Already patched, avoid double patching
        return

    orig_func = getattr(torch, name)
    _ORIGINAL_TENSOR_FUNCS[name] = orig_func

    @functools.wraps(orig_func)
    def wrapped_func(*args, **kwargs):
        device = kwargs.get("device", None)

        # Force meta device if requested
        if force_meta:
            if len(args) == 1 and isinstance(args[0], (list, tuple)):
                shape = args[0]
            else:
                shape = args

            dtype = kwargs.get("dtype", torch.float32)
            return _create_meta_tensor(shape, dtype=dtype)

        # Use meta device when explicitly requested or when is_meta flag is set
        if device == "meta" or kwargs.get("is_meta", False):
            if len(args) == 1 and isinstance(args[0], (list, tuple)):
                shape = args[0]
            else:
                shape = args

            dtype = kwargs.get("dtype", torch.float32)
            return _create_meta_tensor(shape, dtype=dtype)

        # For CPU-only systems, if no device is specified, use meta by default
        if not torch.cuda.is_available() and "device" not in kwargs:
            kwargs["device"] = "meta"

        # Fall back to original implementation
        return orig_func(*args, **kwargs)

    setattr(torch, name, wrapped_func)


def _patch_tensor_like_func(name, force_meta=False):
    """Patch tensor-like creation functions (zeros_like, etc.)."""
    if name in _ORIGINAL_TENSOR_FUNCS:
        # Already patched, avoid double patching
        return

    orig_func = getattr(torch, name)
    _ORIGINAL_TENSOR_FUNCS[name] = orig_func

    @functools.wraps(orig_func)
    def wrapped_func(input_tensor, *args, **kwargs):
        device = kwargs.get("device", None)

        # Force meta device if requested
        if force_meta:
            # Create a meta tensor with the same shape as input_tensor
            dtype = kwargs.get("dtype", input_tensor.dtype)
            return _create_meta_tensor(input_tensor.shape, dtype=dtype)

        # Use meta device when explicitly requested or when is_meta flag is set
        if device == "meta" or kwargs.get("is_meta", False):
            # Create a meta tensor with the same shape as input_tensor
            dtype = kwargs.get("dtype", input_tensor.dtype)
            return _create_meta_tensor(input_tensor.shape, dtype=dtype)

        # For CPU-only systems, if no device is specified, use meta by default
        if not torch.cuda.is_available() and "device" not in kwargs:
            kwargs["device"] = "meta"

        # Fall back to original implementation
        return orig_func(input_tensor, *args, **kwargs)

    setattr(torch, name, wrapped_func)


def _patch_tensor_methods():
    """Patch tensor methods that should be handled specially for meta tensors."""
    # For methods like cuda(), we want to ensure they stay on meta device

    # Patch to() method
    if "Tensor.to" in _ORIGINAL_TENSOR_FUNCS:
        # Already patched
        return

    orig_to = torch.Tensor.to
    _ORIGINAL_TENSOR_FUNCS["Tensor.to"] = orig_to

    @functools.wraps(orig_to)
    def wrapped_to(self, *args, **kwargs):
        # If the tensor is already on meta device, keep it there
        if getattr(self, "is_meta", False):
            return self

        # Otherwise use the original method
        return orig_to(self, *args, **kwargs)

    torch.Tensor.to = wrapped_to

    # Patch cuda() method
    orig_cuda = torch.Tensor.cuda
    _ORIGINAL_TENSOR_FUNCS["Tensor.cuda"] = orig_cuda

    @functools.wraps(orig_cuda)
    def wrapped_cuda(self, *args, **kwargs):
        # If the tensor is on meta device, keep it there
        if getattr(self, "is_meta", False):
            return self

        # Otherwise use the original method
        return orig_cuda(self, *args, **kwargs)

    torch.Tensor.cuda = wrapped_cuda


def apply_meta_tensor_patches(verbose=False, force_meta=True):
    """
    Apply patches to PyTorch to use meta tensors.

    Args:
        verbose: If True, print progress messages
        force_meta: If True, force all tensors to be meta tensors regardless of device
    """
    if verbose:
        print("Applying meta tensor patches to PyTorch")

    # Store originals first before patching
    if not _ORIGINAL_TENSOR_FUNCS:
        for func_name in [
            "empty",
            "zeros",
            "ones",
            "randn",
            "rand",
            "full",
            "tensor",
            "arange",
        ]:
            if hasattr(torch, func_name):
                _ORIGINAL_TENSOR_FUNCS[func_name] = getattr(torch, func_name)

        for like_func in ["zeros_like", "ones_like", "empty_like", "full_like"]:
            if hasattr(torch, like_func):
                _ORIGINAL_TENSOR_FUNCS[like_func] = getattr(torch, like_func)

        _ORIGINAL_TENSOR_FUNCS["Tensor.to"] = torch.Tensor.to
        _ORIGINAL_TENSOR_FUNCS["Tensor.cuda"] = torch.Tensor.cuda

    # Patch tensor creation functions
    for func_name in [
        "zeros",
        "ones",
        "empty",
        "randn",
        "rand",
        "full",
        "tensor",
        "arange",
    ]:
        _patch_tensor_func(func_name, force_meta)

    # Patch tensor-like creation functions
    for like_func in ["zeros_like", "ones_like", "empty_like", "full_like"]:
        _patch_tensor_like_func(like_func, force_meta)

    # Patch tensor methods
    _patch_tensor_methods()

    if verbose:
        print("Meta tensor patches applied successfully")


def remove_meta_tensor_patches(verbose=False):
    """
    Remove the patches applied by apply_meta_tensor_patches.

    Args:
        verbose: If True, print progress messages
    """
    for func_name, orig_func in _ORIGINAL_TENSOR_FUNCS.items():
        if "." in func_name:
            # Handle methods like Tensor.to
            class_name, method_name = func_name.split(".")
            if class_name == "Tensor":
                setattr(torch.Tensor, method_name, orig_func)
        else:
            # Handle top-level functions
            setattr(torch, func_name, orig_func)

    # Clear the original function dictionary
    _ORIGINAL_TENSOR_FUNCS.clear()

    if verbose:
        print("Meta tensor patches removed")


def initialize_meta_device(verbose=False):
    """
    Initialize the meta device for tensor operations.

    Args:
        verbose: If True, print progress messages
    """
    apply_meta_tensor_patches(verbose=verbose)
    if verbose:
        print("Meta device initialized")


def is_meta_tensor(tensor):
    """Check if a tensor is a meta tensor."""
    return getattr(tensor, "is_meta", False)


def wrap_module(module):
    """
    Wrap a PyTorch module to use meta tensors.

    Args:
        module: The PyTorch module to wrap

    Returns:
        The wrapped module using meta tensors
    """
    # For each parameter in the module, convert it to a meta tensor
    for name, param in module.named_parameters():
        # Use original empty to avoid recursion
        orig_empty = _ORIGINAL_TENSOR_FUNCS.get("empty", torch.empty)
        meta_param = orig_empty(param.shape, dtype=param.dtype, device="meta")
        meta_param.requires_grad = param.requires_grad
        setattr(module, name, meta_param)

    # For each buffer in the module, convert it to a meta tensor
    for name, buffer in module.named_buffers():
        orig_empty = _ORIGINAL_TENSOR_FUNCS.get("empty", torch.empty)
        meta_buffer = orig_empty(buffer.shape, dtype=buffer.dtype, device="meta")
        module.register_buffer(name, meta_buffer)

    # Recursively wrap submodules
    for name, submodule in module.named_children():
        wrapped_submodule = wrap_module(submodule)
        setattr(module, name, wrapped_submodule)

    return module


def get_meta_stats():
    """Get statistics about meta tensor usage."""
    return _meta_tensor_stats


def clear_meta_stats():
    """Clear the meta tensor usage statistics."""
    global _meta_tensor_stats
    _meta_tensor_stats = {
        "created": 0,
        "operations": {},
    }


class MetaTensor:
    """
    A thin wrapper around PyTorch meta tensors for backwards compatibility.
    This should be used as a type annotation or for isinstance checks only.
    All actual tensor operations should use proper PyTorch meta tensors.
    """

    @staticmethod
    def create(shape, dtype=None, **kwargs):
        """
        Create a proper PyTorch meta tensor.
        This static method should be used instead of the constructor.

        Args:
            shape: Tensor shape
            dtype: Tensor dtype (defaults to float32)
            **kwargs: Additional arguments

        Returns:
            A PyTorch tensor on the meta device
        """
        return _create_meta_tensor(shape, dtype)
