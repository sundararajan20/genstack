# Copyright (c) 2025, The Board of Trustees of the Leland Stanford Junior University.

# This source code is licensed under the Apache license found in the
# LICENSE file in the root directory of this source tree.

"""
Centralized patching functionality for Megatron-LM instrumentation.
This module applies all required patches to make the instrumentation work.
"""

import sys
import types

from .dist_comm_logger import apply_dist_comm_patches
from .meta_tensor import apply_meta_tensor_patches


def patch_all(verbose=True, force_meta=True):
    """
    Apply all patches required for instrumentation to work correctly.

    Args:
        verbose: If True, print information about applied patches
        force_meta: If True, force all tensors to be meta tensors
    """
    # Apply meta tensor patches
    apply_meta_tensor_patches(verbose=verbose, force_meta=force_meta)

    # Apply distributed communication patches
    apply_dist_comm_patches(verbose=verbose)

    # Apply additional patches
    _patch_additional_modules(verbose=verbose)

    if verbose:
        print("All instrumentation patches applied successfully")


def _patch_additional_modules(verbose=True):
    """
    Apply additional patches needed for compatibility with various frameworks.

    Args:
        verbose: If True, print information about applied patches
    """
    # Get access to builtins to patch isinstance checks if needed
    import builtins

    original_isinstance = builtins.isinstance

    # Patch isinstance to handle our mock modules
    def patched_isinstance(obj, class_or_tuple):
        # First try the original implementation
        try:
            result = original_isinstance(obj, class_or_tuple)
            if result:
                return True
        except Exception:
            pass

        # Custom handling for our mock modules
        if hasattr(obj, "__class__") and hasattr(obj.__class__, "__name__"):
            if obj.__class__.__name__ == "MockModule":
                # Handle checking against common base classes
                if class_or_tuple == types.ModuleType:
                    return True

        return False

    # Apply the isinstance patch
    builtins.isinstance = patched_isinstance

    if verbose:
        print("Applied additional compatibility patches")


def remove_all_patches(verbose=True):
    """
    Remove all patches applied by patch_all.

    Args:
        verbose: If True, print information about removed patches
    """
    from .dist_comm_logger import remove_dist_comm_patches
    from .meta_tensor import remove_meta_tensor_patches

    # Remove meta tensor patches
    remove_meta_tensor_patches(verbose=verbose)

    # Remove distributed communication patches
    remove_dist_comm_patches(verbose=verbose)

    # Remove additional patches
    _remove_additional_patches(verbose=verbose)

    if verbose:
        print("All instrumentation patches removed successfully")


def _remove_additional_patches(verbose=True):
    """
    Remove additional patches applied by _patch_additional_modules.

    Args:
        verbose: If True, print information about removed patches
    """
    # Remove builtins patches
    import builtins

    if hasattr(builtins, "isinstance") and builtins.isinstance.__name__ == "patched_isinstance":
        builtins.isinstance = getattr(builtins.isinstance, "_original", builtins.isinstance)

    if verbose:
        print("Removed additional compatibility patches")
