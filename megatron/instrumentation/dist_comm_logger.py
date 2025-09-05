# Copyright (c) 2025, The Board of Trustees of the Leland Stanford Junior University.

# This source code is licensed under the Apache license found in the
# LICENSE file in the root directory of this source tree.

"""
Distributed communication logger for Megatron-LM.
This module patches PyTorch's distributed communication functions
to log calls instead of performing actual communication.
"""

import contextlib
import functools
import json
import os
import threading
import types

import torch
import torch.distributed as dist

# Store original distributed functions
_ORIGINAL_DIST_FUNCS = {}

# Global variable to store logs per rank
_COMM_LOGS = []
_COMM_LOG_LOCK = threading.Lock()

# Global counter for call IDs
_CALL_ID_COUNTER = 0
_CALL_ID_LOCK = threading.Lock()

# Maps process groups to lists of ranks
_GROUP_TO_RANKS = {}


def get_next_call_id():
    """Get a unique ID for each communication call."""
    global _CALL_ID_COUNTER
    with _CALL_ID_LOCK:
        call_id = _CALL_ID_COUNTER
        _CALL_ID_COUNTER += 1
    return call_id


def log_event(event_dict):
    """Log a communication event."""
    # Add rank info and timestamp
    event_dict["src_rank"] = dist.get_rank() if dist.is_initialized() else 0

    with _COMM_LOG_LOCK:
        _COMM_LOGS.append(event_dict)


def get_comm_logs():
    """Get all logged communication events."""
    with _COMM_LOG_LOCK:
        return list(_COMM_LOGS)


def clear_comm_logs():
    """Clear all logged communication events."""
    with _COMM_LOG_LOCK:
        _COMM_LOGS.clear()


def save_comm_logs(output_dir):
    """Save communication logs to a file."""
    os.makedirs(output_dir, exist_ok=True)
    rank = dist.get_rank() if dist.is_initialized() else 0
    output_file = os.path.join(output_dir, f"comm_log_rank_{rank}.json")

    with open(output_file, "w") as f:
        json.dump(get_comm_logs(), f, indent=2)

    return output_file


def _patch_dist_init():
    """Patch torch.distributed.init_process_group."""
    orig_init = dist.init_process_group
    _ORIGINAL_DIST_FUNCS["init_process_group"] = orig_init

    @functools.wraps(orig_init)
    def wrapped_init(backend="gloo", **kwargs):
        result = orig_init(backend=backend, **kwargs)
        # Record the init_process_group call
        log_event(
            {
                "op": "init_process_group",
                "call_id": get_next_call_id(),
                "backend": backend,
                "world_size": dist.get_world_size(),
            }
        )
        return result

    dist.init_process_group = wrapped_init


def _patch_dist_new_group():
    """Patch torch.distributed.new_group to track group membership."""
    orig_new_group = dist.new_group
    _ORIGINAL_DIST_FUNCS["new_group"] = orig_new_group

    @functools.wraps(orig_new_group)
    def wrapped_new_group(ranks=None, **kwargs):
        # Create the process group
        group = orig_new_group(ranks=ranks, **kwargs)

        # Store mapping of group to ranks
        if ranks is None:
            ranks = list(range(dist.get_world_size()))
        _GROUP_TO_RANKS[group] = ranks

        # Log the group creation
        log_event(
            {
                "op": "new_group",
                "call_id": get_next_call_id(),
                "ranks": ranks,
                "backend": kwargs.get("backend", None),
            }
        )

        return group

    dist.new_group = wrapped_new_group


def _patch_dist_all_reduce():
    """Patch torch.distributed.all_reduce."""
    orig_all_reduce = dist.all_reduce
    _ORIGINAL_DIST_FUNCS["all_reduce"] = orig_all_reduce

    @functools.wraps(orig_all_reduce)
    def wrapped_all_reduce(tensor, op=dist.ReduceOp.SUM, group=None, async_op=False):
        # Get the group ranks
        if group is None:
            ranks = list(range(dist.get_world_size()))
        else:
            ranks = _GROUP_TO_RANKS.get(group, [])

        # Log the all_reduce call
        log_event(
            {
                "op": "all_reduce",
                "call_id": get_next_call_id(),
                "ranks": ranks,
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "reduce_op": str(op),
            }
        )

        # Return a dummy future if async
        if async_op:

            class DummyFuture:
                def wait(self):
                    pass

            return DummyFuture()

        return None

    dist.all_reduce = wrapped_all_reduce


def _patch_dist_all_gather():
    """Patch torch.distributed.all_gather."""
    orig_all_gather = dist.all_gather
    _ORIGINAL_DIST_FUNCS["all_gather"] = orig_all_gather

    @functools.wraps(orig_all_gather)
    def wrapped_all_gather(tensor_list, tensor, group=None, async_op=False):
        # Get the group ranks
        if group is None:
            ranks = list(range(dist.get_world_size()))
        else:
            ranks = _GROUP_TO_RANKS.get(group, [])

        world_size = len(ranks)

        # Fill tensor_list with dummy tensors
        for i in range(len(tensor_list)):
            tensor_list[i].copy_(torch.empty_like(tensor))

        # Log the all_gather call
        log_event(
            {
                "op": "all_gather",
                "call_id": get_next_call_id(),
                "ranks": ranks,
                "tensor_shape": list(tensor.shape),
                "output_shapes": [list(t.shape) for t in tensor_list],
                "dtype": str(tensor.dtype),
            }
        )

        # Return a dummy future if async
        if async_op:

            class DummyFuture:
                def wait(self):
                    pass

            return DummyFuture()

        return None

    dist.all_gather = wrapped_all_gather


def _patch_dist_broadcast():
    """Patch torch.distributed.broadcast."""
    orig_broadcast = dist.broadcast
    _ORIGINAL_DIST_FUNCS["broadcast"] = orig_broadcast

    @functools.wraps(orig_broadcast)
    def wrapped_broadcast(tensor, src, group=None, async_op=False):
        # Get the group ranks
        if group is None:
            ranks = list(range(dist.get_world_size()))
        else:
            ranks = _GROUP_TO_RANKS.get(group, [])

        # Log the broadcast call
        log_event(
            {
                "op": "broadcast",
                "call_id": get_next_call_id(),
                "ranks": ranks,
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "src": src,
            }
        )

        # Return a dummy future if async
        if async_op:

            class DummyFuture:
                def wait(self):
                    pass

            return DummyFuture()

        return None

    dist.broadcast = wrapped_broadcast


def _patch_dist_reduce_scatter():
    """Patch torch.distributed.reduce_scatter."""
    orig_reduce_scatter = dist.reduce_scatter
    _ORIGINAL_DIST_FUNCS["reduce_scatter"] = orig_reduce_scatter

    @functools.wraps(orig_reduce_scatter)
    def wrapped_reduce_scatter(
        output, input_list, op=dist.ReduceOp.SUM, group=None, async_op=False
    ):
        # Get the group ranks
        if group is None:
            ranks = list(range(dist.get_world_size()))
        else:
            ranks = _GROUP_TO_RANKS.get(group, [])

        # Log the reduce_scatter call
        log_event(
            {
                "op": "reduce_scatter",
                "call_id": get_next_call_id(),
                "ranks": ranks,
                "output_shape": list(output.shape),
                "input_shapes": [list(t.shape) for t in input_list],
                "dtype": str(output.dtype),
                "reduce_op": str(op),
            }
        )

        # Return a dummy future if async
        if async_op:

            class DummyFuture:
                def wait(self):
                    pass

            return DummyFuture()

        return None

    dist.reduce_scatter = wrapped_reduce_scatter


def _patch_dist_scatter():
    """Patch torch.distributed.scatter."""
    orig_scatter = dist.scatter
    _ORIGINAL_DIST_FUNCS["scatter"] = orig_scatter

    @functools.wraps(orig_scatter)
    def wrapped_scatter(tensor, scatter_list=None, src=0, group=None, async_op=False):
        # Get the group ranks
        if group is None:
            ranks = list(range(dist.get_world_size()))
        else:
            ranks = _GROUP_TO_RANKS.get(group, [])

        # Log the scatter call
        log_event(
            {
                "op": "scatter",
                "call_id": get_next_call_id(),
                "ranks": ranks,
                "tensor_shape": list(tensor.shape),
                "scatter_list_shapes": [list(t.shape) for t in scatter_list]
                if scatter_list
                else None,
                "dtype": str(tensor.dtype),
                "src": src,
            }
        )

        # Return a dummy future if async
        if async_op:

            class DummyFuture:
                def wait(self):
                    pass

            return DummyFuture()

        return None

    dist.scatter = wrapped_scatter


def _patch_dist_gather():
    """Patch torch.distributed.gather."""
    orig_gather = dist.gather
    _ORIGINAL_DIST_FUNCS["gather"] = orig_gather

    @functools.wraps(orig_gather)
    def wrapped_gather(tensor, gather_list=None, dst=0, group=None, async_op=False):
        # Get the group ranks
        if group is None:
            ranks = list(range(dist.get_world_size()))
        else:
            ranks = _GROUP_TO_RANKS.get(group, [])

        # Log the gather call
        log_event(
            {
                "op": "gather",
                "call_id": get_next_call_id(),
                "ranks": ranks,
                "tensor_shape": list(tensor.shape),
                "gather_list_shapes": [list(t.shape) for t in gather_list] if gather_list else None,
                "dtype": str(tensor.dtype),
                "dst": dst,
            }
        )

        # Return a dummy future if async
        if async_op:

            class DummyFuture:
                def wait(self):
                    pass

            return DummyFuture()

        return None

    dist.gather = wrapped_gather


def _patch_dist_barrier():
    """Patch torch.distributed.barrier."""
    orig_barrier = dist.barrier
    _ORIGINAL_DIST_FUNCS["barrier"] = orig_barrier

    @functools.wraps(orig_barrier)
    def wrapped_barrier(group=None, async_op=False):
        # Get the group ranks
        if group is None:
            ranks = list(range(dist.get_world_size()))
        else:
            ranks = _GROUP_TO_RANKS.get(group, [])

        # Log the barrier call
        log_event(
            {
                "op": "barrier",
                "call_id": get_next_call_id(),
                "ranks": ranks,
            }
        )

        # Return a dummy future if async
        if async_op:

            class DummyFuture:
                def wait(self):
                    pass

            return DummyFuture()

        return None

    dist.barrier = wrapped_barrier


def _patch_dist_send():
    """Patch torch.distributed.send."""
    orig_send = dist.send
    _ORIGINAL_DIST_FUNCS["send"] = orig_send

    @functools.wraps(orig_send)
    def wrapped_send(tensor, dst, group=None, tag=0):
        # Get the group ranks
        if group is None:
            ranks = list(range(dist.get_world_size()))
        else:
            ranks = _GROUP_TO_RANKS.get(group, [])

        # Log the send call
        log_event(
            {
                "op": "send",
                "call_id": get_next_call_id(),
                "ranks": ranks,
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "dst": dst,
                "tag": tag,
            }
        )

        return None

    dist.send = wrapped_send


def _patch_dist_recv():
    """Patch torch.distributed.recv."""
    orig_recv = dist.recv
    _ORIGINAL_DIST_FUNCS["recv"] = orig_recv

    @functools.wraps(orig_recv)
    def wrapped_recv(tensor, src=None, group=None, tag=0):
        # Get the group ranks
        if group is None:
            ranks = list(range(dist.get_world_size()))
        else:
            ranks = _GROUP_TO_RANKS.get(group, [])

        # Log the recv call
        log_event(
            {
                "op": "recv",
                "call_id": get_next_call_id(),
                "ranks": ranks,
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "src": src,
                "tag": tag,
            }
        )

        return None

    dist.recv = wrapped_recv


def _patch_dist_isend():
    """Patch torch.distributed.isend."""
    orig_isend = dist.isend
    _ORIGINAL_DIST_FUNCS["isend"] = orig_isend

    @functools.wraps(orig_isend)
    def wrapped_isend(tensor, dst, group=None, tag=0):
        # Get the group ranks
        if group is None:
            ranks = list(range(dist.get_world_size()))
        else:
            ranks = _GROUP_TO_RANKS.get(group, [])

        # Log the isend call
        log_event(
            {
                "op": "isend",
                "call_id": get_next_call_id(),
                "ranks": ranks,
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "dst": dst,
                "tag": tag,
            }
        )

        # Return a dummy future
        class DummyFuture:
            def wait(self):
                pass

        return DummyFuture()

    dist.isend = wrapped_isend


def _patch_dist_irecv():
    """Patch torch.distributed.irecv."""
    orig_irecv = dist.irecv
    _ORIGINAL_DIST_FUNCS["irecv"] = orig_irecv

    @functools.wraps(orig_irecv)
    def wrapped_irecv(tensor, src=None, group=None, tag=0):
        # Get the group ranks
        if group is None:
            ranks = list(range(dist.get_world_size()))
        else:
            ranks = _GROUP_TO_RANKS.get(group, [])

        # Log the irecv call
        log_event(
            {
                "op": "irecv",
                "call_id": get_next_call_id(),
                "ranks": ranks,
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "src": src,
                "tag": tag,
            }
        )

        # Return a dummy future
        class DummyFuture:
            def wait(self):
                pass

        return DummyFuture()

    dist.irecv = wrapped_irecv


def _patch_dist_all_to_all():
    """Patch torch.distributed.all_to_all."""
    if hasattr(dist, "all_to_all"):
        orig_all_to_all = dist.all_to_all
        _ORIGINAL_DIST_FUNCS["all_to_all"] = orig_all_to_all

        @functools.wraps(orig_all_to_all)
        def wrapped_all_to_all(output_tensor_list, input_tensor_list, group=None, async_op=False):
            # Get the group ranks
            if group is None:
                ranks = list(range(dist.get_world_size()))
            else:
                ranks = _GROUP_TO_RANKS.get(group, [])

            # Log the all_to_all call
            log_event(
                {
                    "op": "all_to_all",
                    "call_id": get_next_call_id(),
                    "ranks": ranks,
                    "output_shapes": [list(t.shape) for t in output_tensor_list],
                    "input_shapes": [list(t.shape) for t in input_tensor_list],
                    "dtype": str(output_tensor_list[0].dtype),
                }
            )

            # Return a dummy future if async
            if async_op:

                class DummyFuture:
                    def wait(self):
                        pass

                return DummyFuture()

            return None

        dist.all_to_all = wrapped_all_to_all


def _patch_dist_batch_isend_irecv():
    """Patch torch.distributed.batch_isend_irecv."""
    if hasattr(dist, "batch_isend_irecv"):
        orig_batch_isend_irecv = dist.batch_isend_irecv
        _ORIGINAL_DIST_FUNCS["batch_isend_irecv"] = orig_batch_isend_irecv

        @functools.wraps(orig_batch_isend_irecv)
        def wrapped_batch_isend_irecv(p2p_op_list):
            # Log each p2p operation
            for i, op in enumerate(p2p_op_list):
                op_type = "isend" if op.op == dist.isend else "irecv"
                tensor = op.tensor
                peer = op.peer

                log_event(
                    {
                        "op": f"batch_{op_type}",
                        "call_id": get_next_call_id(),
                        "batch_idx": i,
                        "batch_size": len(p2p_op_list),
                        "shape": list(tensor.shape),
                        "dtype": str(tensor.dtype),
                        "peer": peer,
                    }
                )

            # Return dummy futures
            class DummyFuture:
                def wait(self):
                    pass

            return [DummyFuture() for _ in range(len(p2p_op_list))]

        dist.batch_isend_irecv = wrapped_batch_isend_irecv


def _patch_dist_all_gather_base():
    """Patch _all_gather_base."""
    if hasattr(dist, "_all_gather_base"):
        orig_all_gather_base = dist._all_gather_base
        _ORIGINAL_DIST_FUNCS["_all_gather_base"] = orig_all_gather_base

        @functools.wraps(orig_all_gather_base)
        def wrapped_all_gather_base(output_tensor, input_tensor, group=None):
            # Get the group ranks
            if group is None:
                ranks = list(range(dist.get_world_size()))
            else:
                ranks = _GROUP_TO_RANKS.get(group, [])

            # Log the all_gather_base call
            log_event(
                {
                    "op": "_all_gather_base",
                    "call_id": get_next_call_id(),
                    "ranks": ranks,
                    "output_shape": list(output_tensor.shape),
                    "input_shape": list(input_tensor.shape),
                    "dtype": str(input_tensor.dtype),
                }
            )

            return None

        dist._all_gather_base = wrapped_all_gather_base


def _patch_dist_reduce_scatter_base():
    """Patch torch.distributed.reduce_scatter_base."""
    if not hasattr(dist, "reduce_scatter_base"):
        return  # Function might not exist in older PyTorch versions
    orig_reduce_scatter_base = dist.reduce_scatter_base
    _ORIGINAL_DIST_FUNCS["reduce_scatter_base"] = orig_reduce_scatter_base

    @functools.wraps(orig_reduce_scatter_base)
    def wrapped_reduce_scatter_base(output_tensor, input_tensor, op=dist.ReduceOp.SUM, group=None):
        # Get the group ranks
        if group is None:
            ranks = list(range(dist.get_world_size()))
        else:
            ranks = _GROUP_TO_RANKS.get(group, [])

        # Log the reduce_scatter_base call
        log_event(
            {
                "op": "reduce_scatter_base",
                "call_id": get_next_call_id(),
                "ranks": ranks,
                "input_shape": list(input_tensor.shape),
                "output_shape": list(output_tensor.shape),
                "dtype": str(input_tensor.dtype),
                "reduce_op": str(op),
            }
        )

        # Perform a dummy copy for meta tensors
        if output_tensor.is_meta:
            output_tensor.copy_(torch.empty_like(output_tensor))

        return None  # Return None as it's synchronous

    dist.reduce_scatter_base = wrapped_reduce_scatter_base


def _patch_coalescing_manager():
    """Patch torch.distributed.distributed_c10d._coalescing_manager."""
    try:
        from torch.distributed import distributed_c10d

        orig_manager = distributed_c10d._coalescing_manager
        _ORIGINAL_DIST_FUNCS["_coalescing_manager"] = orig_manager

        @functools.wraps(orig_manager)
        @contextlib.contextmanager
        def wrapped_coalescing_manager(group, async_ops=False, device=None):
            # We don't need the actual coalescing logic for logging
            # The context manager protocol still needs to be followed (yield)
            # The original manager creates a _CoalescingManager object, we can skip that.
            yield None  # Yield None or a dummy object if needed by the caller
            # The critical part is the finally block in the original manager.
            # We need to replicate its structure safely.
            try:
                # Simulate the potential assignment of 'work' inside the original finally block
                # Since our patches return DummyFutures or None, the actual work object
                # from pg_coalesce_state.wait() won't be created.
                # So, 'work' inside the original finally would likely remain None or unassigned.
                work = None  # Explicitly set to None to avoid UnboundLocalError
            finally:
                # Simulate the work.wait() call safely
                if work is not None:
                    # In a real scenario, wait would be called. Since work is None,
                    # this part is effectively skipped, preventing the error.
                    # work.wait() # This line is now safe due to the check
                    pass

        dist._coalescing_manager = wrapped_coalescing_manager
    except (ImportError, AttributeError):
        print("Warning: Could not patch _coalescing_manager.")


def _patch_dist_P2POp():
    """Patch torch.distributed.P2POp to return a dummy object storing op info."""
    orig_P2POp = dist.P2POp
    _ORIGINAL_DIST_FUNCS["P2POp"] = orig_P2POp

    @functools.wraps(orig_P2POp)
    def wrapped_P2POp(op, tensor, peer, group=None, tag=0):
        # Determine the original operation type for logging/analysis if needed
        if op.__name__ == "wrapped_isend" and "isend" in _ORIGINAL_DIST_FUNCS:
            # Store the original function reference if needed for analysis, but don't use it for execution
            actual_op = _ORIGINAL_DIST_FUNCS["isend"]
        elif op.__name__ == "wrapped_irecv" and "irecv" in _ORIGINAL_DIST_FUNCS:
            actual_op = _ORIGINAL_DIST_FUNCS["irecv"]
        else:
            actual_op = op  # Fallback, might be an unpatched op

        # Return a dummy object (SimpleNamespace) instead of a real P2POp instance
        dummy_p2p_op = types.SimpleNamespace(
            op=actual_op,  # Store the intended operation (e.g., original isend/irecv)
            tensor=tensor,
            peer=peer,
            group=group,
            tag=tag,
        )
        return dummy_p2p_op

    # Replace the P2POp constructor
    dist.P2POp = wrapped_P2POp


def apply_dist_comm_patches(verbose=False):
    """
    Apply all patches for distributed communication logging.

    Args:
        verbose: If True, print information about applied patches
    """
    # Initialize the list of ranks for the default process group
    _GROUP_TO_RANKS[None] = list(range(dist.get_world_size())) if dist.is_initialized() else []

    # Apply all patches
    _patch_dist_init()
    _patch_dist_new_group()
    _patch_dist_all_reduce()
    _patch_dist_all_gather()
    _patch_dist_broadcast()
    _patch_dist_reduce_scatter()
    _patch_dist_scatter()
    _patch_dist_gather()
    _patch_dist_barrier()
    _patch_dist_send()
    _patch_dist_recv()
    _patch_dist_isend()
    _patch_dist_irecv()
    _patch_dist_P2POp()  # Added patch for P2POp

    # Patch all_to_all if available
    if hasattr(dist, "all_to_all"):
        _patch_dist_all_to_all()

    # Patch batch_isend_irecv if available
    if hasattr(dist, "batch_isend_irecv"):
        _patch_dist_batch_isend_irecv()

    # Patch all_gather_base if available
    if hasattr(dist, "all_gather_base"):
        _patch_dist_all_gather_base()

    # Patch reduce_scatter_base if available
    if hasattr(dist, "reduce_scatter_base"):
        _patch_dist_reduce_scatter_base()

    _patch_coalescing_manager()  # Add the new patch call

    if verbose:
        print("Applied distributed communication patches")


def remove_dist_comm_patches():
    """Remove all patches for distributed communication logging."""
    # Restore original functions
    for func_name, orig_func in _ORIGINAL_DIST_FUNCS.items():
        if hasattr(dist, func_name):
            setattr(dist, func_name, orig_func)
        elif func_name == "_coalescing_manager":  # Handle coalescing manager separately
            try:
                from torch.distributed import distributed_c10d

                setattr(distributed_c10d, func_name, orig_func)
            except (ImportError, AttributeError):
                pass  # Ignore if it couldn't be patched initially

    _ORIGINAL_DIST_FUNCS.clear()
    _GROUP_TO_RANKS.clear()
    # Reset counters and logs if needed
    global _CALL_ID_COUNTER
    _CALL_ID_COUNTER = 0
    clear_comm_logs()


# Additional functions needed by __init__.py imports


class CommLogger:
    """Class for logging and analyzing communication patterns."""

    def __init__(self):
        """Initialize the communication logger."""
        self.logs = []

    def capture(self):
        """Start capturing communication events."""
        apply_dist_comm_patches()
        clear_comm_logs()

    def stop(self):
        """Stop capturing communication events and collect logs."""
        self.logs = get_comm_logs()
        remove_dist_comm_patches()

    def analyze(self):
        """Analyze collected communication patterns."""
        if not self.logs:
            return {"message": "No communication logs collected"}

        op_counts = {}
        total_bytes = 0

        for event in self.logs:
            op = event.get("op", "unknown")
            op_counts[op] = op_counts.get(op, 0) + 1

            # Estimate bytes if possible
            if "shape" in event and "dtype" in event:
                shape = event["shape"]
                dtype = event["dtype"]
                element_count = 1
                for dim in shape:
                    element_count *= dim

                # Rough estimation of bytes based on dtype
                bytes_per_element = 4  # Default to float32
                if "float16" in dtype or "half" in dtype:
                    bytes_per_element = 2
                elif "float64" in dtype or "double" in dtype:
                    bytes_per_element = 8
                elif "int64" in dtype or "long" in dtype:
                    bytes_per_element = 8

                total_bytes += element_count * bytes_per_element

        return {
            "op_counts": op_counts,
            "total_events": len(self.logs),
            "estimated_bytes": total_bytes,
        }


def log_communication(tensor, op_type, src=None, dst=None):
    """
    Manually log a communication event.

    Args:
        tensor: The tensor being communicated
        op_type: Type of communication operation
        src: Source rank (if applicable)
        dst: Destination rank (if applicable)
    """
    event = {
        "op": op_type,
        "call_id": get_next_call_id(),
        "shape": list(tensor.shape) if hasattr(tensor, "shape") else None,
        "dtype": str(tensor.dtype) if hasattr(tensor, "dtype") else None,
    }

    if src is not None:
        event["src"] = src

    if dst is not None:
        event["dst"] = dst

    log_event(event)


def get_comm_stats():
    """
    Get statistics about recorded communication events.

    Returns:
        Dictionary with communication statistics
    """
    logs = get_comm_logs()

    if not logs:
        return {"message": "No communication logs collected"}

    stats = {
        "total_events": len(logs),
        "op_counts": {},
        "total_bytes": 0,
    }

    for event in logs:
        op = event.get("op", "unknown")
        stats["op_counts"][op] = stats["op_counts"].get(op, 0) + 1

        # Estimate bytes if possible
        if "shape" in event and "dtype" in event:
            shape = event["shape"]
            if shape:
                dtype = event["dtype"]
                element_count = 1
                for dim in shape:
                    element_count *= dim

                # Rough estimation of bytes based on dtype
                bytes_per_element = 4  # Default to float32
                if "float16" in dtype or "half" in dtype:
                    bytes_per_element = 2
                elif "float64" in dtype or "double" in dtype:
                    bytes_per_element = 8
                elif "int64" in dtype or "long" in dtype:
                    bytes_per_element = 8

                stats["total_bytes"] += element_count * bytes_per_element

    return stats


def clear_comm_stats():
    """Clear all communication statistics."""
    clear_comm_logs()
