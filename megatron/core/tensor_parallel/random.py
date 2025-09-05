# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2025, The Board of Trustees of the Leland Stanford Junior University.
# All rights reserved.

# Parts of the code here are adapted from PyTorch
# repo: https://github.com/pytorch/pytorch

import contextlib
import logging
from typing import Union

import torch
from torch import _C
from torch.cuda import _lazy_call, _lazy_init
from torch.cuda import device as device_ctx_manager
from torch.utils.checkpoint import detach_variable

from megatron.core.parallel_state import (
    get_expert_model_parallel_rank,
    get_expert_tensor_parallel_rank,
    get_tensor_model_parallel_rank,
)
from megatron.core.utils import is_te_min_version, safely_set_viewless_tensor_data

from .utils import gather_split_1d_tensor, split_tensor_into_1d_equal_chunks

try:
    import transformer_engine  # pylint: disable=unused-import

    HAVE_TE = True
except ModuleNotFoundError:
    HAVE_TE = False


# Default name for the model parallel rng tracker.
_MODEL_PARALLEL_RNG_TRACKER_NAME = "model-parallel-rng"
_EXPERT_PARALLEL_RNG_TRACKER_NAME = "expert-parallel-rng"
_DATA_PARALLEL_RNG_TRACKER_NAME = "data-parallel-rng"


def _get_cuda_rng_state(
    device: Union[int, str, torch.device] = "cuda",
    clone: bool = False,
    graph_safe: bool = False,
) -> torch.Tensor:
    """Return the random number generator state of the specified GPU.

    Arguments:
        device (int): The gpu to retrieve the rng state
        clone (bool): Whether to also clone the retrieved RNG state
        graph_safe (bool): Get the rng state in a graph safe manner.

    This function is adapted from torch.cuda.random.get_rng_state()"""

    # Check if CUDA is available, fallback to CPU if not
    if not torch.cuda.is_available():
        return torch.get_rng_state()

    # if not using cuda graphs, just use the builtin pytorch function
    if not graph_safe:
        return torch.cuda.random.get_rng_state(device=device)

    _lazy_init()
    if isinstance(device, str):
        device = torch.device(device)
    elif isinstance(device, int):
        device = torch.device("cuda", device)
    idx = device.index
    if idx is None:
        idx = torch.cuda.current_device()

    default_generator = torch.cuda.default_generators[idx]
    if clone:
        return default_generator.clone_state()
    return default_generator.graphsafe_get_state()


def _set_cuda_rng_state(new_state: torch.Tensor, device: int = -1, graph_safe: bool = False):
    """Sets the random number generator state of the current GPU.

    Arguments:
        new_state (torch.ByteTensor): The desired state
        device (int): The gpu to retrieve the rng state
        graph_safe (bool): Set the rng state in a graph safe manner.

    This function is adapted from PyTorch repo (torch.cuda.set_rng_state)
    with a single change: the input state is not cloned. Cloning caused
    major performance issues for +4 GPU cases.
    """
    # Check if CUDA is available, fallback to CPU if not
    if not torch.cuda.is_available():
        torch.set_rng_state(new_state)
        return

    if hasattr(_C, "_cuda_setRNGState") and callable(_C._cuda_setRNGState):
        # older PyTorch
        def cb():
            with device_ctx_manager(device):
                _C._cuda_setRNGState(new_state)

    else:
        # newer PyTorch
        if device == -1:
            device = torch.device("cuda")
        elif isinstance(device, str):
            device = torch.device(device)
        elif isinstance(device, int):
            device = torch.device("cuda", device)

        def cb():
            idx = device.index
            if idx is None:
                idx = torch.cuda.current_device()
            default_generator = torch.cuda.default_generators[idx]

            # if graph capturing, set the rng state in a cudagraphable way
            if graph_safe:
                default_generator.graphsafe_set_state(new_state)
            else:
                default_generator.set_state(new_state)

    _lazy_call(cb)


def get_expert_parallel_rng_tracker_name():
    """Get the expert parallel rng tracker name"""
    global _EXPERT_PARALLEL_RNG_TRACKER_NAME
    return _EXPERT_PARALLEL_RNG_TRACKER_NAME


def get_data_parallel_rng_tracker_name():
    """Get the data parallel rng tracker name"""
    global _DATA_PARALLEL_RNG_TRACKER_NAME
    return _DATA_PARALLEL_RNG_TRACKER_NAME


class CudaRNGStatesTracker:
    """Tracker for the cuda RNG states.

    Using the `add` method, a cuda rng state is initialized based on
    the input `seed` and is assigned to `name`. Later, by forking the
    rng state, we can perform operations and return to our starting
    cuda state.
    """

    def __init__(self, use_cudagraphable_rng=False, is_inference_rng_tracker=False):
        self.reset()
        self.use_cudagraphable_rng = use_cudagraphable_rng
        self.is_inference_rng_tracker = is_inference_rng_tracker
        # Check if CUDA is available
        self.has_cuda = torch.cuda.is_available()

        if self.has_cuda and self.use_cudagraphable_rng:
            assert (
                hasattr(torch.cuda.CUDAGraph, "register_generator_state")
                and hasattr(torch.Generator, "graphsafe_set_state")
                and hasattr(torch.Generator, "graphsafe_get_state")
                and hasattr(torch.Generator, "clone_state")
            ), "Tried using cudagraphs with RNG, however not detected in pytorch!"

    def is_initialized(self):
        """Checks if the internal RNG state has been set wirth set_states()."""
        return self._is_initialized

    def reset(self):
        """Set to the initial state (no tracker)."""

        # Track if initialized.
        self._is_initialized = False

        # Map from a string name to the cuda rng state.
        self.states_ = {}

        # Seeds are just for book keeping and ensure no seed is set twice.
        self.seeds_ = set()

    def get_states(self):
        """Get rng states. Copy the dictionary so we have direct
        pointers to the states, not just a pointer to the dictionary."""
        states = {}
        for name in self.states_:
            states[name] = self.states_[name]
        return states

    def set_states(self, states):
        """Set the rng states. For efficiency purposes, we do not check
        the size of seed for compatibility."""
        self._is_initialized = True
        self.states_ = states

    def add(self, name, seed):
        """Track the rng state."""
        self._is_initialized = True
        # Check seed is not already used.
        if seed in self.seeds_:
            raise Exception("seed {} already exists".format(seed))
        self.seeds_.add(seed)
        # Check that state is not already defined.
        if name in self.states_:
            raise Exception("cuda rng state {} already exists".format(name))

        # If CUDA is not available, use CPU RNG
        if not self.has_cuda:
            # Save original CPU state
            orig_rng_state = torch.get_rng_state()
            # Set seed
            torch.manual_seed(seed)
            # Store new state
            self.states_[name] = torch.get_rng_state()
            # Restore original state
            torch.set_rng_state(orig_rng_state)
            return

        # If available, create the state in a graph safe manner
        if self.use_cudagraphable_rng:
            new_state = _get_cuda_rng_state(clone=True, graph_safe=True)
            new_state.manual_seed(seed)
            self.states_[name] = new_state
        else:
            # Get the current rng state.
            orig_rng_state = torch.cuda.get_rng_state()
            # Set the new state and store it.
            torch.cuda.manual_seed(seed)
            self.states_[name] = torch.cuda.get_rng_state()
            # Reset rng state to what it was.
            _set_cuda_rng_state(orig_rng_state)

    @contextlib.contextmanager
    def fork(self, name=_MODEL_PARALLEL_RNG_TRACKER_NAME):
        """Fork the cuda rng state, perform operations, and exit with
        the original state."""
        # Check if we have added the state
        if name not in self.states_:
            raise Exception("cuda rng state {} is not added".format(name))

        # Handle CPU case when CUDA is not available
        if not self.has_cuda:
            # Store current CPU RNG state
            orig_cpu_rng_state = torch.get_rng_state()
            # Set RNG state to the desired one
            torch.set_rng_state(self.states_[name])
            # Record CPU RNG state for warning check
            init_cpu_rng_state = torch.get_rng_state()

            try:
                yield
            finally:
                # Throw a warning if CPU RNG state changed
                if not torch.all(init_cpu_rng_state == torch.get_rng_state()).item():
                    logging.getLogger(__name__).warning(
                        "CPU RNG state changed within GPU RNG context"
                    )
                # Update the current rng state for later use
                self.states_[name] = torch.get_rng_state()
                # And set the state to the original state we started with
                torch.set_rng_state(orig_cpu_rng_state)
            return

        # For CUDA case, continue with original implementation
        # Store current rng state.
        orig_cuda_rng_state = _get_cuda_rng_state(graph_safe=self.use_cudagraphable_rng)
        # Set rng state to the desired one
        _set_cuda_rng_state(self.states_[name], graph_safe=self.use_cudagraphable_rng)
        # Record cpu RNG state
        cpu_rng_state = torch.get_rng_state()
        # Do the stuff we wanted to do.
        try:
            yield
        finally:
            # Throw a warning if cpu RNG state changed
            if not torch.all(cpu_rng_state == torch.get_rng_state()).item():
                logging.getLogger(__name__).warning("CPU RNG state changed within GPU RNG context")
            # Update the current rng state for later use.
            self.states_[name] = _get_cuda_rng_state(graph_safe=self.use_cudagraphable_rng)
            # And set the state to the original state we started with.
            _set_cuda_rng_state(orig_cuda_rng_state, graph_safe=self.use_cudagraphable_rng)


# RNG tracker object.
_CUDA_RNG_STATE_TRACKER = None
_CUDA_RNG_STATE_TRACKER_INITIALIZED: bool = False


# Add a MockRNGTracker class
class MockRNGTracker:
    """A mock RNG tracker that does nothing, used for meta tensor analysis."""

    def add(self, name, seed):
        pass

    def get_states(self):
        return {}

    def set_states(self, states):
        pass

    @contextlib.contextmanager
    def fork(self, name=None):
        yield


def initialize_rng_tracker(
    use_te_rng_tracker: bool = False,
    inference_rng_tracker: bool = False,
    use_cudagraphable_rng: bool = False,
    force_reset: bool = False,
):
    """Initialize the random number generator state tracker.

    Used for tensor model parallel replicas.
    """
    global _CUDA_RNG_STATE_TRACKER
    global _CUDA_RNG_STATE_TRACKER_INITIALIZED

    if not force_reset and _CUDA_RNG_STATE_TRACKER_INITIALIZED:
        return

    # Use a mock tracker for meta tensor analysis to avoid CUDA dependency
    _CUDA_RNG_STATE_TRACKER = MockRNGTracker()
    _CUDA_RNG_STATE_TRACKER_INITIALIZED = True

    ''' # Original CUDA/TE tracker initialization logic - disabled
    base_tracker = None
    if HAVE_TE and use_te_rng_tracker:
        if not is_te_min_version("1.5.0"):
            raise RuntimeError(
                "use_te_rng_tracker requires TransformerEngine version >= 1.5"
            )
        from megatron.core.extensions.transformer_engine import TECudaRNGStatesTracker

        base_tracker = TECudaRNGStatesTracker
        tracker_kwargs = {"is_inference_rng_tracker": inference_rng_tracker}
    else:
        base_tracker = CudaRNGStatesTracker
        tracker_kwargs = {
            "use_cudagraphable_rng": use_cudagraphable_rng,
            "is_inference_rng_tracker": inference_rng_tracker,
        }

    if inference_rng_tracker:

        class InferenceCudaRNGStatesTracker(base_tracker):  # type: ignore[valid-type, misc]
            """RNG tracker for inference."""

            def add(self, name, seed):
                """Mirrors the interface from the training RNG tracker."""
                pass

            def set_states(self, states):
                """Mirrors the interface from the training RNG tracker."""
                pass

            def fork(self, name=_MODEL_PARALLEL_RNG_TRACKER_NAME):
                """Mirrors the interface from the training RNG tracker."""
                return contextlib.nullcontext()

        tracker_class = InferenceCudaRNGStatesTracker
    else:
        tracker_class = base_tracker

    _CUDA_RNG_STATE_TRACKER = tracker_class(**tracker_kwargs)
    _CUDA_RNG_STATE_TRACKER_INITIALIZED = True
    '''  # End of disabled section


def get_cuda_rng_tracker(
    use_te_rng_tracker: bool = False,
    inference_rng_tracker: bool = False,
    use_cudagraphable_rng: bool = False,
):
    """Get cuda rng tracker."""
    initialize_rng_tracker(use_te_rng_tracker, inference_rng_tracker, use_cudagraphable_rng)
    return _CUDA_RNG_STATE_TRACKER


def get_all_rng_states():
    """Returns all generator states used by the current `CudaRNGStatesTracker`."""

    assert _CUDA_RNG_STATE_TRACKER_INITIALIZED, (
        "Tried getting all rng states but RNG Tracker has not been initalized!"
    )

    # Handle the MockRNGTracker case
    if isinstance(_CUDA_RNG_STATE_TRACKER, MockRNGTracker):
        return _CUDA_RNG_STATE_TRACKER.get_states()

    if isinstance(_CUDA_RNG_STATE_TRACKER, CudaRNGStatesTracker):
        return _CUDA_RNG_STATE_TRACKER.states_
    # If TE is installed, check if we are using TE's RNG tracker
    elif HAVE_TE and is_te_min_version("1.5.0"):
        from megatron.core.extensions.transformer_engine import TECudaRNGStatesTracker

        if isinstance(_CUDA_RNG_STATE_TRACKER, TECudaRNGStatesTracker):
            from transformer_engine.pytorch.distributed import get_all_rng_states

            return get_all_rng_states()
    # no valid tracker, return an empty dict
    else:
        return {}


def model_parallel_cuda_manual_seed(
    seed: int,
    te_rng_tracker: bool = False,
    inference_rng_tracker: bool = False,
    use_cudagraphable_rng: bool = False,
    tp_rank: int = None,
    ep_rank: int = None,
    etp_rank: int = None,
):
    """Initialize model parallel RNG.

    This function should be called after the model parallel is
    initialized. Also, no torch.manual_seed() should be called
    after this function. Basically, this is replacement for that
    function.
    Two set of RNG states are tracked:
        default state: This is for data parallelism and is the same
            across all the model parallel ranks.
        model-parallel state: This is for operations that are different
            across the model parallel ranks.
    """
    # 2023-optimizations: We avoid needing to store/replicate this state
    # by having each rank initialize their own data. If we're using a known
    # seed, each rank will have the same final parameters.
    # We can override the tp/ep/etp ranks for testing.
    if tp_rank is None:
        tp_rank = get_tensor_model_parallel_rank()
    if ep_rank is None:
        ep_rank = get_expert_model_parallel_rank()
    if etp_rank is None:
        etp_rank = get_expert_tensor_parallel_rank()

    # Check if CUDA is available
    has_cuda = torch.cuda.is_available()

    # Data parallel uses the default RNG
    if has_cuda:
        torch.cuda.manual_seed(seed)
    # CPU fallback
    torch.manual_seed(seed)

    # Make sure to initialize the tracker before adding states
    initialize_rng_tracker(
        use_te_rng_tracker=te_rng_tracker,
        inference_rng_tracker=inference_rng_tracker,
        use_cudagraphable_rng=use_cudagraphable_rng,
    )

    # Use PCG for CPU, and Philox for GPU.
    default_rng = get_cuda_rng_tracker(
        use_te_rng_tracker=te_rng_tracker,
        inference_rng_tracker=inference_rng_tracker,
        use_cudagraphable_rng=use_cudagraphable_rng,
    )
    default_rng.add(_DATA_PARALLEL_RNG_TRACKER_NAME, seed)
    default_rng.add(_MODEL_PARALLEL_RNG_TRACKER_NAME, seed + 1 + tp_rank)
    default_rng.add(_EXPERT_PARALLEL_RNG_TRACKER_NAME, seed + 1 + tp_rank + ep_rank + etp_rank)


def _get_all_rng_states():
    """Get all the rng states."""
    cpu_rng_state = torch.get_rng_state()
    cuda_rng_state = _get_cuda_rng_state()
    cuda_rng_state_tracker = get_cuda_rng_tracker().get_states()
    return cpu_rng_state, cuda_rng_state, cuda_rng_state_tracker


def _set_all_rng_states(cpu_rng_state, cuda_rng_state, cuda_rng_state_tracker):
    """Set all the rng states."""
    torch.set_rng_state(cpu_rng_state)
    _set_cuda_rng_state(cuda_rng_state)
    get_cuda_rng_tracker().set_states(cuda_rng_state_tracker)


@contextlib.contextmanager
def _fork_rng():
    """Fork the rng state."""
    # Store the current states.
    current_states = _get_all_rng_states()
    try:
        yield
    finally:
        # Set the states back to what it was at the start of this function.
        _set_all_rng_states(*current_states)


class CheckpointFunction(torch.autograd.Function):
    """Checkpoint Function

    This function is adapted from torch.utils.checkpoint with two main changes:
    1) torch.cuda.set_rng_state is replaced with `_set_cuda_rng_state`
    2) the states in the model parallel tracker are also properly tracked/set/reset.
    """

    # pylint: disable=missing-function-docstring
    @staticmethod
    def forward(ctx, run_function, distribute_saved_activations, *args):
        """Forward pass."""
        ctx.run_function = run_function
        ctx.distribute_saved_activations = distribute_saved_activations

        # Copy the rng states.
        ctx.rng_states = _get_all_rng_states()

        with torch.no_grad():
            outputs = run_function(*args)

        # Divide hidden states across model parallel group and only keep
        # the chunk corresponding to the current rank.
        if distribute_saved_activations:
            ctx.input_0_shape = args[0].data.shape
            safely_set_viewless_tensor_data(
                args[0],
                split_tensor_into_1d_equal_chunks(args[0].data, new_buffer=True),
            )

        # Store everything.
        ctx.save_for_backward(*args)

        return outputs

    # pylint: disable=missing-function-docstring
    @staticmethod
    def backward(ctx, *args):
        """Backward pass."""
        if not torch.autograd._is_checkpoint_valid():
            raise RuntimeError(
                "Checkpointing is not compatible with .grad(), please use .backward() if possible"
            )
        inputs = ctx.saved_tensors
        if ctx.distribute_saved_activations:
            safely_set_viewless_tensor_data(
                inputs[0],
                gather_split_1d_tensor(inputs[0].data).view(ctx.input_0_shape),
            )

        with _fork_rng():
            # Set the states to what it used to be before the forward pass.
            _set_all_rng_states(*ctx.rng_states)

            # Compute the forward pass.
            detached_inputs = detach_variable(inputs)
            with torch.enable_grad():
                outputs = ctx.run_function(*detached_inputs)

        if isinstance(outputs, torch.Tensor):
            outputs = (outputs,)

        # filter out non tensor outputs for backward pass
        outputs, args = zip(
            *filter(
                lambda x: torch.is_tensor(x[0]) and x[0].requires_grad,
                zip(outputs, args),
            )
        )
        torch.autograd.backward(outputs, args)
        grads = tuple(inp.grad if isinstance(inp, torch.Tensor) else inp for inp in detached_inputs)
        return (None, None) + grads


def checkpoint(function, distribute_saved_activations, *args):
    """Checkpoint a model or part of the model.
    This has been directly copied from torch.utils.checkpoint."""
    return CheckpointFunction.apply(function, distribute_saved_activations, *args)


class CheckpointWithoutOutputFunction(torch.autograd.Function):
    """
    Checkpoint Function Helper for CheckpointWithouOutput.
    Save context for recompute.
    """

    @staticmethod
    def forward(ctx, run_function, checkpoint_without_output_obj, *args):
        """Forward pass."""
        with torch.no_grad():
            outputs = run_function(*args)
        ctx.save_for_backward(*detach_variable(args))
        # the CheckpointWithoutOutput object is passed in, then it can access the saved input
        # tensors later for recomputation
        checkpoint_without_output_obj.ctx = ctx
        return outputs

    @staticmethod
    def backward(ctx, *args):
        """Backward pass."""
        inputs = ctx.saved_tensors
        outputs = ctx.outputs
        torch.autograd.backward(outputs, args)
        ctx.outputs = None
        grads = tuple(inp.grad if isinstance(inp, torch.Tensor) else inp for inp in inputs)
        return (None, None) + grads


class CheckpointWithoutOutput(object):
    """
    Checkpoint a model or part of the model and release the output.

    For the normal 'checkpoint` function, the outputs of it may be cached by the following
    operations for its backward computation. However, the output of the checkpointed function is
    re-generated at recomputation, so the output store is not technically needed. This method can
    manually discard the output in the forward pass and restore it by recomputation in the
    backward pass to reduce the memory usage.
    """

    def __init__(self):
        self.run_function = None
        self.fwd_cpu_rng_state = None
        self.fwd_cuda_rng_state = None
        self.fwd_cuda_rng_state_tracker = None
        self.ctx = None
        self.outputs = None

    def checkpoint(self, run_function, *args):
        """Checkpoint function."""
        self.run_function = run_function

        self.rng_states = _get_all_rng_states()

        outputs = CheckpointWithoutOutputFunction.apply(run_function, self, *args)
        self.outputs = outputs
        if isinstance(self.outputs, torch.Tensor):
            self.outputs = (self.outputs,)
        return outputs

    def _recompute(self, _):
        """Used as a hook to recompute the output."""
        if not torch.autograd._is_checkpoint_valid():
            raise RuntimeError(
                "Checkpointing is not compatible with .grad(), please use .backward() if possible"
            )

        with _fork_rng():
            _set_all_rng_states(*self.rng_states)

            with torch.enable_grad():
                outputs = self.run_function(*self.ctx.saved_tensors)

        self.run_function = None
        self.rng_states = None

        if isinstance(outputs, torch.Tensor):
            outputs = (outputs,)

        # restore the recomputed memory without changing the metadata
        with torch.no_grad():
            for output, recomputation_output in zip(self.outputs, outputs):
                output_size = recomputation_output.untyped_storage().size()
                output.untyped_storage().resize_(output_size)
                output.untyped_storage().copy_(recomputation_output.untyped_storage())

        self.ctx.outputs = outputs
        self.outputs = None
        self.ctx = None

    def discard_output_and_register_recompute(self, hook_tensor):
        """
        Release the output tensor storages and register the recompute function as a grad hook of
        the hook_tensor.

        Note: the caller should make sure that the output tensors are no longer used
        in the forward pass and the gradient of the hook_tensor is computed before the recomputed
        tensors are used.
        """
        # use resize to release the output tensor memory and still keep the metadata in the tensors.
        # the metadata is still needed for backward
        for output in self.outputs:
            output.untyped_storage().resize_(0)

        # register the recomputation as a backward hook, when the the gradient of the hook_tensor
        # is computed, the recomputation will be triggered. The hook_tensor should be selected
        # carefully to ensure that the tensors are recomputed before it is used by other backward
        # computations.
        if hook_tensor.requires_grad:
            hook_tensor.register_hook(self._recompute)
