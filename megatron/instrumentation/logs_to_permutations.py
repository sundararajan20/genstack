# Copyright (c) 2025, The Board of Trustees of the Leland Stanford Junior University.

# This source code is licensed under the Apache license found in the
# LICENSE file in the root directory of this source tree.

"""
Utility to read a communication log JSON and print operation names excluding
"new_group" events.

For now, this script defaults to the attached log file path used during
development, but you may pass a different path via CLI.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Any, Dict, Iterable, List, Set, Tuple

import numpy as np


def read_events_from_json(json_path: str) -> List[Dict[str, Any]]:
    """Read and return the list of event dicts from the given JSON file.

    Args:
        json_path: Absolute or relative path to the JSON file containing a list of events.

    Returns:
        List of events as dictionaries.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the JSON does not contain a list of dictionaries.
    """
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"JSON file not found: {json_path}")

    with open(json_path, "r") as fp:
        data = json.load(fp)

    if not isinstance(data, list):
        raise ValueError("Expected top-level JSON to be a list of events")

    return data


def extract_rank_numbers_from_paths(log_paths: List[str]) -> List[int]:
    """Extract rank numbers from a list of log file paths.

    Args:
        log_paths: List of full paths to log files like 'path/to/comm_log_rank_0.json'

    Returns:
        List of rank numbers in the same order as input
    """

    def extract_rank_from_path(path: str) -> int:
        """Extract rank number from full path."""
        filename = os.path.basename(path)  # Get just the filename
        match = re.search(r'comm_log_rank_(\d+)\.json', filename)
        if match:
            return int(match.group(1))
        # Fallback: try to extract any number after 'rank_'
        match = re.search(r'rank_(\d+)', filename)
        if match:
            return int(match.group(1))
        # If no rank found, return -1
        return -1

    return [extract_rank_from_path(path) for path in log_paths]


def sort_log_files_by_rank(log_files: List[str]) -> List[str]:
    """Sort log files by the rank number in ascending order.

    Extracts the rank number from filenames like 'comm_log_rank_X.json'
    and sorts them numerically rather than lexicographically.

    Args:
        log_files: List of log filenames

    Returns:
        Sorted list of log filenames
    """

    def extract_rank(filename: str) -> int:
        """Extract rank number from filename like 'comm_log_rank_123.json'."""
        match = re.search(r'comm_log_rank_(\d+)\.json', filename)
        if match:
            return int(match.group(1))
        # Fallback: try to extract any number after 'rank_'
        match = re.search(r'rank_(\d+)', filename)
        if match:
            return int(match.group(1))
        # If no rank found, return 0
        return 0

    return sorted(log_files, key=extract_rank)


def get_sorted_log_files_from_directory(directory_path: str) -> List[str]:
    """Get all log files from a directory's summary.json, sorted by rank.

    Args:
        directory_path: Path to directory containing summary.json

    Returns:
        List of sorted log filenames (relative to directory_path)

    Raises:
        FileNotFoundError: If summary.json not found
        ValueError: If no log files listed in summary.json
    """
    summary_path = os.path.join(directory_path, "summary.json")
    if not os.path.exists(summary_path):
        raise FileNotFoundError(f"summary.json not found in directory: {directory_path}")

    with open(summary_path, "r") as fp:
        summary = json.load(fp)

    log_files = summary.get("log_files")
    if not isinstance(log_files, list) or not log_files:
        raise ValueError(f"No log files listed in summary.json at: {summary_path}")

    return sort_log_files_by_rank(log_files)


def resolve_log_paths_from_input(input_path: str) -> List[str]:
    """Resolve the actual list of log JSON path from a directory or a file path.

    If ``input_path`` is a directory, this function reads ``summary.json`` in that
    directory and returns the full path to the first log file listed there
    (rank 0 after sorting by rank number).
    Otherwise, it assumes ``input_path`` is the direct path to a log JSON file.
    """
    if os.path.isdir(input_path):
        summary_path = os.path.join(input_path, "summary.json")
        if not os.path.exists(summary_path):
            raise FileNotFoundError(f"summary.json not found in directory: {input_path}")

        with open(summary_path, "r") as fp:
            summary = json.load(fp)

        log_files = summary.get("log_files")
        if not isinstance(log_files, list) or not log_files:
            raise ValueError(f"No log files listed in summary.json at: {summary_path}")

        # Sort log files by rank number in ascending order
        sorted_log_files = sort_log_files_by_rank(log_files)
        return [os.path.join(input_path, log_file) for log_file in sorted_log_files]

    # Otherwise, treat as a file path
    return [input_path]


def iter_ops_excluding_new_group(
    events: Iterable[Dict[str, Any]],
) -> Iterable[Tuple[str, List[int], List[int], str, str, int, int]]:
    (
        """Yield operation names for events whose op != 'new_group' """
        """and op != 'init_process_group' and op != 'barrier'."""
    )
    for event in events:
        operation = event.get("op")
        if operation is None:
            continue
        if operation == "new_group" or operation == "init_process_group" or operation == "barrier":
            continue
        ranks = event.get("ranks", None)
        shape = event.get("shape", None)
        if shape is None or shape == []:
            continue
        dtype = event.get("dtype", None)
        reduce_op = event.get("reduce_op", None)
        src_rank = event.get("src_rank", None)
        peer = event.get("peer", None)
        yield operation, ranks, shape, dtype, reduce_op, src_rank, peer


def dtype_size(dtype: str) -> int:
    """Return the size in bytes for a given PyTorch dtype."""
    if dtype == "torch.bool":
        return 1
    elif dtype == "torch.bfloat16":
        return 2
    elif dtype == "torch.float8":
        return 1
    elif dtype == "torch.float16":
        return 2
    elif dtype == "torch.float32":
        return 4
    elif dtype == "torch.float64":
        return 8
    elif dtype == "torch.int8":
        return 1
    elif dtype == "torch.int16":
        return 2
    elif dtype == "torch.int32":
        return 4
    elif dtype == "torch.int64":
        return 8
    elif dtype == "torch.uint8":
        return 1
    elif dtype == "torch.uint16":
        return 2
    elif dtype == "torch.uint32":
        return 4
    elif dtype == "torch.uint64":
        return 8
    else:
        raise ValueError(f"Unknown dtype: {dtype}")


def shape_product(shape: List[int]) -> int:
    """Calculate the product of all dimensions in a shape."""
    return int(np.prod(shape))


def process_events(
    resolved_log_paths: List[str],
    events: Dict[str, List[Tuple[str, List[int], List[int], str, str, int, int]]],
    dp_size: int,
    tp_size: int,
    pp_size: int,
    seq_len: int,
    hidden_size: int,
    num_micro_batches: int,
    global_batch_size: int,
) -> Tuple[List[Tuple[int, Tuple[Tuple[int, int], ...]]], Set[Tuple[Tuple[int, int], ...]]]:
    """Process communication events and extract communication patterns."""
    ranks = extract_rank_numbers_from_paths(resolved_log_paths)

    permutations: Set[Tuple[Tuple[int, int], ...]] = set()
    weighted_permutations: List[Tuple[int, Tuple[Tuple[int, int], ...]]] = []

    world_size = dp_size * tp_size * pp_size
    micro_batch_size = global_batch_size // num_micro_batches // dp_size
    log_idx = {rank: 0 for rank in ranks}

    # FIRST:get the pre-tensor parallelism events first
    # the initial full size all_reduce with the MIN reduce op
    if log_idx[0] < len(events[resolved_log_paths[0]]):
        op_name, ranks, shape, dtype, reduce_op, src_rank, peer = events[resolved_log_paths[0]][
            log_idx[0]
        ]
        if tuple(sorted(ranks)) == tuple(range(world_size)):
            for r_ in ranks:
                log_idx[r_] += 1
            weighted_permutations.append(
                (
                    shape_product(shape) * dtype_size(dtype),
                    tuple((r_, (r_ + 1) % world_size) for r_ in ranks),
                )
            )
            permutations.add(weighted_permutations[-1][1])
    # the weight tying between embedding and lm head; this may not be present if they are untied
    if log_idx[0] < len(events[resolved_log_paths[0]]):
        op_name, ranks, shape, dtype, reduce_op, src_rank, peer = events[resolved_log_paths[0]][
            log_idx[0]
        ]
        if tuple(sorted(ranks)) == tuple([0, world_size - dp_size * tp_size]):
            for r_ in range(dp_size * tp_size):
                log_idx[r_] += 1
            weighted_permutations.append(
                (
                    shape_product(shape) * dtype_size(dtype),
                    tuple(
                        (r_, r_ + world_size - dp_size * tp_size) for r_ in range(dp_size * tp_size)
                    ),
                )
            )
            permutations.add(weighted_permutations[-1][1])
    # the broadcast to all nodes about something
    if log_idx[0] < len(events[resolved_log_paths[0]]):
        op_name, ranks, shape, dtype, reduce_op, src_rank, peer = events[resolved_log_paths[0]][
            log_idx[0]
        ]
        if tuple(sorted(ranks)) == tuple(range(world_size)):
            for r_ in ranks:
                log_idx[r_] += 1
            weighted_permutations.append(
                (
                    shape_product(shape) * dtype_size(dtype),
                    tuple((r_, (r_ + 1) % world_size) for r_ in ranks),
                )
            )
            permutations.add(weighted_permutations[-1][1])
    # SECOND: process the tensor parallel communication events for the entire world size
    while True:
        if log_idx[0] < len(events[resolved_log_paths[0]]):
            op_name, ranks, shape, dtype, reduce_op, src_rank, peer = events[resolved_log_paths[0]][
                log_idx[0]
            ]
            if op_name == "batch_isend" or op_name == "batch_irecv":
                break
            if len(ranks) == 1:
                log_idx[0] += 1
                continue

            if tuple(shape) != (seq_len, micro_batch_size, hidden_size):
                permutation_list = []
                for i in range(world_size // tp_size):
                    for j in range(i * tp_size, (i + 1) * tp_size):
                        permutation_list.append(
                            (j, i * tp_size if (j + 1) % tp_size == 0 else (j + 1))
                        )
                permutation = tuple(permutation_list)
                permutations.add(permutation)

                weighted_permutations.append(
                    (
                        shape_product(shape) * dtype_size(dtype),
                        permutation,
                    )
                )
                permutations.add(weighted_permutations[-1][1])
            else:
                # for the first pipeline stage
                permutation_list = []
                for i in range(world_size // pp_size // tp_size):
                    for j in range(i * tp_size, (i + 1) * tp_size):
                        permutation_list.append(
                            (j, i * tp_size if (j + 1) % tp_size == 0 else (j + 1))
                        )
                permutation = tuple(permutation_list)
                permutations.add(permutation)
                weighted_permutations.append(
                    (
                        shape_product(shape) * dtype_size(dtype),
                        permutation,
                    )
                )
                permutations.add(weighted_permutations[-1][1])

                # for the last pipeline stage
                permutation_list = []
                for i in range(world_size // pp_size // tp_size):
                    for j in range(world_size - (i + 1) * tp_size, world_size - i * tp_size):
                        permutation_list.append(
                            (
                                j,
                                world_size - (i + 1) * tp_size
                                if (j + 1) % tp_size == 0
                                else (j + 1),
                            )
                        )
                permutation = tuple(sorted(permutation_list))
                permutations.add(permutation)
                weighted_permutations.append(
                    (
                        shape_product(shape) * dtype_size(dtype),
                        permutation,
                    )
                )
                permutations.add(weighted_permutations[-1][1])
            log_idx[0] += 1
    # THIRD: process the pipeline parallel communication events for the entire world size
    #   TODO: Tech debt here as we might not have full permutations always as instantiated below
    #         and it might depend on the pipeline parallel schedule
    if pp_size > 1:
        shape = [seq_len, micro_batch_size, hidden_size]
        dtype = "torch.float32"

        # Forward pass
        permutation_list = []
        for i in range(world_size - tp_size * dp_size):
            permutation_list.append((i, i + tp_size * dp_size))
        permutation = tuple(sorted(permutation_list))
        permutations.add(permutation)
        weighted_permutations.append(
            (
                shape_product(shape) * dtype_size(dtype),
                permutation,
            )
        )
        permutations.add(weighted_permutations[-1][1])
        # Backward pass
        permutation_list = []
        for i in range(tp_size * dp_size, world_size):
            permutation_list.append((i, i - tp_size * dp_size))
        permutation = tuple(sorted(permutation_list))
        permutations.add(permutation)
        weighted_permutations.append(
            (
                shape_product(shape) * dtype_size(dtype),
                permutation,
            )
        )
        permutations.add(weighted_permutations[-1][1])

    # FOURTH, process the data parallel communication events for the entire world size
    #   TODO: Tech debt here as the overlapping of computation and communication might mean that
    #         all the data parallel communication doesn't happen at the same time and might
    #         overlap with pipeline parallel or tensor parallel communication of another stage
    #  TODO: Dummy shape here but full permutation is correct
    if dp_size > 1:
        shape = [seq_len, micro_batch_size, hidden_size]
        dtype = "torch.float32"

        permutation_list = []
        for i in range(world_size // dp_size // tp_size):
            for j in range(i * tp_size * dp_size, i * tp_size * dp_size + tp_size):
                ranks = []
                for k in range(dp_size):
                    ranks.append(j + k * tp_size)
                for l in range(len(ranks)):
                    permutation_list.append((ranks[l], ranks[(l + 1) % len(ranks)]))
        permutation = tuple(permutation_list)
        permutations.add(permutation)
        weighted_permutations.append(
            (
                shape_product(shape) * dtype_size(dtype),
                permutation,
            )
        )
        permutations.add(weighted_permutations[-1][1])

    return weighted_permutations, permutations


def main(argv: List[str]) -> int:
    """CLI entry point.

    Parses arguments, resolves the appropriate log JSON (from a directory with
    summary.json or a direct file path), reads events, and prints op names
    excluding "new_group". If --unique is provided, prints unique ops only.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Read a comm log JSON and print op names excluding 'new_group' events. "
            "Input can be a directory containing summary.json or a direct path to a log JSON."
        )
    )
    parser.add_argument(
        "input_path",
        nargs="?",
        default=None,
        help=("Path to comm log directory (with summary.json) or direct log JSON file. "),
    )

    args = parser.parse_args(argv)

    dp_size = None
    tp_size = None
    pp_size = None
    seq_len = None
    hidden_size = None
    num_layers = None
    num_micro_batches = None
    global_batch_size = None
    resolved_log_paths = None
    events = {}
    try:
        with open(os.path.join(args.input_path, "summary.json"), "r") as fp:
            config = json.load(fp)
        dp_size = config["config"].get("data_parallel_size", None)
        tp_size = config["config"].get("tensor_model_parallel_size", None)
        pp_size = config["config"].get("pipeline_model_parallel_size", None)
        seq_len = config["config"].get("model_config").get("seq_length", None)
        hidden_size = config["config"].get("model_config").get("hidden_size", None)
        num_layers = config["config"].get("model_config").get("num_layers", None)
        num_micro_batches = config["config"].get("num_micro_batches", None)
        global_batch_size = config["config"].get("global_batch_size", None)
        assert (
            dp_size is not None
            and tp_size is not None
            and pp_size is not None
            and seq_len is not None
            and hidden_size is not None
            and num_layers is not None
            and num_micro_batches is not None
            and global_batch_size is not None
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    try:
        # Resolve the log paths from the input path
        resolved_log_paths = resolve_log_paths_from_input(args.input_path)
        # Read the events from the log paths
        for resolved_log_path in resolved_log_paths:
            events[resolved_log_path] = list(
                iter_ops_excluding_new_group(read_events_from_json(resolved_log_path))
            )
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    """
    for item in events[resolved_log_paths[47]]:
        op_name, ranks, shape, dtype, reduce_op, src_rank, peer = item
        print(
            f"op_name: {op_name}, ranks: {ranks}, shape: {shape}, "
            f"dtype: {dtype}, reduce_op: {reduce_op}, src_rank: {src_rank}, peer: {peer}"
        )
    """

    weighted_permutations, permutations = process_events(
        resolved_log_paths,
        events,
        dp_size,
        tp_size,
        pp_size,
        seq_len,
        hidden_size,
        num_micro_batches,
        global_batch_size,
    )
    for permutation in permutations:
        print(f"permutation: {permutation}")

    return 0


if __name__ == "__main__":
    main(sys.argv[1:])
