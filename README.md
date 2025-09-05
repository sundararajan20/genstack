# Genstack: A Megatron-LM based communication tracer and permutation generator

This module provides tools to capture and analyze the distributed communication patterns in Megatron-LM without requiring actual GPU resources. It uses meta tensors and process spoofing to simulate distributed training with various parallelism strategies.

## Key Features

- **Meta Tensor Implementation**: Intercepts tensor creation and replaces with zero-memory meta tensors
- **Process Group Spoofing**: Creates multiple processes to simulate ranks
- **Communication Interception**: Monkey-patches PyTorch's distributed communication functions
- **Support for 3 Main Parallelism Dimensions**: Data, Tensor and Pipeline Parallelism (Context Parallelism and Sequence Parallelism are work in progress)

## Setup instuctions

- Install [Micromamba](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html) (a lightweight and performant Conda).
- ```micromamba env create -n genstack```
- ```micromamba activate genstack```
- ```micromamba install python uv``` (this project has been tested with python=3.12 and uv=0.8.8)

Now the instructions differ for linux and MacOS (note that this project has not been tested on Windows). Please follow the appropriate instuctions below

### On Linux

- ```uv pip install --index https://download.pytorch.org/whl/cpu -r requirements/pytorch_25.03/requirements.txt```
- (optional) If you face errors create a fresh micromamba environment and install python 3.12 and uv 0.8.8 in it and then run the following command: ```uv pip install --index https://download.pytorch.org/whl/cpu -r requirements_genstack_reproduce_linux.txt```

### On MacOS

- ```uv pip install -r requirements/pytorch_25.03/requirements.txt```
- (optional) If you face errors create a fresh micromamba environment and install python 3.12 and uv 0.8.8 in it and then run the following command: ```uv pip install -r requirements_genstack_reproduce_macos.txt```

## Usage

The communication tracer can be run using the provided command-line tool:

```bash
python tools/comm_trace.py --dp 2 --tp 2 --pp 2 --model-size medium
```

Note: On a single node with (say) 8-16 cores, don't exceed a world size (produce of data parallel, pipeline parallel and tensor parallel) of about 64 otherwise the logging will stall. Multi node logging is a work in progress.

### Command-Line Arguments

- `--dp`: Data parallel size (default: 1)
- `--tp`: Tensor parallel size (default: 1)
- `--pp`: Pipeline parallel size (default: 1)
- `--model-size`: GPT model size to simulate (options: small, medium, large, xl, 2b, 6b, 13b, 30b, 175b)
- `--micro-batches`: Number of micro-batches (default: 4)
- `--global-batch-size`: Global batch size (default: 16)
- `--seed`: Random seed (default: 1234)
- `--output-dir`: Directory to save communication logs (default: "comm_logs")

## Output

The tool generates one log file per rank in JSON format, containing detailed information about all communication operations:

- Collective operation type (all_reduce, all_gather, send/recv, etc.)
- Tensor shapes and data types
- Participating ranks
- Call identifiers for tracking the sequence of operations

## Examples

### Pre-run examples

Example communication logs are in the `comm_logs/` directory.

### Data Parallelism

```bash
python tools/comm_trace.py --dp 4 --tp 1 --pp 1 --model-size small
```

### Tensor Model Parallelism

```bash
python tools/comm_trace.py --dp 1 --tp 4 --pp 1 --model-size medium
```

### Pipeline Parallelism

```bash
python tools/comm_trace.py --dp 1 --tp 1 --pp 4 --model-size medium
```

### Combined Parallelism

```bash
python tools/comm_trace.py --dp 2 --tp 2 --pp 2 --model-size large
```

## Collective calls to permutations

Invoke the below script and point it to the logs directory created by running the `comm_trace.py` script

```bash
python megatron/instrumentation/logs_to_permutations.py <path-to-logs-dir>
```

Note: the transformation from collective calls to permutations makes some simplyfying assumptions that you can find in the code.

### Example

```bash
python megatron/instrumentation/logs_to_permutations.py comm_logs/175b_dp2_tp4_pp6_20250810_184348
```

## Implementation Details

The communication tracer works by:

1. **Patching tensor creation**: All tensor creation operations are intercepted to produce meta tensors with the same shape and dtype but no actual memory allocation.

2. **Spoofing processes**: Uses Python's multiprocessing to create multiple processes that simulate distributed ranks.

3. **Patching communication primitives**: Intercepts all PyTorch distributed operations to log their usage patterns without performing actual communication.

4. **Running a training iteration**: Executes a single forward and backward pass to capture the full set of communication operations in a training step.
