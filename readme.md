# Federated Learning with Differential Privacy

A PyTorch implementation of Federated Averaging (FedAvg) and FedAvg with Local Differential Privacy for the Federated EMNIST dataset, as part of a university assignment. This repository includes sequential, parallel (via Ray), and differentially private implementations.

## Requirements

- Python: 3.10 or later
- Package manager: pip
- CUDA (for GPU): An NVIDIA GPU with appropriate CUDA drivers is recommended for faster training.
- The scripts will automatically fall back to CPU if no GPU is detected.

## Installation

1.  Clone the repository and navigate into the project directory.

2.  Create and activate a Conda environment:

    ```bash
    conda create -n fl_hw3 python=3.10 -y
    conda activate fl_hw3
    ```

3.  Install the required Python packages:

    ```bash
    pip install -r requirements.txt
    ```

4.  Place the `Assignment3-data` directory (containing `train_data.npy` and `test_data.npy`) in the root of the project directory.

## GPU usage notes

- The scripts automatically detect CUDA via `torch.cuda.is_available()` and move models/tensors to the GPU if one is available. If not, they will run on the CPU.
- The parallel script (`fedavg_part1_2.py`) uses Ray and will allocate 1 GPU per client actor if a GPU is available and configured.
- To control which GPU is used, you can set the `CUDA_VISIBLE_DEVICES` environment variable before running a script. For example, to use the first GPU (ID 0):

  ```bash
  # On macOS/Linux
  export CUDA_VISIBLE_DEVICES=0
  python fedavg_part1_1.py
  
  # On Windows (PowerShell)
  $env:CUDA_VISIBLE_DEVICES="0"
  python fedavg_part1_1.py
  ```

## File structure (important files)

- `fedavg_part1_1.py` — Implements the sequential FedAvg algorithm as required by Part 1, question 1.
- `fedavg_part1_2.py` — Implements a parallel version of FedAvg using Ray actors (Part 1, question 2).
- `fedavg_part2.py` — Implements FedAvg with local differential privacy using the Laplace mechanism (Part 2).
- `requirements.txt` — A list of all Python dependencies.
- `Assignment3-data/` — The directory containing the training and test datasets.

## How to run

Run the scripts directly from your terminal. Each script corresponds to a specific part of the assignment and will produce plots upon completion.

**Part 1: Sequential FedAvg**

This script runs the baseline federated averaging algorithm.

```bash
nohup python -u fedavg_part1_1.py > fedavg_part1_1.log 2>&1 &
```

**Part 1: Parallel FedAvg with Ray**

This script runs the parallelized version of FedAvg, which should be faster if you have multiple CPU cores.

```bash
nohup python -u fedavg_part1_2.py > fedavg_part1_2.log 2>&1 &
```

**Part 2: FedAvg with Differential Privacy**

This script first runs the training with a fixed noise scale and then runs an experiment to show the trade-off between accuracy and privacy for different noise scales.

```bash
nohup python -u fedavg_part2.py > fedavg_part2.log 2>&1 &
```

You can watch the progress directly in your terminal. Each script will print the training progress per round and display plots at the end of the run.