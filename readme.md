# AutoML for Image Classification

A PyTorch implementation of hyperparameter tuning algorithms (Genetic Algorithm and Bayesian Optimization) to optimize a neural network for the image classification task using a subset of the Federated EMNIST dataset (digits only).

## Requirements

- Python: 3.10 or later
- Package manager: pip
- CUDA (for GPU): An NVIDIA GPU with appropriate CUDA drivers is recommended for faster training.
- The scripts will automatically fall back to CPU if no GPU is detected.

## Installation

1.  Clone the repository and navigate into the project directory.

2.  Create and activate a Conda environment:

    ```bash
    conda create -n automl_hw4 python=3.10 -y
    conda activate automl_hw4
    ```

3.  Install the required Python packages:

    ```bash
    pip install -r requirements.txt
    ```
    *Note: Ensure `bayesian-optimization` is included in your requirements.*

4.  Place the `data` directory (containing `train_data.npy` and `test_data.npy`) in the root of the project directory.

## GPU usage notes

- The scripts automatically detect CUDA via `torch.cuda.is_available()` and move models/tensors to the GPU if one is available. If not, they will run on the CPU.
- To control which GPU is used, you can set the `CUDA_VISIBLE_DEVICES` environment variable before running a script. For example, to use the first GPU (ID 0):

  ```bash
  # On macOS/Linux
  export CUDA_VISIBLE_DEVICES=0
  python genetic_algorithm.py
  
  # On Windows (PowerShell)
  $env:CUDA_VISIBLE_DEVICES="0"
  python genetic_algorithm.py
  ```

## File structure (important files)

  - `genetic_algorithm.py` — Implements the Genetic Algorithm (GA) to tune batch size and activation function (Part 1).
  - `bayesian_optimization.py` — Implements Bayesian Optimization using the `bayesian-optimization` package (Part 2).
  - `requirements.txt` — A list of all Python dependencies.
  - `data/` — The directory containing the training and test datasets.

## How to run

Run the scripts directly from your terminal. Each script corresponds to a specific part of the assignment and will produce plots and report the best hyperparameters found.

**Part 1: Genetic Algorithm**

This script runs the custom Genetic Algorithm implementation (using roulette selection, one-point crossover, and mutation) to find the optimal hyperparameters.

```bash
nohup python -u genetic_algorithm.py > genetic_algorithm.log 2>&1 &
```

**Part 2: Bayesian Optimization**

This script utilizes the `bayesian-optimization` library to tune the hyperparameters by maximizing the black-box function (validation F1 score).

```bash
nohup python -u bayesian_optimization.py > bayesian_optimization.log 2>&1 &
```