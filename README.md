# Typical Machine Learning Datasets as Low-Depth Quantum Circuits

This project provides a comprehensive toolset for representing image datasets as quantum circuits, optimizing them, and evaluating the impact of this optimization on classification tasks. The workflow is divided into three main stages: data preparation, quantum circuit optimization, and classification. The `plots` directory contains scripts and notebooks to visualize the results of these stages.

## Getting Started

### Prerequisites

You can install the required packages using the `requirements.txt` file after creating a environment with python==3.12.0:

```bash
conda create -n "quantum_datasets" python==3.12.0
conda activate quantum_datasets
pip install -r requirements.txt
```

### Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/fkiwit/low-depth-image-circuits
    cd low-depth-image-circuits
    ```

2.  The project uses custom scripts and does not require a formal installation. Ensure all dependencies from `requirements.txt` are installed.

## Workflow

The project is structured around a three-step pipeline:

### 1. Data Preparation

The first step is to download and process the image datasets. The `prepare_data.py` script handles this.

-   **Supported Datasets:** `MNIST`, `Fashion-MNIST`, `CIFAR-10`, and `Imagenette`.
-   **Processing:** Images are resized, cropped, and converted to either RGB or grayscale.
-   **Quantum Encoding:** The processed images are encoded into quantum states using the Flexible Representation of Quantum Images (FRQI) method.
-   **Output:** The script saves the processed images, labels, and their corresponding quantum states as `.npy` files in the `data/` directory.

**Usage:**

```bash
python prepare_data.py --dataset_name <dataset> [--dataset_id <id>] [--indexing <name>] [--n_patches <num_patches>]
```

Example:

```bash
python prepare_data.py --dataset_name mnist --indexing row_major --n_patches 1
```

### Pixel Orderings / 2D Encodings

Pixel orderings are managed via the permutation registry in `circuit_optimization/encodings/registry.py`.
Each encoding is defined in its own file and must implement:

```python
def get_permutation(m: int, n: int) -> np.ndarray
```

Available encodings (CLI names):
`row_major`, `column_major`, `snake`, `vertical_snake`, `diagonal`,
`diagonal_zigzag`, `corner_spiral`, `morton`, `hilbert`.

Aliases: `hierarchical`, `zorder`, `z-order` map to `morton`.

To add a new encoding:
1) Add a new file under `circuit_optimization/encodings/` with `get_permutation`.
2) Register it in `circuit_optimization/encodings/registry.py`.

### Encoding Ablation (Reproducible Runner)

Run end-to-end ablations (dataset generation + training + evaluation):

```bash
python scripts/experiment_runner.py \
  --tfds_name mnist \
  --base_dataset mnist \
  --indexings row_major morton hilbert \
  --models svm \
  --folds 0 \
  --max_per_class 5 \
  --timeout_seconds 600
```

Summarize results:

```bash
python scripts/summarize_results.py --results_dir results/encoding_ablation
```

Each run writes a `run.json` with status `SUCCESS`, `FAILED`, or `TIMEOUT`,
plus stdout/stderr logs for transparent failure handling.

### 2. Quantum Circuit Optimization

The `circuit_optimization/` directory contains scripts to optimize the quantum state representations of the images into shallow quantum circuits.

-   **Core Logic:** Scripts like `run_sweeping.py`, `run_bfgs.py`, and `run_sweeping_random.py` load quantum states and use various optimization algorithms (sweeping, BFGS, randomized) to find a low-depth quantum circuit that approximates each state.
-   **Data Collection:** The `data_collector_*.py` scripts are used to gather data from the optimization runs.
-   **Parallelization:** The optimization process can be parallelized using `ray` for efficiency.
-   **Circuit Types:** Different types of circuits can be used for optimization, such as `unitary`, `orthogonal`, or `sparse`, which correspond to different gate sets and circuit structures.
-   **Output:** The scripts save the parameters of the optimized circuits, which can be used for classification.

**Usage:**

The `run_sweeping_batch.sh`, `run_sweeping_local.sh`, and `run_bfgs_local.sh` scripts can be used to run the optimizations.

Example of running `run_sweeping.py` directly:

```bash
cd circuit_optimization
python run_sweeping.py --dataset mnist --circuit orthogonal --layers 4
```

### 3. Classification

The `classifier/` directory is dedicated to training various classifiers on the (potentially optimized) quantum data to evaluate the performance of the optimization.

-   **Goal:** To assess how different levels of optimization affect classification accuracy.
-   **Supported Classifiers:**
    -   **Support Vector Machines (SVM):** With a custom quantum kernel (`utils/svm_training.py`).
    -   **Tensor Network Classifiers:** Matrix Product State (MPS) and Matrix Product Operator (MPO) based classifiers (`utils/tensor_network_training.py`).
    -   **Variational Quantum Classifiers (VQC):** (`utils/vqc_training.py`).
    -   **Classical Convolutional Neural Networks (CNN):** As a baseline (`utils/cnn_training.py`).
-   **Training Orchestration:** The `launch_training_NN.py` script (for neural networks, VQCs, and tensor networks) and `launch_training_SVM.sh` (for SVMs) manage the training and hyperparameter tuning, using `ray.tune` and GNU `parallel`.

**Usage:**

To train the neural network based classifiers:

```bash
cd classifier
python launch_training_NN.py
```

To train SVM classifiers:

```bash
cd classifier
./launch_training_SVM.sh
```

## Project Structure

```
.
├── circuit_optimization/       # Scripts for quantum circuit optimization
│   ├── utils/                  # Utility functions for optimization
│   ├── run_sweeping.py         # Main script for sweeping optimization
│   ├── run_bfgs.py             # Main script for BFGS optimization
│   └── ...
├── classifier/                 # Scripts for training classifiers
│   ├── utils/                  # Implementations of different classifiers
│   ├── launch_training_NN.py   # Script to launch NN-based training
│   ├── launch_training_SVM.sh  # Script to launch SVM training
│   └── ...
├── data/                       # Default directory for datasets
├── plots/                      # Contains scripts and notebooks for generating plots and visualizations of the results from the optimization and classification tasks.
├── scripts/                    # Experiment runner and summarizer for encoding ablations
├── prepare_data.py             # Script for data preparation and quantum encoding
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## Tests / Smoke Checks

Run unit tests:

```bash
python -m unittest discover -s tests
```

Smoke check (small subset, SVM only):

```bash
python scripts/experiment_runner.py \
  --tfds_name mnist \
  --base_dataset mnist \
  --indexings row_major morton hilbert \
  --models svm \
  --folds 0 \
  --max_per_class 5 \
  --timeout_seconds 600
```

Smoke check (small subset, VQC):

```bash
python scripts/experiment_runner.py --indexings row_major morton --models vqc_linear --folds 0 --seeds 42 --max_per_class 2 --timeout_seconds 1800
```

## Pennylane Datasets
The datasets are also available on Pennylane Datasets: [https://pennylane.ai/datasets/collection/low-depth-image-circuits](https://pennylane.ai/datasets/collection/low-depth-image-circuits)

## Citation
If you use this project in your research, please cite the following paper:
```

@article{10.1088/2058-9565/ae0123,
	author={Kiwit, Florian J and Jobst, Bernhard and Luckow, Andre and Pollmann, Frank and Riofrío, Carlos A.},
	title={Typical Machine Learning Datasets as Low-Depth Quantum Circuits},
	journal={Quantum Science and Technology},
	url={http://iopscience.iop.org/article/10.1088/2058-9565/ae0123},
	year={2025}
}

```
The paper is available at: [https://iopscience.iop.org/article/10.1088/2058-9565/ae0123](https://iopscience.iop.org/article/10.1088/2058-9565/ae0123)
