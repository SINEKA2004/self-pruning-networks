# Self-Pruning Neural Network

A PyTorch implementation of a neural network that learns to prune itself during training using learnable gates and L1 sparsity regularization.

## Project Overview

This project implements the case study **"The Self-Pruning Neural Network"** for CIFAR-10 image classification.  
Instead of pruning after training, the network learns which weights are unnecessary while it is being trained.

Each weight has a learnable gate value:
- gate close to `1` → weight stays active
- gate close to `0` → weight is effectively pruned

The model uses a custom `PrunableLinear` layer, a sparsity regularization loss, and a training loop that compares multiple values of `λ` to study the sparsity-vs-accuracy trade-off.

## Features

- Custom `PrunableLinear` layer built from scratch
- Learnable `gate_scores` for every weight
- Sigmoid-based gating mechanism
- L1 sparsity regularization on gates
- CIFAR-10 training and evaluation
- Sparsity level calculation
- Gate distribution plot for the best model
- Comparison of multiple `λ` values

## How It Works

### 1. Prunable Linear Layer
Each linear layer has:
- `weight`
- `bias`
- `gate_scores`

During the forward pass:
1. `gate_scores` are passed through `sigmoid()`
2. gates become values between `0` and `1`
3. weights are multiplied by these gates
4. the pruned weights are used for prediction

### 2. Sparsity Loss
The total loss is:

```text
Total Loss = Classification Loss + λ × Sparsity Loss
```

The sparsity loss is the sum of all gate values.  
This pushes unimportant gates toward zero and makes the network sparse.

### 3. Evaluation
After training, the script reports:
- test accuracy
- sparsity level
- gate value distribution

## Dataset

- **CIFAR-10**
- 50,000 training images
- 10,000 test images
- 10 classes
- Image size: `32 x 32 x 3`

## Repository Structure

```text
self_pruning_network.py   # Main training script
README.md                 # Project documentation
data/                     # CIFAR-10 dataset download folder
```

## Requirements

- Python 3.9+
- PyTorch
- torchvision
- matplotlib
- seaborn
- numpy

Install dependencies:

```bash
pip install torch torchvision matplotlib seaborn numpy
```

## How to Run

```bash
python self_pruning_network.py
```

The script will:
1. download CIFAR-10,
2. train the self-pruning model,
3. test the model,
4. calculate sparsity,
5. compare different values of `λ`,
6. display the gate distribution plot.

## Key Parameters

| Parameter | Meaning |
|---|---|
| `λ = 0.0` | No pruning pressure |
| `λ = 0.001` | Mild pruning |
| `λ = 0.01` | Strong pruning |

## Results

Example output from training:

| Lambda | Test Accuracy | Sparsity Level |
|---|---:|---:|
| 0.0 | 52.77% | 0.00% |
| 0.001 | To be filled after run | To be filled after run |
| 0.01 | To be filled after run | To be filled after run |

Your final results may change depending on:
- number of epochs
- learning rate
- model architecture
- pruning strength

## Output Files

The script generates:
- training and test logs
- sparsity statistics
- gate distribution plot

## Explanation of Sparsity Level

Sparsity level means the percentage of gates whose value is below a small threshold like `1e-2`.  
If a gate is below this value, that connection is treated as pruned.

## Why This Project Matters

This project is useful for:
- model compression
- faster inference
- lower memory usage
- efficient deployment on limited hardware

## Future Improvements

Possible improvements:
- use a deeper CNN backbone
- tune `λ` more carefully
- add learning-rate scheduling
- use stronger augmentation
- prune convolution layers too

## Author

Prepared for the AI Engineer case study:  
**The Self-Pruning Neural Network**

## License

This project is for academic and demonstration purposes.
