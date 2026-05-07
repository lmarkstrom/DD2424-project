# E-Level

## Basic Model
Model with no extra implementations as a baseline. 
SGD optimizer with momentum, cyclic learning rate and lambda regularization, 
No data augmentation, no dropout, no normalization and no label smoothening.

| Model | Lambda (λ) | Performance | f | n_f | Cycles | Epochs | m | k | Batch Size | Etas |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Basic model 1** | 0.003 | 66.91% | 4 | 40 | 4 | 4 | 400 | 10 | 100 | [1e-5, 0.1] |
| **Basic model 2** | 0.005 | 66.18% | 4 | 40 | 4 | 4 | 400 | 10 | 100 | [1e-5, 0.1] |

## Adam and 1-block-VGG extension

| Model | Lambda (λ) | Performance | f | n_f | Cycles | Epochs | m | k | Batch Size | Etas |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Basic** | 0.0025 | 67.89% | 4 | 40 | 3 | 4 | 400 | 10 | 100 | [1e-5, 0.1] |
| **Adam** | 0.0025 | 63.16% | 4 | 40 | 3 | 4 | 400 | 10 | 100 | [1e-6, 1e-3] |
| **1-block-VGG** | 0.0025 | 74.80% | 3 | 32 | 3 | 4 | 128 | 10 | 100 | [1e-5, 0.1] |
| **Combined** | 0.0025 | ??.??% | 4 | 40 | 3 | 4 | 400 | 10 | 100 | [1e-5, 0.1] |

## 3-block-VGG extension

# Todo
Add batch normalization.

Add data augmentation.

Add dropout.

Add label smoothening.

Early stopping?