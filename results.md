# E-Level

## Basic Model
Model with no extra implementations as a baseline. 
SGD optimizer with momentum, cyclic learning rate and lambda regularization, 
No data augmentation, no dropout, no normalization and no label smoothening.

| Model | Lambda (λ) | Performance | f | n_f | Cycles | Epochs | m | k | Batch Size | Etas |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Basic model 1** | 0.003 | 66.91% | 4 | 40 | 4 | 4 | 400 | 10 | 100 | [1e-5, 0.1] |
| **Basic model 2** | 0.005 | 66.18% | 4 | 40 | 4 | 4 | 400 | 10 | 100 | [1e-5, 0.1] |

## Adam and VGG extension

| Model | Lambda (λ) | Performance | f | n_f | Cycles | Epochs | m | k | Batch Size | Etas |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Basic** | 0.003 | 67.89% | 4 | 40 | 3 | 4 | 400 | 10 | 100 | [1e-5, 0.1] |
| **Adam** | 0.005 | ??.??% | 4 | 40 | 3 | 4 | 400 | 10 | 100 | [1e-5, 0.1] |
| **VGG** | 0.005 | ??.??% | 4 | 40 | 4 | 4 | 400 | 10 | 100 | [1e-5, 0.1] |
| **Combined** | 0.005 | ??.??% | 4 | 40 | 3 | 4 | 400 | 10 | 100 | [1e-5, 0.1] |

## Models

# v.0
Basic model.

# v.1
Added Adam Optimizer and VGG block.

# v.2
...

# Todo
Add batch normalization.

Add data augmentation.

Add dropout.

Add label smoothening.