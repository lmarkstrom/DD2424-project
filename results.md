# E-Level

## Results

| Model | Lambda (λ) | Performance | f | n_f | Cycles | Epochs | m | k | Batch Size | Etas |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **v.0** | 0.003 | 66.91% | 4 | 40 | 4 | 4 | 400 | 10 | 100 | [1e-5, 0.1] |
| **v.0** | 0.005 | 66.18% | 4 | 40 | 4 | 4 | 400 | 10 | 100 | [1e-5, 0.1] |
| **v.1** | ... | ...% | 4 | 40 | 3 | 4 | 400 | 10 | 100 | [1e-5, 0.1] |

## Models

# v.0
Model with no extra implementations as a baseline. 
SGD optimizer, cyclic learning rate and lambda regularization, 
No data augmentation, no dropout, no normalization and no label smoothening.

# v.1
Added Adam Optimizer and VGG block.

# Todo
Add batch normalization.

Add data augmentation.

Add dropout.

Add label smoothening.