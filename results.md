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
Extension of initial model with the addition of a single VGG layer, with filter size 2x2 and 32 filters, between the first patchify layer and the fully connected layer, as well as substitution of the SGD optimizer to AdamW optimizer.

| Model | Lambda (λ) | Performance | f | n_f | Cycles | Epochs | m | k | Batch Size | Etas |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Basic** | 0.0025 | 67.89% | 4 | 40 | 3 | 4 | 400 | 10 | 100 | [1e-5, 0.1] |
| **Adam** | 0.0025 | 63.16% | 4 | 40 | 1 | 28 | 400 | 10 | 100 | 1e-3 |
| **1-block-VGG** | 0.0025 | 74.80% | 3 | 32 | 3 | 4 | 128 | 10 | 100 | [1e-5, 0.1] |
| **Combined** | 0.0025 | 66.92% | 4 | 40 | 1 | 28 | 400 | 10 | 100 | 1e-3 |

## 3-block-VGG extension

**Parameters:**
- f: 4
- n_f: 40
- n_s: 800
- n_cycles: 1
- n_epochs: 28
- n_hidden: 300
- n_batch: 100
- Lambda: 0.1
- Eta: 0.001
- Patchify layer: f: 2, s: 2, n_f: 32
- VGG layer 1: f: 3, s: 1, n_f: 32
- VGG layer 2: f: 3, s: 1, n_f: 64
- VGG layer 3: f: 3, s: 1, n_f: 128
- Fully connected layer 1: in: 2048, out: n_hidden
- Fully connected layer 2: in: n_hidden, out: 10

**Accuracy:** 74.03%

## Basic Regularization Techniques

**Parameters:**
- f: 4
- n_f: 40
- n_s: 800
- n_cycles: 1
- n_epochs: 28
- n_hidden: 300
- n_batch: 100
- Lambda: 0 (try independently)
- Eta: 0.001
- Patchify layer: f: 2, s: 2, n_f: 32
- VGG layer 1: f: 3, s: 1, n_f: 32
- VGG layer 2: f: 3, s: 1, n_f: 64
- VGG layer 3: f: 3, s: 1, n_f: 128
- Fully connected layer 1: in: 2048, out: n_hidden
- Fully connected layer 2: in: n_hidden, out: 10

### Dropout
| Model | Performance | Dropout Rate |
| :--- | :--- | :--- |
| Dropout 1 | x% | 10% |
| Dropout 2 | x% | 20% |

### Weight Decay (L2-regularization)
| Model | Performance | Lambda |
| :--- | :--- | :--- |
| Weight Decay 1 | 74.03% | 0.1 |
| Weight Decay 2 | x% | 0.01 |

### Data augmentation
| Model | Performance | Mirror Probability (%) | Shifting Probability (%) | Shifting (pixels) |
| :--- | :--- | :--- | :--- | :--- |
| Image Mirror and shifting |  x | 50% | 50% | 3 |


# Combined Regularization Techniques and Normalization
Added batch normalization after each convolutional layer and the first fully connected layer.

Different combinations of previously seen best results:
| Model | Performance | Lambda | Dropout | Image Mirror | Image Shift |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Combined 1** | x% | x | x | x | x | 
| **Combined 2** | x% | x | x | x | x |
| **Combined 3** | x% | x | x | x | x |
| **Combined 4** | x% | x | x | x | x |
| **Combined 5** | x% | x | x | x | x |

# Todo
Add batch normalization.

Add label smoothening.

Early stopping?