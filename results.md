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
- Lambda: 0.1 (if nothing else stated, taken from the basic network)
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
| Dropout 2 | 75.11% | 20% |

### Weight Decay (L2-regularization)
| Model | Performance | Lambda |
| :--- | :--- | :--- |
| Weight Decay 1 | 74.03% | 0.1 |
| Weight Decay 2 | 66.63% | 0.01 |

### Data augmentation
| Model | Performance | Mirror Probability (%) | Shifting (%) |
| :--- | :--- | :--- | :--- |
| Image Mirror and shifting |  76.59 | 50% | 0.1 |

### Combined

**Accuracy:** 74.51%

## Combined Regularization Techniques and Normalization
Added batch normalization after each convolutional layer and the first fully connected layer.

Different combinations of previously seen best results:
| Model | Performance | Lambda | Dropout | Image Mirror | Image Shift |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Combined 1** | 82.89% | 0.0001 | 0.2 | 0.5| +/- 0.1 |
| **Combined 2** | 83.10% | 0.001 | 0.2 | 0.5| +/- 0.1 | 
| **Combined 3** | 85.66% | 0.001 | 0.2 | 0.5| +/- 0.1 | longmodel 3 (30 epoch, 0.001 lr)
| **Combined 4** | 85.63% | 0.001 | 0.2 | 0.5| +/- 0.1 | longmodel 4 (35 epoch, 0.01 lr)
| **Combined 5** | 85.85% | 0.001 | 0.2 | 0.5| +/- 0.1 | longmodel 5 (45 epoch, 0.01 lr)
| **Combined 6** | 86.74% | 0.001 | 0.2 | 0.5| +/- 0.1 | longmodel 6 (60 epoch, 0.01 lr)
| **Combined 7** | 83.85% | 0.01 | 0.2 | 0.5| +/- 0.1 | longmodel 7 (60 epoch, 0.01 lr)
| **Combined 8** | 86.48% | 0.001 | 0.2 | 0.5| +/- 0.1 | longmodel 8 (60 epoch, 0.01 lr) Higher dropout, crop
| **Combined 9** | 86.68% | 0.0001 | 0.2 | 0.5| +/- 0.1 | longmodel 9 (80 epoch, 0.01 lr) Higher dropout, crop
| **Combined 10** | 86.98% | 0.0001 | 0.2 | 0.5| +/- 0.1 | longmodel 10 (60 epoch, 0.01 lr) corrected augmentation - FINAL
| **Combined 11** | 86.80% | 0.0001 | 0.2 | 0.5| +/- 0.1 | longmodel 11 (60 epoch, 0.01 lr) lower droput

## Further improvements

### Label smoothing
| Model | Performance | Epochs | Lambda | lr | label smoothing |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Model 1** | 87.23% | 60 | 0.001 | 0.01 | 0.1 |

### Learning rate scheduling
| Model | Performance | Epochs | Lambda | lr |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Step** | 88.39% | 60 | 0.001 | 0.01 |
| **Step** | 89.46% | 60 | 0.001 | 0.01 |

### Down sampling

| Model | Performance | Epochs | Lambda | lr |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Step** | x% | 60 | 0.001 | 0.01 |

### Replace fc1 with average pooling

| Model | Performance | Epochs | Lambda | lr |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Step** | 86.97% | 60 | 0.001 | 0.01 |

