# Project Title: CNN classifier 

A simple CNN 3-layer classifier for the CIFAR-10 dataset as part of the course DD2424 project.

---

## Features
- **Custom Architecture:** 3-layer CNN with a 2-layer MLP trained using Backpropagation
- **Data Augmentation:** Real-time flipping, and image shifting
- **Drop-out:** Node dropout during training
- **Cyclic learning:** Cyclic learning with learning scheduler

## Dataset
Provide details about the data used.
- **Source:** [https://www.cs.toronto.edu/~kriz/cifar.html]
- **Classes:** `Class A`, `Class B`, `Class C`...
- **Splits:** 49'000 Training | 1'000 Validation | 10'000 Test
- **Preprocessing:** Normalized using mean/std.

## Architecture

| Layer Type | Configuration | Output Shape |
| :--- | :--- | :--- |
| **Input** | RGB Image | (3, 32, 32) |
| **1-layer filter + ReLU** | 32 filters, 3x3 kernel | (32, 222, 222) |
| **Dropout** | 0.1 rate | - |
| **Hidden layer** | 400 neurons | (400) |
| **Softmax** | 10-classes | (10) |

## Installation

```bash
# clone repo
git clone https://github.com/lmarkstrom/DD2424-project
cd DD2424-project

# create python envirement
python -m venv venv
.\venv\Scripts\activate

# install requirements
pip install --upgrade pip
pip install -r requirements.txt

```

### Windows
```bash
# create python envirement
python -m venv venv
.\venv\Scripts\activate

# install requirements
pip install --upgrade pip
pip install -r requirements.txt
```

### MacOS

```bash
# create python envirement
python3 -m venv venv
source venv/bin/activate

# install requirements
pip install --upgrade pip
pip install -r requirements.txt
```

## Run 

```bash
python src/main.py --mode full
```

## Exit envirement

```bash
deactivate
```