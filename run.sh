#!/bin/bash

# Configuration Variables
VENV_DIR=".venv"
OUTPUT_FILE="output.txt"

# Ensure the AMD ROCm override flag is set globally for all sub-commands 
export HSA_OVERRIDE_GFX_VERSION=11.0.0

# ---------------------------------------------------------
# ARGUMENT: setup
# ---------------------------------------------------------
if [ "$1" == "setup" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv "$VENV_DIR"
    
    echo "Activating virtual environment..."
    source "$VENV_DIR"/bin/activate

    echo "Upgrading pip, setuptools, and wheel..."
    pip install --upgrade pip setuptools wheel

    echo "Installing AMD 7900 XTX tailored PyTorch suite..."
    # We target the PyTorch ROCm index URL explicitly to avoid grabbing CUDA wheels
    pip install -r requirements2.txt

    echo "Verifying hardware acceleration..."
    python3 -c "import torch; print('ROCm/HIP available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
    
    echo "Setup complete. Virtual environment is active and ready."

# ---------------------------------------------------------
# ARGUMENT: train
# ---------------------------------------------------------
elif [ "$1" == "train" ]; then
    if [ -f "$VENV_DIR/bin/activate" ]; then
        echo "Activating virtual environment for execution..."
        source "$VENV_DIR"/bin/activate
    else
        echo "Warning: No virtual environment found at $VENV_DIR. Running with default system python."
    fi

    echo "Launching python src/main --mode full..."
    echo "Tailing output to terminal while saving to $OUTPUT_FILE..."
    
    # Run python script and fork output to both file and stdout in real time
    python3 src/main --mode full 2>&1 | tee "$OUTPUT_FILE"

else
    echo "Usage: $0 {setup|train}"
    exit 1
fi
