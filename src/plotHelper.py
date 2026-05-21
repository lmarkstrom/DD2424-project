import matplotlib.pyplot as plt
import os
from datetime import datetime

def plotPerformance(results, save_dir="../images"):
    loss, val_loss = results["loss"], results["val_loss"]
    acc, val_acc = results["acc"], results["val_acc"]
    x_axis = results["steps"]

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(x_axis, loss, label="Training Loss")
    plt.plot(x_axis, val_loss, label="Validation Loss")
    plt.xlabel("Update step", fontsize=14)
    plt.ylabel("Loss", fontsize=14)
    plt.title("Loss", fontsize=18)
    plt.legend(fontsize=14)

    plt.subplot(1, 2, 2)
    plt.plot(x_axis, acc, label="Training Accuracy")
    plt.plot(x_axis, val_acc, label="Validation Accuracy")
    plt.xlabel("Epochs", fontsize=14)
    plt.ylabel("Accuracy", fontsize=14)
    plt.title("Accuracy", fontsize=18)
    plt.legend(fontsize=14)

    plt.tight_layout()

    # Save the plot automatically
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(save_dir, f"training_performance_{timestamp}.png")
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    print(f"Performance plot saved to: {filepath}")

    plt.close()