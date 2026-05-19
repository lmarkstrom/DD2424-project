import matplotlib.pyplot as plt


def plotPerformance(results):
    loss, val_loss = results["loss"], results["val_loss"]
    acc, val_acc = results["acc"], results["val_acc"]
    x_axis = results["steps"]

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(x_axis, loss, label="Training Loss")
    plt.plot(x_axis, val_loss, label="Validation Loss")
    plt.xlabel("Epoch", fontsize=14)
    plt.ylabel("Loss", fontsize=14)
    plt.title("Loss", fontsize=18)
    plt.legend(fontsize=14)

    plt.subplot(1, 2, 2)
    plt.plot(x_axis, acc, label="Training Accuracy")
    plt.plot(x_axis, val_acc, label="Validation Accuracy")
    plt.xlabel("Epoch", fontsize=14)
    plt.ylabel("Accuracy", fontsize=14)
    plt.title("Accuracy", fontsize=18)
    plt.legend(fontsize=14)

    plt.tight_layout()
    plt.show()
