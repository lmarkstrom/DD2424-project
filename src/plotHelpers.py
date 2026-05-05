import numpy as np
import matplotlib.pyplot as plt
import pickle

def plotPerformance(losses, costs, accuracies, steps):
    loss, val_loss = losses['training'], losses['validation']
    cost, val_cost = costs['training'], costs['validation']
    acc, val_acc = accuracies['training'], accuracies['validation']
    x_axis = steps
    
    plt.figure(figsize=(15, 4))
    
    plt.subplot(1, 3, 1)
    plt.plot(x_axis, loss, label='Training Loss')
    plt.plot(x_axis, val_loss, label='Validation Loss')
    plt.xlabel('Update step', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.title('Loss', fontsize=18)
    plt.legend(fontsize=14)
    
    plt.subplot(1, 3, 2)
    plt.plot(x_axis, cost, label='Training Cost')
    plt.plot(x_axis, val_cost, label='Validation Cost')
    plt.xlabel('Update step', fontsize=14)
    plt.ylabel('Cost', fontsize=14)
    plt.title('Cost', fontsize=18)
    plt.legend(fontsize=14)

    plt.subplot(1, 3, 3)
    plt.plot(x_axis, acc, label='Training Accuracy')
    plt.plot(x_axis, val_acc, label='Validation Accuracy')
    plt.xlabel('Update step', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    plt.title('Accuracy', fontsize=18)
    plt.legend(fontsize=14)

    plt.tight_layout()
    plt.show()

# Visualize the first ni images in the batch specified by filename.
def plotImages(filename, ni):
    # Load a batch of training data
    with open(filename, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')

    # Extract the image data and cast to float from the dict dictionary
    X = dict[b'data'].astype(np.float64) / 255.0
    X = X.transpose()
    nn = X.shape[1]

    # Horizontal flip (optional)
    #for i in range(ni):
    #    X[:, i] = ImageFlipHorizontal(X[:, i])

    # Image shift (optional)
    #for i in range(ni):
    #    X[:, i] = ImageShift(X[:, i], 15, 15)

    # Reshape each image from a column vector to a 3d array
    X_im = X.reshape((32, 32, 3, nn), order='F')
    X_im = np.transpose(X_im, (1, 0, 2, 3))

    # Display the first ni images
    fig, axes = plt.subplots(1, ni, figsize=(20, 5))
    for i in range(ni):
        axes[i].imshow(X_im[:, :, :, i])
        axes[i].axis('off')
    plt.show()