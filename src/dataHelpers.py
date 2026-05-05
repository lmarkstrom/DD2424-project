import numpy as np
import pickle

def loadBatch(filename):
    cifar_dir = 'data/cifar-10-batches-py/'
    with open(cifar_dir + filename, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    
    X = dict[b'data'].astype(np.float64) / 255.0
    X = X.transpose()
    nn = X.shape[1]

    y = np.array(dict[b'labels'])

    Y = np.zeros((10, nn))
    Y[y, np.arange(nn)] = 1

    return X, Y, y

def loadData():
    X_1, Y_1, y_1 = loadBatch('data_batch_1')
    X_2, Y_2, y_2 = loadBatch('data_batch_2')
    X_3, Y_3, y_3 = loadBatch('data_batch_3')
    X_4, Y_4, y_4 = loadBatch('data_batch_4')
    X_5, Y_5, y_5 = loadBatch('data_batch_5')

    X = np.hstack((X_1, X_2, X_3, X_4, X_5))
    Y = np.hstack((Y_1, Y_2, Y_3, Y_4, Y_5))
    y = np.hstack((y_1, y_2, y_3, y_4, y_5))

    X_train, Y_train, y_train = X[:, :49000], Y[:, :49000], y[:49000]
    X_val, Y_val, y_val = X[:, 49000:], Y[:, 49000:], y[49000:]

    X_test, Y_test, y_test = loadBatch('test_batch')

    return X_train, Y_train, y_train, X_val, Y_val, y_val, X_test, Y_test, y_test

def preprocess(X_train, X_val, X_test, d):
    mean_X = np.mean(X_train, axis=1).reshape(d, 1)
    std_X = np.std(X_train, axis=1).reshape(d, 1)

    X_train = (X_train - mean_X) / std_X
    X_val = (X_val - mean_X) / std_X
    X_test = (X_test - mean_X) / std_X

    return X_train, X_val, X_test

def smoothLabels(Y, eps=0.1):
    K = Y.shape[0]
    Y_smooth = Y * (1 - eps) + (1 - Y) * (eps / (K - 1))
    return Y_smooth

def augmentDataFlip(X):
    X_T = X.T 
    X_reshaped = X_T.reshape(-1, 3, 32, 32)
    X_flipped = X_reshaped[:, :, :, ::-1]
    return X_flipped.reshape(-1, 3072).T

def dataAugmentationTransform(xx):
    tx = np.random.randint(-3, 4)
    ty = np.random.randint(-3, 4)
    
    # empty hold padded image
    xx_shifted = np.zeros_like(xx)
    
    # Handle negative
    dx = abs(tx)
    
    aa = np.arange(32).reshape((32, 1))
    vv = np.tile(32*aa, (1, 32-dx))
    
    bb1 = np.arange(max(0, tx), max(0, tx) + 32 - dx, 1).reshape((32 - dx, 1))
    bb2 = np.arange(max(0, -tx), max(0, -tx) + 32 - dx, 1).reshape((32-dx, 1))
    
    ind_fill = vv.reshape((32*(32-dx), 1)) + np.tile(bb1, (32, 1))
    ind_xx = vv.reshape((32*(32-dx), 1)) + np.tile(bb2, (32, 1))
    
    v_fill = (ind_fill // 32 >= max(0, ty)) & (ind_fill // 32 < 32 - max(0, -ty))
    v_xx = (ind_xx // 32 >= max(0, -ty)) & (ind_xx // 32 < 32 - max(0, ty))
    
    ind_fill = ind_fill[v_fill].reshape(-1, 1)
    ind_xx = ind_xx[v_xx].reshape(-1, 1)
    
    inds_fill = np.vstack((ind_fill, 1024+ind_fill))
    inds_fill = np.vstack((inds_fill, 2048+ind_fill))
    
    inds_xx = np.vstack((ind_xx, 1024+ind_xx))
    inds_xx = np.vstack((inds_xx, 2048+ind_xx))
    
    xx_shifted = np.zeros_like(xx)
    xx_shifted[inds_fill] = xx[inds_xx]
    
    return xx_shifted