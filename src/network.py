
import numpy as np
from dataHelpers import loadData, preprocess, augmentDataFlip, dataAugmentationTransform, smoothLabels
from plotHelpers import plotPerformance
from networkHelpers import softmax, computeAccuracy, computeCost, computeLoss, generateMX, learningRateSchedule, adamOptimizer

class Network:
    def __init__(self, LR_params, GD_params, CN_params):
        self.LR_params = LR_params
        self.GD_params = GD_params
        self.CN_params = CN_params
        
        self.data = {}
        
        self.network = {}
        
        self.results = {}
        
        self.initNetwork()
    
    # Initialize the network parameters (Kaiming He) and results
    def initNetwork(self):
        f, n_f = self.CN_params['f'], self.CN_params['n_f']
        m, k = self.GD_params['m'], self.GD_params['k']
        
        self.network = {}
        
        self.CN_params['n_p'] = (32 // f) ** 2
        
        fan_in_conv = f * f * 3
        self.network['conv_filters'] = np.random.randn(f, f, 3, n_f) * np.sqrt(2 / fan_in_conv)
        self.network['b_conv'] = np.zeros(n_f)
        self.network['MX'] = None
        
        self.network['W'] = [None] * 2
        self.network['b'] = [None] * 2
        
        fan_in_w1 = (32 // f)**2 * n_f
        self.network['W'][0] = np.random.randn(m, fan_in_w1) * np.sqrt(2 / fan_in_w1)
        self.network['b'][0] = np.zeros((m, 1))
        
        fan_in_w2 = m
        self.network['W'][1] = np.random.randn(k, m) * np.sqrt(2 / fan_in_w2)
        self.network['b'][1] = np.zeros((k, 1))
        
        self.results = {
            'losses': {'training': [], 'validation': []},
            'costs': {'training': [], 'validation': []},
            'accuracies': {'training': [], 'validation': []},
            'steps': []
        }
    
    # Load data and preprocess it
    def loadData(self, full=True, preprocess_data=True, augment_data=False, smooth_labels=False):
        if full:
            X_train, Y_train, y_train, X_val, Y_val, y_val, X_test, Y_test, y_test = loadData()
            self.data = {
                'X_train': X_train, 'Y_train': Y_train, 'y_train': y_train,
                'X_val': X_val, 'Y_val': Y_val, 'y_val': y_val,
                'X_test': X_test, 'Y_test': Y_test, 'y_test': y_test
            }
        else:
            # TODO: create and change to function which loads a small subset of the data for testing
            X_train, Y_train, y_train, _, _, _, _, _, _ = loadData()
            self.data = {
                'X_train': X_train[:, :1000], 'Y_train': Y_train[:, :1000], 'y_train': y_train[:1000]
            }
        
        if preprocess_data:
            self.data['X_train'], self.data['X_val'], self.data['X_test'] = preprocess(
                self.data['X_train'], self.data['X_val'], self.data['X_test'], self.data['X_train'].shape[0]
            )
            
        if smooth_labels:
            self.data['Y_train'] = smoothLabels(self.data['Y_train'], 0.1)
        
        if augment_data:
            pass
        
        self.GD_params['n'] = self.data['X_train'].shape[1]
        
        self.CN_params['n_s'] = (self.GD_params['n'] // self.GD_params['n_batch']) * self.GD_params['n_epochs'] // 2
    
    # Train the network using mini-batch gradient descent 
    def train(self, debug=True, augmentation=False, dropout=False):
        n_cycles, n_batch, lam = self.GD_params['n_cycles'], self.GD_params['n_batch'], self.GD_params['lam']
        f, n_p = self.CN_params['f'], self.CN_params['n_p']
        
        n = self.data['X_train'].shape[1]
        batches_per_epoch = n // n_batch
        
        current_n_s = self.CN_params['n_s']
        current_cycle = 0
        steps_in_cycle = 0
        t = 0
        
        epoch = 1

        MX_train = generateMX(self.data['X_train'], self.data['X_train'].shape[1], f, n_p)
        MX_val = generateMX(self.data['X_val'], self.data['X_val'].shape[1], f, n_p)
        
        while current_cycle < n_cycles:
            indices = np.random.permutation(n)            
            X_cycle = self.data['X_train'][:, indices]
            Y_cycle = self.data['Y_train'][:, indices]
            y_cycle = self.data['y_train'][indices]
            MX_cycle = MX_train[:, :, indices]

            if steps_in_cycle == 0:
                cycle_length = 2 * current_n_s
                cycle_start_t = t
                total_epochs_cycle = cycle_length // batches_per_epoch
                epoch = 1

            for j in range(batches_per_epoch):
                epoch = ((t - cycle_start_t) // batches_per_epoch) + 1
                
                j_start = j * n_batch
                j_end = (j + 1) * n_batch
                
                X_batch = X_cycle[:, j_start:j_end]
                Y_batch = Y_cycle[:, j_start:j_end]
                
                # augementation 
                if augmentation:
                    flip_ind = np.random.rand(X_batch.shape[1]) < 0.5
                    if np.any(flip_ind):
                        X_batch[:, flip_ind] = augmentDataFlip(X_batch[:, flip_ind])
                    for i in range(X_batch.shape[1]):
                        if np.random.rand() < 0.5:
                            X_batch[:, i] = dataAugmentationTransform(X_batch[:, i])
                
                MX_batch = generateMX(X_batch, n_batch, f, n_p)
                
                eta = learningRateSchedule(self.LR_params, current_n_s, steps_in_cycle)
                
                fp_data = self.forwardPass(MX_batch, dropout=dropout)
                grads, grad_Fs_flat = self.backwardPass(MX_batch, fp_data, Y_batch, lam)
                self.updateNetwork(eta, grads, grad_Fs_flat)

                
                if (t % ((n_cycles * 2 * current_n_s) // (n_cycles * 9)) == 0) and debug:
                    model = self.forwardPass(MX_cycle)
                    cost = computeCost(model['P'], Y_cycle, self.network, lam)
                    loss = computeLoss(model['P'], Y_cycle)
                    acc = computeAccuracy(model['P'], y_cycle)

                    val_model = self.forwardPass(MX_val)
                    val_cost = computeCost(val_model['P'], self.data['Y_val'], self.network, lam)
                    val_loss = computeLoss(val_model['P'], self.data['Y_val'])
                    val_acc = computeAccuracy(val_model['P'], self.data['y_val'])

                    self.results['losses']['training'].append(loss)
                    self.results['losses']['validation'].append(val_loss)
                    self.results['costs']['training'].append(cost)
                    self.results['costs']['validation'].append(val_cost)
                    self.results['accuracies']['training'].append(acc)
                    self.results['accuracies']['validation'].append(val_acc)
                    self.results['steps'].append(t)
                    
                if debug and (t + 1) % batches_per_epoch == 0:
                    epoch = ((t + 1) - cycle_start_t) // batches_per_epoch
                    print(f'Cycle {current_cycle+1}, Epoch {epoch}/{total_epochs_cycle}, Acc: {acc:.4f}, V.acc: {val_acc:.4f}')
                
                t += 1
                steps_in_cycle += 1
                
                if steps_in_cycle == cycle_length:
                    current_n_s *= 2
                    steps_in_cycle = 0
                    current_cycle += 1
                    total_epochs_cycle = (2 * current_n_s) // batches_per_epoch
                    if current_cycle >= n_cycles:
                        break
        if debug:
            plotPerformance(self.results['losses'], self.results['costs'], self.results['accuracies'], self.results['steps'])
    
    # Evaluate network performance after training
    def evaluate(self):
        n_test, n_p = self.data['X_test'].shape[1], self.CN_params['n_p']
        
        MX_test = generateMX(self.data['X_test'], n_test, self.CN_params['f'], n_p)
        test_model = self.forwardPass(MX_test)
        test_acc = computeAccuracy(test_model['P'], self.data['y_test'])
        test_loss = computeLoss(test_model['P'], self.data['Y_test'])
        
        print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}')
    
    # Forward pass through the network
    def forwardPass(self, MX, dropout=False, keep_prob=0.8):
        Fs = self.network['conv_filters']
        
        n = MX.shape[2]
        
        f, n_f, n_p = self.CN_params['f'], self.CN_params['n_f'], self.CN_params['n_p']

        b_conv = self.network.get('b_conv', np.zeros(n_f))
        
        Fs_flat = Fs.reshape((f * f * 3, n_f), order='C')
        
        conv_outputs = np.einsum('ijk, jl -> ilk', MX, Fs_flat, optimize=True)
        conv_outputs = conv_outputs + b_conv.reshape(1, n_f, 1)
        
        fp_data = {}
        
        # Relu
        conv_flat = np.fmax(conv_outputs.reshape((n_p*n_f, n), order='C'), 0) 
        
        H = conv_flat
        if dropout:
            dropout_mask = (np.random.rand(*H.shape) < keep_prob) / keep_prob
            H = H * dropout_mask
            fp_data['dropout_mask'] = dropout_mask
        fp_data['H'] = H
        
        S1 = self.network['W'][0] @ H + self.network['b'][0]
        X1 = np.maximum(0, S1)
        fp_data['S1'] = S1
        fp_data['X1'] = X1
        
        S = self.network['W'][1] @ X1 + self.network['b'][1]
        fp_data['S'] = S
        
        P = softmax(S)
        fp_data['P'] = P
        
        return fp_data
    
    # Backward pass to compute gradients
    def backwardPass(self, MX, fp_data, Y, lam):
        gradients = {'W': [None] * 2, 'b': [None] * 2}
    
        n = Y.shape[1]
        
        S1, X1, H, P = fp_data['S1'], fp_data['X1'], fp_data['H'], fp_data['P']
        f, n_f, n_p = self.network['conv_filters'].shape[0], self.network['conv_filters'].shape[3], self.CN_params['n_p']
        
        G = P - Y
        
        gradients['W'][1] = (G @ X1.T) / n + 2 * lam * self.network['W'][1]
        gradients['b'][1] = np.mean(G, axis=1, keepdims=True)
        
        G = self.network['W'][1].T @ G
        G = G * (S1 > 0)

        gradients['W'][0] = (G @ H.T) / n + 2 * lam * self.network['W'][0]
        gradients['b'][0] = np.mean(G, axis=1, keepdims=True)
        
        G_batch = self.network['W'][0].T @ G
        G_batch = G_batch * (H > 0)
        
        if 'dropout_mask' in fp_data:
            G_batch = G_batch * fp_data['dropout_mask']
        
        gradients['b_conv'] = np.sum(G_batch.reshape((n_p, n_f, n), order='C'), axis=(0, 2)) / n

        GG = G_batch.reshape((n_p, n_f, n), order='C')
        
        MXt = np.transpose(MX, (1, 0, 2))
        grad_Fs_flat = np.einsum('ijn, jln -> il', MXt, GG, optimize=True) / n
        grad_Fs_flat = grad_Fs_flat + 2 * lam * self.network['conv_filters'].reshape((f * f * 3, n_f), order='C')
        
        return gradients, grad_Fs_flat
    
    # Update network parameters using computed gradients
    def updateNetwork(self, eta, grads, grad_Fs_flat):
        f, n_f = self.CN_params['f'], self.CN_params['n_f']

        
        for k in range(2):
            self.network['W'][k] -= eta * grads['W'][k]
            self.network['b'][k] -= eta * grads['b'][k]
        self.network['conv_filters'] -= eta * grad_Fs_flat.reshape((f, f, 3, n_f), order='C')
        self.network['b_conv'] -= eta * grads['b_conv']