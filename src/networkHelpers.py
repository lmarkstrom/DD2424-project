import numpy as np

def softmax(S):
    S_exp = np.exp(S - np.max(S, axis=0, keepdims=True))
    P = S_exp / np.sum(S_exp, axis=0, keepdims=True)
    return P

def computeAccuracy(P, y):
    pred = np.argmax(P, axis=0)
    return np.mean(pred == y)

def computeLoss(P, Y):
    P_correct = np.clip(P, 1e-15, 1.0)
    loss = np.mean(np.sum(-Y * np.log(P_correct), axis=0))
    
    return loss

def computeCost(P, Y, network, lam):
    W1 = network['W'][0]
    W2 = network['W'][1]
    CF = network['conv_filters']
    
    P_correct = np.clip(P, 1e-15, 1.0)
    loss = np.mean(np.sum(-Y * np.log(P_correct), axis=0))
    l2 = np.sum(W1 ** 2) + np.sum(W2 ** 2) + np.sum(CF ** 2)
    
    return loss + lam * l2

def generateMX(X, n, f, n_p):
    X_ims = np.transpose(X.reshape((32, 32, 3, n), order='F'), (1, 0, 2, 3))
    MX = np.zeros((n_p, f * f * 3, n))
    for i in range(n):
        l = 0
        for x in range(32 // f):
            for y in range(32 // f):
                X_patch = X_ims[x*f:x*f+f, y*f:y*f+f, :, i]
                MX[l, :, i] = X_patch.reshape((1, f*f*3)).flatten(order='C')
                l += 1
    
    return MX

def learningRateSchedule(LR_params, n_s, steps_in_cycle):
    eta_min, eta_max = LR_params['etas']

    if steps_in_cycle < n_s:
        fraction = steps_in_cycle / n_s
        eta_t = eta_min + fraction * (eta_max - eta_min)
    else:
        fraction = (steps_in_cycle - n_s) / n_s
        eta_t = eta_max - fraction * (eta_max - eta_min)
        
    return eta_t

def adamOptimizer(adam, network, grads, eta, t, beta1=0.9, beta2=0.999, eps=1e-8):
    curr_t = t + 1
    
    for i in range(2):
        adam['m_W'][i] = beta1 * adam['m_W'][i] + (1 - beta1) * grads['W'][i]
        adam['v_W'][i] = beta2 * adam['v_W'][i] + (1 - beta2) * (grads['W'][i]**2)
        
        m_hat = adam['m_W'][i] / (1 - beta1**curr_t)
        v_hat = adam['v_W'][i] / (1 - beta2**curr_t)
        
        network['W'][i] -= eta * m_hat / (np.sqrt(v_hat) + eps)

        adam['m_b'][i] = beta1 * adam['m_b'][i] + (1 - beta1) * grads['b'][i]
        adam['v_b'][i] = beta2 * adam['v_b'][i] + (1 - beta2) * (grads['b'][i]**2)
        
        m_hat_b = adam['m_b'][i] / (1 - beta1**curr_t)
        v_hat_b = adam['v_b'][i] / (1 - beta2**curr_t)
        
        network['b'][i] -= eta * m_hat_b / (np.sqrt(v_hat_b) + eps)