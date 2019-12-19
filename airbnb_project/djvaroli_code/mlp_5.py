import numpy as np
import matplotlib.pyplot as plt

class TwoLayerMLP_5(object):

    def __init__(self, input_size, hidden_size_1, hidden_size_2, hidden_size_3, output_size, std=1e-4, activation='relu'):
        self.params = {}
        self.params['W1'] = np.sqrt(2/hidden_size_1) * np.random.randn(input_size, hidden_size_1)
        self.params['b1'] = np.zeros((1,hidden_size_1))
        self.params['W2'] = np.sqrt(2/hidden_size_2) * np.random.randn(hidden_size_1, hidden_size_2)
        self.params['b2'] = np.zeros((1,hidden_size_2))
        self.params['W3'] = np.sqrt(2/hidden_size_3) * np.random.randn(hidden_size_2, hidden_size_3)
        self.params['b3'] = np.zeros((1,hidden_size_3))
        self.params['W4'] = np.sqrt(2/output_size) * np.random.randn(hidden_size_3, output_size)
        self.params['b4'] = np.zeros((1,output_size))
        
        self.activation = activation
        
        

    def loss(self, X, y=None, reg=0.0):
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']
        W4, b4 = self.params['W4'], self.params['b4']
        _, C = W2.shape
        N, D = X.shape
        
        b1 = b1.reshape((1,-1))
        b2 = b2.reshape((1,-1))
        b3 = b3.reshape((1,-1))
        b4 = b4.reshape((1,-1))
        
        y = y.reshape((N, 1))

        # Compute the forward pass
        z1 = np.dot(X, W1) + b1  # 1st layer activation, N*H
        if self.activation is 'relu':
            hidden_1 = np.maximum(0, z1)        

        z2 = np.dot(hidden_1, W2) + b2  # 2nd layer activation, N*C
        if self.activation is 'relu':
            hidden_2 = np.maximum(0, z2)
            
        z3 = np.dot(hidden_2,W3) + b3
        if self.activation is 'relu':
            hidden_3 = np.maximum(0, z3)
            
        scores = np.dot(hidden_3, W4) + b4
        
        scores = scores.reshape((-1,1))
#         loss = 0.5*np.mean(np.square(scores - y))#np.mean(np.abs(a2 - y))#np.mean(0.5 * np.square(a2 - y))
        loss = np.mean(np.abs(scores - y))
        loss += 0.5 * reg * np.sum(W1 * W1)
        loss += 0.5 * reg * np.sum(W2 * W2)
        loss += 0.5 * reg * np.sum(W3 * W3)
        loss += 0.5 * reg * np.sum(W4 * W4)

        # Backward pass: compute gradients
        grads = {}

        #############################################################################
#         dscore = np.square(y - scores)
#         dscore = (scores - y)
        dscore = 2*(scores - y > 0).astype(int) - 1
        dscore = dscore.reshape((-1,1))
        
        dW4 = np.dot(hidden_3.T,dscore)
        db4 = np.mean(dscore,axis = 0)
        dhidden_3 = np.dot(dscore, W4.T)  # N*H
        if self.activation is 'relu':
            dz3 = dhidden_3
            dz3[z3 <= 0] = 0.01
             
        dW3 = np.dot(hidden_2.T,dscore)
        db3 = np.mean(dz3,axis = 0)
        dhidden_2 = np.dot(dz3, W3.T)  # N*H
        if self.activation is 'relu':
            dz2 = dhidden_2
            dz2[z2 <= 0] = 0.01
        
        dW2 = np.dot(hidden_1.T, dz2)  # H*C
        db2 = np.mean(dz2, axis=0)  # C
        dhidden = np.dot(dz2, W2.T)  # N*H
        if self.activation is 'relu':
            dz1 = dhidden
            dz1[z1 <= 0] = 0.01

        dW1 = np.dot(X.T, dz1)/N # D*H
        db1 = np.mean(dz1, axis=0)  # D
        
        #############################################################################
        grads['W4'] = dW4 + reg*W4
        grads['b4'] = db4
        grads['W3'] = dW3 + reg*W3
        grads['b3'] = db3
        grads['W2'] = dW2 + reg*W2
        grads['b2'] = db2
        grads['W1'] = dW1 + reg*W1
        grads['b1'] = db1
        return loss, grads


    def train(self, X, y, X_val, y_val,
            learning_rate=1e-3, learning_rate_decay=0.95,
            reg=1e-5, num_epochs=10,
            batch_size=200, verbose=False):

        num_train = X.shape[0]
        iterations_per_epoch = max(int(num_train/batch_size), 1)
        epoch_num = 0

        # Use SGD to optimize the parameters in self.model
        loss_history = []
        grad_magnitude_history = []
        train_acc_history = []
        val_acc_history = []

        np.random.seed(1)
#         print(self.params)
        for epoch in range(num_epochs):
            # fixed permutation (within this epoch) of training data
            perm = np.random.permutation(num_train)

            # go through minibatches
            for it in range(iterations_per_epoch):
                X_batch = None
                y_batch = None

                idx = perm[it*batch_size:(it+1)*batch_size]
                X_batch = X[idx, :]
                y_batch = y[idx]

                # Compute loss and gradients using the current minibatch
                loss, grads = self.loss(X_batch, y=y_batch, reg=reg)
                loss_history.append(loss)

                for param in self.params:
                    self.params[param] -= grads[param] * learning_rate

                # record gradient magnitude (Frobenius) for W1
                grad_magnitude_history.append(np.linalg.norm(grads['W1']))

            # Every epoch, check train and val accuracy and decay learning rate.
            # Check accuracy
            train_acc = np.sqrt((np.square(self.predict(X_batch) - y_batch)).mean())
            val_acc = np.sqrt((np.square(self.predict(X_val) - y_val)).mean())
            train_acc_history.append(train_acc)
            val_acc_history.append(val_acc)
            if verbose:
                print('Epoch %d: loss %f, train_acc %f, val_acc %f'%(
                    epoch+1, loss, train_acc, val_acc))

            # Decay learning rate
            learning_rate *= learning_rate_decay
        
#         print(self.params)
        return {
          'loss_history': loss_history,
          'grad_magnitude_history': grad_magnitude_history, 
          'train_acc_history': train_acc_history,
          'val_acc_history': val_acc_history,
        }

    def predict(self, X):   
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']
        W4, b4 = self.params['W4'], self.params['b4']
        
        z1 = np.dot(X, W1) + b1
        if self.activation == 'relu':
            a1 = np.maximum(z1, 0)
            
        z2 = np.dot(a1, W2) + b2
        if self.activation == 'relu':
            a2 = np.maximum(z2, 0)
        
        z3 = np.dot(a2,W3) + b3
        if self.activation == 'relu':
            a3 = np.maximum(z3, 0)
         
        y_pred = np.dot(a3, W4) + b4

        return y_pred


