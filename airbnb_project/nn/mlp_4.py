import numpy as np
import matplotlib.pyplot as plt

class TwoLayerMLP_4(object):
    """
    A two-layer fully-connected neural network. The net has an input dimension of
    N, a hidden layer dimension of H, and performs classification over C classes.
    We train the network with a softmax loss function and L2 regularization on the
    weight matrices. The network uses a ReLU nonlinearity after the first fully
    connected layer.

    In other words, the network has the following architecture:

    input - fully connected layer - ReLU - fully connected layer - softmax

    The outputs of the second fully-connected layer are the scores for each class.
    """

    def __init__(self, input_size, hidden_size_1, hidden_size_2, output_size, std=1e-4, activation='relu'):
        """
        Initialize the model. Weights are initialized to small random values and
        biases are initialized to zero. Weights and biases are stored in the
        variable self.params, which is a dictionary with the following keys:

        W1: First layer weights; has shape (D, H)
        b1: First layer biases; has shape (H,)
        W2: Second layer weights; has shape (H, C)
        b2: Second layer biases; has shape (C,)

        Inputs:
        - input_size: The dimension D of the input data.
        - hidden_size: The number of neurons H in the hidden layer.
        - output_size: The number of classes C.
        """
        self.params = {}
        self.params['W1'] = np.sqrt(2/hidden_size_1) * np.random.randn(input_size, hidden_size_1)
        self.params['b1'] = np.zeros(hidden_size_1)
        self.params['W2'] = np.sqrt(2/hidden_size_2) * np.random.randn(hidden_size_1, hidden_size_2)
        self.params['b2'] = np.zeros(hidden_size_2)
        self.params['W3'] = np.sqrt(2/output_size) * np.random.randn(hidden_size_2, output_size)
        self.params['b3'] = np.zeros(output_size)
        
        self.activation = activation
        
        

    def loss(self, X, y=None, reg=0.0):
        """
        Compute the loss and gradients for a two layer fully connected neural
        network.

        Inputs:
        - X: Input data of shape (N, D). Each X[i] is a training sample.
        - y: Vector of training labels. y[i] is the label for X[i], and each y[i] is
          an integer in the range 0 <= y[i] < C. This parameter is optional; if it
          is not passed then we only return scores, and if it is passed then we
          instead return the loss and gradients.
        - reg: Regularization strength.

        Returns:
        If y is None, return a matrix scores of shape (N, C) where scores[i, c] is
        the score for class c on input X[i].

        If y is not None, instead return a tuple of:
        - loss: Loss (data loss and regularization loss) for this batch of training
          samples.
        - grads: Dictionary mapping parameter names to gradients of those parameters
          with respect to the loss function; has the same keys as self.params.
        """
        # Unpack variables from the params dictionary
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']
        _, C = W2.shape
        N, D = X.shape

        y = y.reshape((N, 1))

        # Compute the forward pass
        scores = None
        #############################################################################
        z1 = np.dot(X, W1) + b1  # 1st layer activation, N*H

        # 1st layer nonlinearity, N*H
        if self.activation is 'relu':
            hidden_1 = np.maximum(0, z1)        

        z2 = np.dot(hidden_1, W2) + b2  # 2nd layer activation, N*C
        #############################################################################
        
        # 1st layer nonlinearity, N*H
        if self.activation is 'relu':
            hidden_2 = np.maximum(0, z2)        
            
        scores = np.dot(hidden_2, W3) + b3
        
        
        # If the targets are not given then jump out, we're done
        if y is None:
            return scores

        loss = 0.5*np.mean(np.square(scores - y))#np.mean(np.abs(a2 - y))#np.mean(0.5 * np.square(a2 - y))
        # add regularization terms
        loss = np.mean(np.abs(scores - y))
        loss += 0.5 * reg * np.sum(W1 * W1)
        loss += 0.5 * reg * np.sum(W2 * W2)
        loss += 0.5 * reg * np.sum(W3 * W3)

        # Backward pass: compute gradients
        grads = {}

        #############################################################################
#         dscore = np.square(y - scores)
#         dscore = (scores - y)
        dscore = 2*(scores - y > 0).astype(int) - 1
        
        dW3 = np.dot(hidden_2.T,dscore)
        db3 = np.mean(dscore,axis = 0)
        
        # hidden layer 2
        dhidden_2 = np.dot(dscore, W3.T)  # N*H
        
        if self.activation is 'relu':
            dz2 = dhidden_2
            dz2[z2 <= 0] = 0.1
        
        dW2 = np.dot(hidden_1.T, dz2)   # H*C
        db2 = np.mean(dz2, axis=0)  # C

        # hidden layer
        dhidden = np.dot(dz2, W2.T)  # N*H
        if self.activation is 'relu':
            dz1 = dhidden
            dz1[z1 <= 0] = 0.1

        dW1 = np.dot(X.T, dz1)  # D*H
        db1 = np.mean(dz1, axis=0)  # D
        
        #############################################################################
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
        """
        Train this neural network using stochastic gradient descent.

        Inputs:
        - X: A numpy array of shape (N, D) giving training data.
        - y: A numpy array f shape (N,) giving training labels; y[i] = c means that
          X[i] has label c, where 0 <= c < C.
        - X_val: A numpy array of shape (N_val, D) giving validation data.
        - y_val: A numpy array of shape (N_val,) giving validation labels.
        - learning_rate: Scalar giving learning rate for optimization.
        - learning_rate_decay: Scalar giving factor used to decay the learning rate
          after each epoch.
        - reg: Scalar giving regularization strength.
        - num_iters: Number of steps to take when optimizing.
        - batch_size: Number of training examples to use per step.
        - verbose: boolean; if true print progress during optimization.
        """
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
        """
        Use the trained weights of this two-layer network to predict labels for
        data points. For each data point we predict scores for each of the C
        classes, and assign each data point to the class with the highest score.

        Inputs:
        - X: A numpy array of shape (N, D) giving N D-dimensional data points to
          classify.

        Returns:
        - y_pred: A numpy array of shape (N,) giving predicted labels for each of
          the elements of X. For all i, y_pred[i] = c means that X[i] is predicted
          to have class c, where 0 <= c < C.
        """
        y_pred = None
        
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']
        
#         print(X.shape,W1.shape)
        z1 = np.dot(X, W1) + b1

        if self.activation == 'relu':
            a1 = np.maximum(z1, 0)
            
        z2 = np.dot(a1, W2) + b2
        
        if self.activation == 'relu':
            a2 = np.maximum(z2, 0)
            
        y_pred = np.dot(a2, W3) + b3

        
        ###########################################################################
#         scores = self.loss(X)
#         y_pred = np.argmax(scores, axis=1)
        ###########################################################################

        return y_pred


