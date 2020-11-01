# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.6.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # Simple Neural Network - Binary Classification
# ###### Neural network from scratch

# +
import numpy as np
import matplotlib.pyplot as plt
# binary classification data:
from data_binary_classification import plot_decision_boundary, load_2D_dataset
# multi-class classification data:
from data_multiclass import load_multiclass_dataset
import warnings

# ignore warnings by message (caused by log of zeros/close-to-zero numbers during training):
warnings.filterwarnings('ignore', message='divide by zero encountered in log')
warnings.filterwarnings('ignore', message='invalid value encountered in multiply')


# -

class NN_Model():
    __PRINT_COST_INTERVAL = 2000    # the interval (i.e. iteration #) when the cost should be printed
    __EPSILON = 1e-8                # a small value greater than zero
    
    def __init__(self,
                 X,
                 Y,
                 hidden_layers,
                 classification_type='binary',
                 learning_rate=0.01,
                 learning_rate_decay_rate=0,
                 num_iterations=10000,
                 regularization=None,
                 lambda_=0.1,
                 initialization=None,
                 optimizer=None,
                 print_cost=False):
        np.random.seed(1)

        self.X = X # inputs
        self.Y = Y # labels

        self.num_iterations = num_iterations
        # we use the term 'initial' learning rate because we will often use learning rate decay:
        self.initial_learning_rate = learning_rate
        # n.b. the decay rate for alpha should be non-negative (otherwise, the learning rate would increase every epoch):
        self.learning_rate_decay_rate = learning_rate_decay_rate if learning_rate_decay_rate >= 0 else 0
        self.print_cost = print_cost                        # boolean - determines if costs per interval should be printed
        self.regularization = str(regularization).upper()   # type of regularization to be implemented
        self.lambda_ = lambda_                              # lambda constant used for L2 regularization equation
        self.classification_type = classification_type      # whether model is binary classification, multi-class classification, etc.

        self.b = {}             # bias unit (trainable); each key represents layer
        self.W = {}             # weights (trainable); each key represents layer
        self.Z = {}             # linear unit; each key represents layer
        self.A = {0: X}         # post-activation unit; each key represents layer; 0th layer is just input layer
        self.db = {}            # gradient of bias unit; each key represents layer
        self.dW = {}            # gradient of weights; each key represents layer
        self.dA = {}            # gradient of post-activation unit; each key represents layer
        self.costs = []         # the cost at a given interval/number of iterations, __PRINT_COST_INTERVAL
        self.m = X.shape[1]     # number of training examples

        self.is_training = True         # boolean to indicate training vs testing/predicting
        self.dropout_mask = {}          # the mask of boolean values (per layer) used for dropout regularization

        # array of layer shapes (nodes per layer), including input and output layers:
        # in binary classification, the last layer is a single neuron (for sigmoid activation), so we append a layer of size 1
        # for multi-class classification (e.g. softmax) the last layer is the size of the number of classes (i.e. the # of rows in Y)
        if classification_type == 'binary':
            self.layers = [{'size': X.shape[0]}] + hidden_layers + [{'size': 1}]
        elif classification_type == 'multi-class' or classification_type == 'multiclass' or classification_type == 'softmax':
            self.layers = [{'size': X.shape[0]}] + hidden_layers + [{'size': Y.shape[0]}]
            # reassign the type to 'softmax' so that we don't have to keep checking variations of the name in our code:
            self.classification_type = 'softmax'
        else:
            raise TypeError('invalid "classification_type" argument -- set parameter as "binary" or "softmax"')

        self.L = len(self.layers)-1  # number of layers

        # for gradient optimization:
        # TODO: make Adam beta values user tunable; though, tuning them is rare:
        self.optimizer = optimizer.upper()
        self.beta_RMS = 0.999
        self.beta_momentum = 0.9
        self.moving_avg_dW = {}  # aka exponentially weighted average of dW -- its "velocity" (v)
        self.moving_avg_db = {}  # aka exponentially weighted average of db -- its "velocity" (v)
        self.RMS_dW = {}  # aka exponentially weighted average of squares of dW
        self.RMS_db = {}  # aka exponentially weighted average of squares of db

        # initialize parameters, W and b:
        for l in range(1, self.L+1):
            # ensure hidden_layers was properly passed as a list of dictionaries:
            # TODO: n.b. code will fail anyway if hidden layers is not properly defined,
            #  so this is check is mostly for the additional error hint and may not be that useful
            if type(self.layers[l]) != dict and 'size' not in self.layers[l]:
                raise TypeError('each item in "layers" must be a dict and must have a property of "size"')

            # determine layer sizes:
            layer_size = self.layers[l]['size']
            previous_layer_size = self.layers[l-1]['size']

            self.W[l] = np.random.randn(layer_size, previous_layer_size)
            self.b[l] = np.zeros((layer_size, 1))

            # weights initialization (scaling weights):
            initialization = str(initialization).upper()

            if initialization == 'HE':
                self.W[l] *= 2 / np.sqrt(previous_layer_size)
            elif initialization == 'XAVIER':
                self.W[l] *= 1 / np.sqrt(previous_layer_size)
            else:
                self.W[l] *= 0.01

            # initialize velocity (if selected) to zeros:
            W_shape = self.W[l].shape
            b_shape = self.b[l].shape
            if self.optimizer == 'MOMENTUM' or self.optimizer == 'ADAM':
                self.moving_avg_dW[l] = np.zeros(W_shape)
                self.moving_avg_db[l] = np.zeros(b_shape)
            if self.optimizer == 'RMSPROP' or self.optimizer == 'ADAM':
                self.RMS_dW[l] = np.zeros(W_shape)
                self.RMS_db[l] = np.zeros(b_shape)

    # ########## model training functions: ##########
    def relu(self, Z):
        return np.maximum(Z, 0)
    
    def leaky_relu(self, Z):
        return np.where(Z > 0, Z, 0.01*Z)

    def sigmoid(self, Z):
        return 1 / (1 + np.exp(-Z))

    def relu_gradient(self, Z):
        return np.where(Z > 0, 1, 0)
    
    def leaky_relu_gradient(self, Z):
        return np.where(Z > 0, 1, 0.01)

    def sigmoid_gradient(self, Z):
        return self.sigmoid(Z) * (1 - self.sigmoid(Z))

    def softmax(self, Z):
        t = np.exp(Z)

        # the exponential can become so large that even float64 is unable to handle it, resulting in an overflow warning and resulting in NaN values.
        # the NaN values can be effectively replaced with zeros (and thus, preventing our algorithm from breaking):
        #t[np.isnan(t)] = 0

        return t/np.sum(t, axis=0, keepdims=True)

    def forward_prop(self):
        for l in range(1, self.L+1):
            Z = np.dot(self.W[l], self.A[l-1]) + self.b[l]
            self.Z[l] = np.copy(Z)

            # in the last layer of our network, we apply either sigmoid or softmax activation (for binary or softmax classification, respectively):
            if l == self.L:
                if self.classification_type == 'binary':
                    self.A[l] = self.sigmoid(Z)
                elif self.classification_type == 'softmax':
                    self.A[l] = self.softmax(Z)
            else:
                self.A[l] = self.relu(Z)

            # if dropout is set for layer:
            # need to scale the values of neurons that weren't shut down:
            if self.is_training and (type(self.layers[l]) == dict) and ('keep_prob' in self.layers[l]):
                # create matrix of boolean values (with same shape as A)
                # we set value to `1` if RNG is greater than the probability to keep the neuron (`keep_prob`):
                self.dropout_mask[l] = np.random.random((self.A[l].shape[0], self.A[l].shape[1])) < self.layers[l]['keep_prob']
                self.A[l] *= self.dropout_mask[l] # apply mask
                self.A[l] /= self.layers[l]['keep_prob'] # scale by keep_probability

        Y_hat = self.A[self.L]
        return Y_hat

    def back_prop(self):
        for l in reversed(range(1, self.L+1)):
            if l == self.L:
                dZ = self.A[l] - self.Y

                # # alternatively, we can compute dZ for layer L the long way -- replace the previous line with:
                # self.dA[l] = -(np.divide(self.Y, self.A[l]) - np.divide(1-self.Y, 1-self.A[l]))
                # dZ = self.dA[l] * self.sigmoid_gradient(self.Z[l]) 
            else:
                dZ = self.dA[l] * self.relu_gradient(self.Z[l])

            self.db[l] = 1.0/self.m * np.sum(dZ, axis=1, keepdims=True)
            self.dA[l-1] = np.dot(self.W[l].T, dZ)
            self.dW[l] = 1.0/self.m * np.dot(dZ, self.A[l-1].T)

            # if dropout is set for layer,
            # apply dropout mask to the same neurons as in forward prop:
            if self.is_training and (type(self.layers[l-1]) == dict) and ('keep_prob' in self.layers[l-1]):
                self.dA[l-1] *= self.dropout_mask[l-1]
                self.dA[l-1] /= self.layers[l-1]['keep_prob']

            # if L2 regularization is set:
            if self.regularization == 'L2':
                 self.dW[l] += self.lambda_/self.m * self.W[l]

    def update_parameters(self, iteration):
        """ Update parameters step during gradient descent. """

        # if decay rate for learning rate (alpha) has been set, update learning rate by epoch:
        if self.learning_rate_decay_rate and iteration % 100 == 0:
            # n.b. there are *several* variations of the learning rate decay formula
            learning_rate = self.initial_learning_rate / (1 + (self.learning_rate_decay_rate * iteration))

            # alternate learning rate decay formula:
            # learning_rate = np.power(0.95, iteration) * self.initial_learning_rate
        else:
            learning_rate = self.initial_learning_rate

        for l in range(1, self.L+1):
            # update either moving avg or RMS of gradients:
            if self.optimizer == 'MOMENTUM' or self.optimizer == 'ADAM':
                self.moving_avg_dW[l] = (self.beta_momentum * self.moving_avg_dW[l]) + ((1-self.beta_momentum) * self.dW[l])
                self.moving_avg_db[l] = (self.beta_momentum * self.moving_avg_db[l]) + ((1-self.beta_momentum) * self.db[l])

                # optional bias correction for exponentially weight avg (i.e. for it to "warm up"):
                moving_avg_dW_corrected = self.moving_avg_dW[l] / (1-np.power(self.beta_momentum, iteration))
                moving_avg_db_corrected = self.moving_avg_db[l] / (1-np.power(self.beta_momentum, iteration))

            if self.optimizer == 'RMSPROP' or self.optimizer == 'ADAM':
                self.RMS_dW[l] = (self.beta_RMS * self.RMS_dW[l]) + ((1-self.beta_RMS) * np.square(self.dW[l]))
                self.RMS_db[l] = (self.beta_RMS * self.RMS_db[l]) + ((1-self.beta_RMS) * np.square(self.db[l]))

                # optional bias correction for exponentially weight avgs of squares:
                RMS_dW_corrected = self.RMS_dW[l] / (1-np.power(self.beta_RMS, iteration))
                RMS_db_corrected = self.RMS_db[l] / (1-np.power(self.beta_RMS, iteration))

            # update parameters based on optimizer algorithm:
            if self.optimizer == 'MOMENTUM':
                self.W[l] -= learning_rate * self.moving_avg_dW[l]
                self.b[l] -= learning_rate * self.moving_avg_db[l]
            elif self.optimizer == 'RMSPROP':
                # n.b. epsilon added to prevent division by zero:
                self.W[l] -= learning_rate * (self.dW[l] / (np.sqrt(RMS_dW_corrected) + self.__EPSILON))
                self.b[l] -= learning_rate * (self.db[l] / (np.sqrt(RMS_db_corrected) + self.__EPSILON))
            elif self.optimizer == 'ADAM':
                # combining moving average of gradients (corrected) and moving average of squared gradients (corrected) gives us Adam:
                # n.b. epsilon added to prevent division by zero:
                self.W[l] -= learning_rate * (moving_avg_dW_corrected / (np.sqrt(RMS_dW_corrected) + self.__EPSILON))
                self.b[l] -= learning_rate * (moving_avg_db_corrected / (np.sqrt(RMS_db_corrected) + self.__EPSILON))
            else:
                self.W[l] -= learning_rate * self.dW[l]
                self.b[l] -= learning_rate * self.db[l]

    def train(self):
        """ Trains the model after initialization. """
        self.is_training = True
        self.costs = []

        self.print_hyperparameters()

        for i in range(1, self.num_iterations+1):
            Y_hat = self.forward_prop()
            self.back_prop()

            if (i == 1) or (i % self.__PRINT_COST_INTERVAL == 0):
                cost = self.compute_cost(Y_hat)
                self.costs.append(cost)

                # we only want to print cost during training for debugging purposes:
                if self.print_cost:
                    print('cost at iteration %s: %s' % (i, cost))

            self.update_parameters(i)

    # ########## public functions: ##########
    def print_hyperparameters(self):
        print('*** Training model with the following hyperparameters: ***')
        print('learning rate (alpha): ', self.initial_learning_rate)
        print('number of iterations:', self.num_iterations)
        if self.optimizer == 'MOMENTUM':
            print('using gradient descent with momentum')
        elif self.optimizer == 'ADAM':
            print('using Adam optimizer')
        else:
            print('using gradient descent')
        print()  # just to create newline

        # TODO: print other hyper parameters
        # TODO: maybe format output a bit?

    def plot_cost(self):
        """ 
        Plots training costs.
        
        Used for debugging training algorithm and hyperparameters.
        """

        plt.plot(self.costs)
        plt.title('Cost per Iteration')
        plt.xlabel('Iteration (x%s)' % self.__PRINT_COST_INTERVAL)
        plt.ylabel('Cost')
        plt.show()

    def print_accuracy(self, X, Y):
        """ Prints the accuracy of train/test sets. """
        predictions = self.predict(X)
        
        accuracy = np.mean(np.int8(predictions == Y))
        print('Accuracy: %s%%' % (accuracy * 100))


# + pycharm={"name": "#%%\n"}
class NN_Binary(NN_Model):
    def __init__(self,
                  X,
                  Y,
                  hidden_layers,
                  classification_type='binary',
                  learning_rate=0.01,
                  learning_rate_decay_rate=0,
                  num_iterations=10000,
                  regularization=None,
                  lambda_=0.1,
                  initialization=None,
                  optimizer=None,
                  print_cost=False):
        NN_Model.__init__(self,
                          X,
                          Y,
                          hidden_layers,
                          classification_type,
                          learning_rate,
                          learning_rate_decay_rate,
                          num_iterations,
                          regularization,
                          lambda_,
                          initialization,
                          optimizer,
                          print_cost)

    def compute_cost(self, Y_hat):
        """
        Computes cost of binary classification (i.e. sigmoid function in last hidden layer)

        If specified, function also computes cost of L2 regularization implementation.

        :param Y_hat: Final layer from forward propagation.
        :return: Cost
        """
        # cross entropy part of cost function:
        cross_entropy_cost = -1/self.m * np.nansum( (self.Y*np.log(Y_hat)) + ((1-self.Y) * np.log(1-Y_hat)) )
        cross_entropy_cost = np.squeeze(cross_entropy_cost)

        if self.regularization == 'L2':
            L2_regularization_cost = 1/self.m * self.lambda_/2 * np.sum([np.sum(np.square(self.W[l])) for l in range(1, self.L+1)])
            cost = cross_entropy_cost + L2_regularization_cost
        else:
            cost = cross_entropy_cost

        return cost

    # ########## public functions: ##########
    def predict(self, X):
        """ Predict """
        self.is_training = False
        self.A[0] = X
        self.m = X.shape[1]

        Y_hat = self.forward_prop()

        predictions = np.where(Y_hat > 0.5, 1, 0)
        return predictions


# + pycharm={"name": "#%%\n"}
# N.b. requisite dimensions of X = (# of features, # of examples)
# N.b. requisite dimensions of Y = (# nodes in final layer, # of examples)
class NN_Softmax(NN_Model):
    def __init__(self,
                  X,
                  Y,
                  hidden_layers,
                  classification_type='softmax',
                  learning_rate=0.01,
                  learning_rate_decay_rate=0,
                  num_iterations=20000,
                  regularization=None,
                  lambda_=0.1,
                  initialization=None,
                  optimizer=None,
                  print_cost=False):

        NN_Model.__init__(self,
                          X,
                          Y,
                          hidden_layers,
                          classification_type,
                          learning_rate,
                          learning_rate_decay_rate,
                          num_iterations,
                          regularization,
                          lambda_,
                          initialization,
                          optimizer,
                          print_cost)

        # TODO: should we convert labels to one-hot encoding inside the instance or during data processing?:
        # self.Y = self.to_one_hot_encoding(self.Y)

    @staticmethod
    def to_one_hot_encoding(labels):
        """ Converts multi-class data to array to one-hot encoding.

            How this method works: The identity matrix is essentially one-hot encoding already.
            We then use array slices to map the labels (array) to the correct indices.

            N.b. the returned identity matrix is transposed so that each column represents a single 1-hot encoding.
            N.b. this function won't work if any of the label values are negative or non-integers.
        """
        assert type(labels) == type(np.array([]))

        # flatten array:
        labels = labels.reshape(-1)

        # need to get the max value because the identity matrix needs to have a unique 1-hot encoding for each value up to the max:
        # the `+ 1` is due to zero-indexing:
        num_values = np.max(labels) + 1
        # n.b. the transpose:
        return np.eye(num_values)[labels].T

    @staticmethod
    def to_categorical(one_hot_encoding):
        """ Converts one-hot encoding back to categorical data.

        :param one_hot_encoding:
        :return:
        """
        assert type(one_hot_encoding) == type(np.array([]))

        return one_hot_encoding.argmax(axis=0)

    def compute_cost(self, Y_hat):
        """
        Computes cost of softmax/multi-class classification.

        TODO: update to also calculate cost w/ L2 regularization implemented.

        :param Y_hat: Output from final layer in forward propagation.
        :return: Cost
        """

        return -1/self.m * np.sum(np.log(Y_hat) * self.Y)

    def predict(self, X):
        """ Predict class """
        self.is_training = False
        self.A[0] = X
        self.m = X.shape[1]

        Y_hat = self.forward_prop()

        # convert predictions (which are probabilities) to one-hot:
        # see <https://stackoverflow.com/a/42397281>:
        predictions = (Y_hat == Y_hat.max(axis=0, keepdims=True)).astype(int)
        return predictions

    def print_accuracy(self, X, Y):
        """ Prints the accuracy of train/test sets. """
        predictions = self.predict(X)

        Y_hat_classes = np.argmax(predictions, axis=0)
        Y_classes = np. argmax(Y, axis=0)

        accuracy = np.mean(np.int8(Y_classes == Y_hat_classes))
        print('Accuracy: %s%%' % (accuracy * 100))


# -

# ## Main
#
# #### Binary Classification:

# + pycharm={"name": "#%%\n"}
# import data for binary classification:
bin_class_train_X, bin_class_train_Y, bin_class_test_X, bin_class_test_Y = load_2D_dataset()

# +
# create model:

# set hyper parameters:
train_hidden_layers = [{'size': 20, 'keep_prob': 0.95}, {'size': 3, 'keep_prob': 1}]

binary_model = NN_Binary(bin_class_train_X,
                        bin_class_train_Y,
                        train_hidden_layers,
                        learning_rate=0.1,
                        learning_rate_decay_rate=1e-4,
                        num_iterations=50000,
                        regularization='l2',
                        initialization='xavier',
                        optimizer='adam',
                        print_cost=True)

# + pycharm={"name": "#%%\n"}
# train model:
binary_model.train()

# + pycharm={"name": "#%%\n"}
# print accuracy:

print('Training set:')
binary_model.print_accuracy(bin_class_train_X, bin_class_train_Y)
print('Test set:')
binary_model.print_accuracy(bin_class_test_X, bin_class_test_Y)

# plot decision boundary after training:

plt.title("Model Decision Boundaries")
axes = plt.gca()
axes.set_xlim([-0.75,0.40])
axes.set_ylim([-0.75,0.65])
plot_decision_boundary(lambda x: binary_model.predict(x.T), bin_class_train_X, np.squeeze(bin_class_train_Y))
# -

# #### Multi-class/Softmax Classification:

# + pycharm={"name": "#%%\n"}
# Spectral Data:
from data_multiclass import generate_spectral_data

multiclass_train_X, multiclass_train_Y = generate_spectral_data()

# transpose input data to correct dimensions:
multiclass_train_X = multiclass_train_X.T

# fix multiclass_train_Y labels to be 0-indexed -- i.e. in range of [0,2] rather than range of [1,3]:
# transform to rank-one array:
multiclass_train_Y -= 1
multiclass_train_Y = multiclass_train_Y.reshape(300, -1)

# convert to one-hot-array:
multiclass_train_Y = NN_Softmax.to_one_hot_encoding(multiclass_train_Y)

# + pycharm={"name": "#%%\n"}
# create model:

# set hyper parameters:
train_hidden_layers = [{'size': 20}, {'size':6}]

multiclass_model = NN_Softmax(multiclass_train_X,
                              multiclass_train_Y,
                              train_hidden_layers,
                              learning_rate=0.1,
                              learning_rate_decay_rate=1e-4,
                              num_iterations=20000,
                              initialization='xavier',
                              optimizer='adam',
                              print_cost=True)

# + pycharm={"name": "#%%\n"}
multiclass_model.train()

# + pycharm={"name": "#%%\n"}
multiclass_model.print_accuracy(multiclass_train_X, multiclass_train_Y)

# + pycharm={"name": "#%%\n"}

