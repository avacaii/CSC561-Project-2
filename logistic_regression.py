# Name: Ava Cai
# Date: 10/21/26
# Assignment: Project 2
# Purpose of the program: Logistic Regression Class
# AI DECLARATION: AI was not used in the creation of this program.

import numpy as np
from variable_class import Variable

class LogisticRegression:
    def __init__(self, learning_rate=0.1, iterations=1000, decay=0.0):
        # a negative decay would make the learning rate grow instead of shrink
        if decay < 0:
            raise ValueError("decay must be non-negative")

        self.learning_rate = learning_rate # how large each gradient descent update should be at the start
        self.iterations = iterations # how many times we update the parameters during training
        self.decay = decay # how quickly the learning rate shrinks over time (0 means no decay)
        self.b = None # Creates a placeholder for the bias b
        self.w = None # Creates a placeholder for the weights w

    # returns the name of the weight for feature j (e.g. "w_0", "w_1")
    def _weight_name(self, j):
        return "w_" + str(j)

    # create the symbolic variables for the bias and weights, and set their starting values
    def _init_parameters(self, k):
        # k = number of features per example
        b = Variable("b") #symbolic variable representing the bias

        weights = []

        # need one weight for every feature
        # will create a list of the weight variables (e.g. "w_0" and "w_1")
        for j in range(k):
            weights.append(Variable(self._weight_name(j)))

        # Initialize all parameter values to zero
        values = {"b": 0.0} # dictionary containing the value of the variables (b, w_0, w_1,...)

        # add all the weights and their initial value into the dictionary
        for j in range(k):
            values[self._weight_name(j)] = 0.0

        return b, weights, values

    # build the symbolic expression for the average cross-entropy loss over all training examples
    def _build_cost(self, X, y, b, weights):
        n, k = X.shape # n = number of training examples; k = number of features per example

        # Start building the total loss expression
        cost = 0

        # Loop through each training example to make linear combination of input features
        # Learned this from:
        # https://www.geeksforgeeks.org/machine-learning/understanding-logistic-regression/
        for i in range(n):
            # Build z = b + w0*x0 + w1*x1 + ...
            z = b
            for j in range(k):
                # weights[j] is the symbolic weight for feature j
                # X[i, j] is the numerical value of feature j for training example i
                z = z + weights[j] * X[i, j]

            # Apply sigmoid to get the predicted probability
            # logistic regression needs to convert z into a probability between 0 and 1
            # Learned this from:
            # https://www.geeksforgeeks.org/machine-learning/understanding-logistic-regression/
            p = 1 / (1 + Variable.exp(-z))

            # Calculate cross-entropy loss for this example
            # Learned this from:
            # https://www.geeksforgeeks.org/machine-learning/ml-cost-function-in-logistic-regression/
            loss = -(y[i] * Variable.log(p)
                    + (1 - y[i]) * Variable.log(1 - p))

            # Add this example's loss to the total
            cost = cost + loss

        # Divide by the number of examples to get average loss
        return cost / n

    # compute the learning rate for this iteration using time-based decay
    # lr = learning_rate / (1 + decay * iteration)
    # I learned this from: https://www.geeksforgeeks.org/machine-learning/learning-rate-decay/
    def _current_learning_rate(self, iteration):
        return self.learning_rate / (1 + self.decay * iteration)

    # perform one gradient descent step on the bias and every weight
    def _update_parameters(self, values, grads, k, lr):
        # lr = the learning rate for this iteration
        # Update the bias using its gradient
        values["b"] = values["b"] - lr * grads["b"]

        # Update each weight using its gradient
        for j in range(k):
            name = self._weight_name(j)
            values[name] = values[name] - lr * grads[name]

    def fit(self, X, y):
        # X contains the input features
        # y contains the correct labels (0 or 1)
        k = X.shape[1] # k = number of features per example

        # Create variables for the bias and weights, all starting at zero
        b, weights, values = self._init_parameters(k)

        # Build the total loss expression that gradient descent will minimize
        cost = self._build_cost(X, y, b, weights)

        # Create lists to track the changes in loss and learning rate
        self.loss_history = []
        self.lr_history = []

        # Use gradient descent to minimize the cost
        for iteration in range(self.iterations):
            # Record the current cost before updating the parameters
            self.loss_history.append(cost.evaluate(values))

            # Shrink the learning rate as training goes on
            lr = self._current_learning_rate(iteration)
            self.lr_history.append(lr)

            # Calling cost.grad(values) recursively calculates the partial derivative
            # of the cost with respect to every weight and bias
            grads = cost.grad(values)

            # move each parameter a small step in the direction that lowers the cost
            self._update_parameters(values, grads, k, lr)

        # save the final learned weights and bias as attributes of the LogisticRegression object
        self.b = values["b"]
        self.w = np.array([values[self._weight_name(j)] for j in range(k)]) # creates a numpy array of all the weights

    # use the learned weights and bias to make predictions
    def predict(self, X):
        # learned this from: https://numpy.org/doc/stable/reference/generated/numpy.matmul.html 
        z = X @ self.w + self.b # @ is the numpy matrix multiplier 

        # apply Sigmoid function to convert the weighted sums into probabilities between 0 and 1
        # use np.exp() instead of Variable.exp() because z is now a NumPy array of numerical values
        p = 1 / (1 + np.exp(-z))

        # Convert probabilities into class labels
        # p >= 0.5 will return Boolean values, .astype(int) converts the Boolean values into integers,
        # 1 for True, 0 for False
        # Essentially, return 1 for probabilities >= 0.5, otherwise return 0
        return (p >= 0.5).astype(int)
