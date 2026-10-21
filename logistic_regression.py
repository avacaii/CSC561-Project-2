# Name: Ava Cai
# Date: 10/21/26
# Assignment: Project 2
# Purpose of the program: Logistic Regression Class
# AI DECLARATION: AI was not used in the creation of this program.

import numpy as np
from variable_class import Variable

class LogisticRegression:
    def __init__(self, learning_rate=0.1, iterations=1000):
        self.learning_rate = learning_rate # how large each gradient descent update should be
        self.iterations = iterations # how many times we update the parameters during training
        self.b = None # Creates a placeholder for the bias b
        self.w = None # Creates a placeholder for the weights w

    def fit(self, X, y):
        # X contains the input features
        # y contains the correct labels (0 or 1) 
        n, k = X.shape # n = number of training examples; k = number of features per example
        
        # Create variables for the bias and weights
        b = Variable("b") #symbolic variable representing the bias

        weights = [] 

        # need one weight for every feature
        # will create a list of the weight variables (e.g. "w_0" and "w_1")
        for j in range(k):
            weights.append(Variable("w_" + str(j)))

        # Initialize all parameter values to zero
        values = {"b": 0.0} # dictionary containing the value of the variables (b, w_0, w_1,...)

        # add all the weights and their initial value into the dictionary 
        for j in range(k):
            values["w_" + str(j)] = 0.0

        # Start building the total loss expression
        cost = 0

        # Loop through each training example
        for i in range(n):
            # Build z = b + w0*x0 + w1*x1 + ...
            z = b
            for j in range(k):
                # weights[j] is the symbolic weight for feature j
                # X[i, j] is the numerical value of feature j for training example i
                z = z + weights[j] * X[i, j]

            # Apply sigmoid to get the predicted probability
            # logistic regression needs to convert z into a probability between 0 and 1 
            p = 1 / (1 + Variable.exp(-z))

            # Calculate cross-entropy loss for this example
            loss = -(y[i] * Variable.log(p)
                    + (1 - y[i]) * Variable.log(1 - p))

            # Add this example's loss to the total
            cost = cost + loss

        # Divide by the number of examples to get average loss
        cost = cost / n

        # Create list to track the changes in loss
        self.loss_history = []

        # Use gradient descent to minimize the cost
        for iteration in range(self.iterations):
            # Record the current cost before updating the parameters
            self.loss_history.append(cost.evaluate(values))
            
            # Calling cost.grad(values) recursively calculates the partial derivative 
            # of the cost with respect to every weight and bias
            grads = cost.grad(values)

            # Update the bias using its gradient
            values["b"] = values["b"] - self.learning_rate * grads["b"]

            # Update each weight using its gradient
            for j in range(k):
                name = "w_" + str(j)
                values[name] = values[name] - self.learning_rate * grads[name]

        # save the final learned weights and bias as attributes of the LogisticRegression object
        self.b = values["b"]
        self.w = np.array([values["w_" + str(j)] for j in range(k)]) # creates a numpy array of all the weights 

    # use the learned weights and bias to make predictions
    def predict(self, X):
        z = X @ self.w + self.b # @ is the numpy matrix multiplier 

        # apply Sigmoid function to convert the weighted sums into probabilities between 0 and 1
        # use np.exp() instead of Variable.exp() because z is now a NumPy array of numerical values
        p = 1 / (1 + np.exp(-z))

        # Convert probabilities into class labels
        # p >= 0.5 will return Boolean values, .astype(int) converts the Boolean values into integers,
        # 1 for True, 0 for False
        # Essentially, return 1 for probabilities >= 0.5, otherwise return 0
        return (p >= 0.5).astype(int)
