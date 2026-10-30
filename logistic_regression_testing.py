# Name: Ava Cai
# Date: 10/21/26
# Assignment: Project 2
# Purpose of the program: Test the Logistic Regression Class
# AI DECLARATION: AI was not used in the creation of this program.

import numpy as np
from logistic_regression import LogisticRegression

# Fit model to data
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# use a seed so results are reproducible
# np.random.seed(0) makes NumPy generate the same sequence of random numbers each time 
np.random.seed(0)

# Generate 50 examples with one feature between 0 and 4
X = np.random.uniform(0, 4, size=(50, 1))
# Assign labels based on whether the feature is approximately above 2
# Add random noise with np.random.normal so that it is not perfectly split at 2
y = ((X[:, 0] + np.random.normal(0, 0.5, 50)) > 2).astype(int)

# Split data into training and testing sets
# This way we can evaluate the model on data that it was not trained on
# so we can better see how accurate it is
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train on the training data
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict labels for the unseen testing data
y_preds = model.predict(X_test)

# Evaluate prediction accuracy
print("Test accuracy:", accuracy_score(y_test, y_preds))


import matplotlib.pyplot as plt

# Graph the changes in loss 
# Confirms that gradient descent improves the loss 
plt.figure()
plt.plot(model.loss_history)
plt.xlabel("Iteration")
plt.ylabel("Average loss")
plt.title("Loss During Gradient Descent")


# Graph the output against the data to test quality of fit
# Create evenly spaced x-values between 0 and 4 for the probability curve
# Learned this from: https://www.geeksforgeeks.org/python/numpy-linspace/ 
X_plot = np.linspace(0, 4, 200).reshape(200, 1)

# Calculate predicted probabilities using the learned parameters
z = X_plot @ model.w + model.b
probabilities = 1 / (1 + np.exp(-z))

# Plot the actual testing data
plt.figure()
plt.scatter(X_test[:, 0], y_test, label="Actual labels")

# Plot the logistic regression probability curve
plt.plot(X_plot[:, 0], probabilities, label="Predicted probability")

plt.xlabel("Feature value")
plt.ylabel("Probability / Class")
plt.title("Logistic Regression Fit")
plt.legend()



# Test to see if model learns multiple features
np.random.seed(1)

# Generate 100 examples with 3 features each
X = np.random.uniform(0, 4, size=(100, 3))

# Generate labels based on a combination of the features.
y = (
    X[:, 0] + 2 * X[:, 1] - X[:, 2]
    + np.random.normal(0, 0.5, 100) > 4
).astype(int)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train on the training data
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict labels for the unseen testing data
y_preds = model.predict(X_test)

# Evaluate prediction accuracy
print("Test accuracy (mult. features):", accuracy_score(y_test, y_preds))

# Verify that the model learned one weight per feature
assert len(model.w) == 3


# Unit tests for each piece of the LogisticRegression class
# Each test uses small, hand-picked numbers so the correct answer can be worked out by hand
print("\nUnit Tests")

# _weight_name should build the name used to look up each weight in the values dictionary
model = LogisticRegression()
assert model._weight_name(0) == "w_0"
assert model._weight_name(5) == "w_5"
print("_weight_name passed")


# __init__ should reject a negative decay, since it would make the learning rate grow
try:
    LogisticRegression(decay=-1)
    assert False, "a negative decay should raise a ValueError"
except ValueError:
    pass
print("__init__ decay check passed")


# _init_parameters should make one weight per feature and start every parameter at zero
b, weights, values = model._init_parameters(2)
assert b.name == "b"
assert [w.name for w in weights] == ["w_0", "w_1"]
assert values == {"b": 0.0, "w_0": 0.0, "w_1": 0.0}
print("_init_parameters passed")


# _build_cost with every parameter at zero
# z = 0 for every example, so p = sigmoid(0) = 0.5
# the loss for each example is -log(0.5) = log(2), no matter what the label is
# so the average loss should also be log(2)
X = np.array([[1.0], [2.0], [3.0]])
y = np.array([0, 1, 1])
b, weights, values = model._init_parameters(1)
cost = model._build_cost(X, y, b, weights)
assert np.isclose(cost.evaluate(values), np.log(2))
print("_build_cost (all zeros) passed")


# _build_cost with known parameters on a single example
# with b = 0, w_0 = 1, x = 1, y = 1: z = 1, p = sigmoid(1)
# the loss is -log(sigmoid(1)) = log(1 + e^-1)
X_one = np.array([[1.0]])
y_one = np.array([1])
b, weights, values = model._init_parameters(1)
cost_one = model._build_cost(X_one, y_one, b, weights)
assert np.isclose(cost_one.evaluate({"b": 0.0, "w_0": 1.0}), np.log(1 + np.exp(-1)))
print("_build_cost (known weights) passed")


# Gradient of the cost at zero
# for cross-entropy loss, the partial derivatives are:
# dJ/db   = average of (p - y)
# dJ/dw_0 = average of (p - y) * x
# with every parameter at zero, p = 0.5 for every example
grads = cost.grad(values)
p = 0.5
assert np.isclose(grads["b"], np.mean(p - y))
assert np.isclose(grads["w_0"], np.mean((p - y) * X[:, 0]))
print("cost gradient passed")


# _current_learning_rate should stay the same when there is no decay
assert model._current_learning_rate(0) == 0.1
assert model._current_learning_rate(500) == 0.1

# with decay = 0.5, at iteration 2: lr = 0.1 / (1 + 0.5 * 2) = 0.05
decay_model = LogisticRegression(learning_rate=0.1, decay=0.5)
assert np.isclose(decay_model._current_learning_rate(2), 0.05)
print("_current_learning_rate passed")


# _update_parameters should move each parameter opposite to its gradient
# b:   1.0 - 0.1 * 0.5  = 0.95
# w_0: 2.0 - 0.1 * -1.0 = 2.1
values = {"b": 1.0, "w_0": 2.0}
grads = {"b": 0.5, "w_0": -1.0}
model._update_parameters(values, grads, 1, 0.1)
assert np.isclose(values["b"], 0.95)
assert np.isclose(values["w_0"], 2.1)
print("_update_parameters passed")


# fit should record one loss and one learning rate per iteration,
# and gradient descent should lower the loss
model = LogisticRegression(iterations=50, decay=0.1)
model.fit(X, y)
assert len(model.loss_history) == 50
assert len(model.lr_history) == 50
assert model.loss_history[-1] < model.loss_history[0]

# with decay, every learning rate should be smaller than the one before it
for i in range(1, 50):
    assert model.lr_history[i] < model.lr_history[i - 1]

# fit should save the weights as a NumPy array with one weight per feature
assert isinstance(model.w, np.ndarray)
assert model.w.shape == (1,)
print("fit passed")


# predict with hand-picked parameters
# with w = [1] and b = -2, z = x - 2, so the decision boundary is at x = 2
# x = 0 -> p < 0.5 -> 0
# x = 2 -> p = 0.5 exactly -> 1, since predict uses p >= 0.5
# x = 3 -> p > 0.5 -> 1
model = LogisticRegression()
model.w = np.array([1.0])
model.b = -2.0
assert list(model.predict(np.array([[0.0], [2.0], [3.0]]))) == [0, 1, 1]
print("predict passed")

print("All unit tests passed")


# Compare my implementation with scikit-learn's LogisticRegression
# import it under a different name so it doesn't replace my LogisticRegression class
from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression

print("\nComparison with scikit-learn")

# Regenerate the same single-feature data from the first test
np.random.seed(0)
X = np.random.uniform(0, 4, size=(50, 1))
y = ((X[:, 0] + np.random.normal(0, 0.5, 50)) > 2).astype(int)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train my model
my_model = LogisticRegression()
my_model.fit(X_train, y_train)

# Train scikit-learn's model
# scikit-learn adds L2 regularization by default, which my model does not have,
# so penalty=None makes both models minimize the same cross-entropy cost
sk_model = SklearnLogisticRegression(penalty=None)
sk_model.fit(X_train, y_train)

# Compare the learned parameters
# coef_ holds the weights and intercept_ holds the bias
print("My model:     w =", my_model.w, " b =", my_model.b)
print("scikit-learn: w =", sk_model.coef_[0], " b =", sk_model.intercept_[0])

# Compare accuracy on the unseen testing data
my_preds = my_model.predict(X_test)
sk_preds = sk_model.predict(X_test)
print("My test accuracy:          ", accuracy_score(y_test, my_preds))
print("scikit-learn test accuracy:", accuracy_score(y_test, sk_preds))

# Check how often the two models predict the same label
print("Prediction agreement:      ", np.mean(my_preds == sk_preds))

# Graph both probability curves against the data
X_plot = np.linspace(0, 4, 200).reshape(200, 1)

# My model's probabilities, calculated from the learned parameters
my_probabilities = 1 / (1 + np.exp(-(X_plot @ my_model.w + my_model.b)))

# predict_proba returns one column per class, column 1 is the probability of class 1
# Learned this from: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
sk_probabilities = sk_model.predict_proba(X_plot)[:, 1]

plt.figure()
plt.scatter(X_test[:, 0], y_test, label="Actual labels")
plt.plot(X_plot[:, 0], my_probabilities, label="My model")
plt.plot(X_plot[:, 0], sk_probabilities, linestyle="--", label="scikit-learn")

plt.xlabel("Feature value")
plt.ylabel("Probability / Class")
plt.title("My Model vs. scikit-learn")
plt.legend()

# Show all graphs at once
plt.show()