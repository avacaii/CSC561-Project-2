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

# Show both graphs at once
plt.show()
