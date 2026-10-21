# Logistic Regression Class and Variable Class

A Python implementation of logistic regression that uses gradient descent for training and a custom class (`variable_class.py`) to compute gradients.

**Author:** Ava Cai
**Course:** CSC 561
**Assignment:** Project 2

## Overview

A Python implementation of logistic regression that uses gradient descent for training and a custom class (`variable_class.py`) to compute gradients.

The model performs binary classification, predicting whether an input belongs to class 0 or class 1. Unlike implementations that manually calculate the derivatives of the logistic regression loss function, this model uses a custom implementation of a mathematical expression tree of `Variable` objects to compute gradients.

The implementation follows a similar interface to scikit-learn, with `fit()` for training and `predict()` for making predictions.

## How It Works

### Logistic Regression

Logistic regression is a supervised machine learning algorithm used for binary classification.

For each input example, the model calculates a weighted sum of its features and a bias:

`z = w₀x₀ + w₁x₁ + ... + wₖxₖ + b`

The result is passed through the sigmoid function:

`p = 1 / (1 + e^(-z))`

This produces a probability between 0 and 1. The model predicts class 1 when the probability is greater than or equal to 0.5 and class 0 otherwise.

### Loss Function

The model uses binary cross-entropy to measure the difference between predicted probabilities and actual labels:

`L = -[y log(p) + (1-y) log(1-p)]`

The cost function is the average loss across all training examples.

### Automatic Differentiation

The model uses the custom `Variable` class in `variable_class.py` to calculate gradients.

Each mathematical operation creates a node in an expression tree. These nodes implement `evaluate()` to calculate the expression's numerical value and `grad()` to recursively compute its partial derivatives using differentiation rules such as the sum, product, power, and chain rules.

During training, the logistic regression model builds an expression tree representing the cost function. It then calls `cost.grad(values)` to calculate the gradients with respect to every weight and the bias.

### Gradient Descent

The model uses gradient descent to minimize the cost function.

During each training iteration, the weights and bias are updated according to:

`parameter = parameter - learning_rate * gradient`

This process repeats for the specified number of iterations. The final parameter values are stored and used to make predictions.

## Dependencies

- Python 3
- NumPy
- Matplotlib (for testing and visualization)
- scikit-learn (for evaluation)
- `variable_class.py` (custom class)

## LogisticRegression Class

### Parameters

| Parameter | Default | Description |
|---|---|---|
| `learning_rate` | `0.1` | Controls the size of each gradient descent update. |
| `iterations` | `1000` | Number of gradient descent iterations performed during training. |

### Attributes

| Attribute | Description |
|---|---|
| `w` | NumPy array containing the learned weights, one for each input feature. |
| `b` | Learned bias parameter. |

Both attributes are initialized to `None` and assigned numerical values after training.

### Methods

**`fit(X, y)`**

Trains the logistic regression model using gradient descent.

- `X`: A two-dimensional NumPy array of shape `(n_samples, n_features)` containing the training features.
- `y`: A one-dimensional NumPy array of shape `(n_samples,)` containing binary labels (0 or 1).

The method creates symbolic variables for each weight and the bias, constructs the binary cross-entropy cost function, and uses automatic differentiation to update the parameters.

**`predict(X)`**

Predicts binary class labels using the learned weights and bias.

- `X`: A two-dimensional NumPy array of shape `(n_samples, n_features)` containing input features.

Returns a one-dimensional NumPy array containing predicted labels (0 or 1).

## Usage Example

The following example generates a dataset with one feature and trains the model to classify examples based on whether the feature value is approximately greater than 2.

```python
import numpy as np
from logistic_regression import LogisticRegression

np.random.seed(0)

# Generate 50 examples with one feature
X = np.random.uniform(0, 4, size=(50, 1))

# Generate binary labels with random noise
y = (X[:, 0] + np.random.normal(0, 0.5, 50) > 2).astype(int)

# Train the model
model = LogisticRegression(learning_rate=0.1, iterations=1000)
model.fit(X, y)

# Make predictions
predictions = model.predict(X)

# Evaluate training accuracy
accuracy = np.mean(predictions == y)

print("Weights:", model.w)
print("Bias:", model.b)
print("Training accuracy:", accuracy)
```

## Testing and Evaluation

The implementation is evaluated on previously unseen test data to determine whether it learns the expected relationships and generalizes to new examples.

### Single-Feature Classification

A synthetic dataset of 50 examples is generated with one feature ranging from 0 to 4. Binary labels are assigned using a threshold of approximately 2, with random noise added.

This test evaluates whether the model can learn a simple binary classification relationship.

**Results:** [Insert learned parameters and training/test accuracy.]

### Multiple-Feature Classification

A separate dataset with multiple input features can be used to verify that the model correctly creates and updates one weight for each feature.

**Results:** [Insert the number of features, learned weights, and accuracy.]

### Gradient Descent Convergence

The average cost can be recorded across training iterations and plotted to determine whether gradient descent successfully reduces the loss.

**Results:** [Insert loss graph and observations.]

### Quality of Fit

For a single-feature dataset, the model's predicted probability curve can be plotted against the actual labels to visually evaluate the fit.

**Results:** [Insert graph and observations.]

## Assumptions and Limitations

- The implementation supports two classes, represented by 0 and 1. It does not currently support multiclass classification.
- Directly calculating sigmoid probabilities and logarithms may cause numerical errors when probabilities approach 0 or 1.
- The recursive automatic differentiation implementation may repeatedly evaluate parts of the expression tree, making training slower than optimized machine learning libraries.
- Features with very different numerical ranges may affect gradient descent convergence. The implementation does not automatically standardize features.
- Training runs for a specified number of iterations rather than stopping automatically when the cost converges.

## Potential Applications

This model can be used for binary classification tasks where input features are numerical and the relationship can be approximated by a linear decision boundary. Examples include predicting whether a student passes an exam based on study habits or classifying whether a patient is positive for a diagnosis.

## References

- [Scikit-learn: Automatic Relevance Determination Regression Example](https://scikit-learn.org/stable/auto_examples/linear_model/plot_ard.html)
- [Scikit-learn: Linear Models User Guide](https://scikit-learn.org/stable/modules/linear_model.html)
- [Scikit-learn: LogisticRegression API Reference](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
- [Scikit-learn GitHub README](https://github.com/scikit-learn/scikit-learn/blob/main/README.rst)
- [Make a README](https://www.makeareadme.com/)