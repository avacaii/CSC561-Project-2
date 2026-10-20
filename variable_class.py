import math

# each Variable object represents an independent mathematical variable
class Variable: 

    # name is an attribute of the Variable class and is the key used to look up it's value
    def __init__(self, name=None): # None means that the name is optional  
        self.name = name

    # the evaluation method for independent variables, base case
    # purpose of evaluate() is to figure out the numerical value of a variable
    # by looking up the value assoiated with its name in the values dictionary 
    def evaluate(self, values):
        # values will be a dictionary that looks like: {"x_1": 3, "x_2": -4}
        return values[self.name]

    # the gradient of a single variable, base case
    # d(x_1)/d(x_1) = 1, and d(x_1)/d(anything else) = 0
    # returns a dictionary with one partial derivative per variable,
    # e.g. for x_1 with values {"x_1": 3, "x_2": -4} -> {"x_1": 1, "x_2": 0}
    def grad(self, values):
        result = {} # creates an empty dictionary, will have the partial derivatives for every variable
        
        for name in values:
            if name == self.name: # checks if var currently differentiating with is var represented by this object
                result[name] = 1 # if so, derivative is 1 
            else:
                result[name] = 0 # if taking a deriv. of another var will be 0, treated as constant
        
        return result

    # Use operator overloading so Python knows how to perform operations on Variable class objects 
    # Instead of calculating a number immediately, each operation creates a new
    # expression node that can later be evaluated and differentiated
    # Since all the operation classes inherit from Variable, we can combine
    # simple variables and more complicated expressions using the same operators

    # x + y, creates an addition node 
    def __add__(self, other):
        return AdditionVariable(self, wrap(other)) # wrap() converts regular numbers into Constant objects

    # const + x, handle addition when the number is on the left
    def __radd__(self, other):
        return AdditionVariable(wrap(other), self)

    # x * y, create a multiplication node
    def __mul__(self, other):
        return MultiplicationVariable(self, wrap(other))

    # const * x, handle multiplication when the number is on the left
    def __rmul__(self, other):
        return MultiplicationVariable(wrap(other), self)

    # -x, rewrite as -1 * x so we can reuse multiplication
    # -1 is on the left, so uses __rmul__ 
    def __neg__(self):
        return -1 * self

    # Subtraction can be rewritten as addition of a negative
    # No separate SubtractionVariable class is needed,
    # AdditionVariable and MultiplicationVariable already handle evaluate() and grad()
    # x - y becomes x + (-1 * y)
    def __sub__(self, other):
        return self + (-1 * wrap(other))

    # const - x becomes const + (-1 * x)
    # handle subtraction when the number is on the left
    def __rsub__(self, other):
        return wrap(other) + (-1 * self)

    # Division is rewritten as multiplying by the reciprocal
    # No separate DivisionVariable class is needed,
    # MultiplicationVariable and PowerVariable already handle evaluate() and grad()
    # (the product rule + power rule together give the quotient rule)
    # x / y becomes x * y^(-1)
    def __truediv__(self, other):
        return self * (wrap(other) ** (-1))

    # const / x becomes const * x^(-1)
    # handle division when the number is on the left
    def __rtruediv__(self, other):
        return wrap(other) * (self ** (-1))

    # x ** p, create a power node
    # p must be a constant number so we can use the standard power rule
    def __pow__(self, p):
        return PowerVariable(self, p)

    # e^x, create a node representing e^x
    def exp(self):
        return ExpVariable(wrap(self))

    # ln(x), create a node representing ln(x)
    def log(self):
        return LogVariable(wrap(self))


# Represents a constant number in an expression
# Inherits from Variable so constants can use the same operators as variables 
class Constant(Variable):
    def __init__(self, value):
        self.value = value # store the constant's value as an attribute

    # Return the stored number's value, not related to input values
    def evaluate(self, values):
        return self.value

    # A constant doesn't depend on any variable, so its partial derivative is always 0
    def grad(self, values):
        
        result = {} # create a dictionary to store the combined gradients

        # partial derivative with respect to every variable should be 0 
        for name in values:
            result[name] = 0
        
        return result

# Convert regular numbers into Constant objects so they can be used
# alongside Variable objects in expression trees.
def wrap(x):
    # If x is already a Variable or subclass, leave it unchanged
    if isinstance(x, Variable):
        return x
    # If not, create a Constant object containing x
    return Constant(x)


# Operation nodes represent mathematical operations in the expression tree.
# Each operation has two children, left and right, which can be
# individual variables, constants, or other expressions.

# Represents the addition of two expressions: f + g
class AdditionVariable(Variable):
    def __init__(self, left, right):
        # Store the two expressions being added as attributes.
        self.left = left   
        self.right = right  

    # Override evaluate() to calculate the sum of the two expressions
    def evaluate(self, values):
        # recursively evaluate each child, then add their results
        return self.left.evaluate(values) + self.right.evaluate(values)

    # Use the sum rule, the derivative of f + g is f' + g'
    # Override grad() to calculate the partial derivatives of the expression
    def grad(self, values):
        # recursively calculate the gradients of both children.
        left_grad = self.left.grad(values)    
        right_grad = self.right.grad(values)

        result = {} # create a dictionary to store the combined gradients
        
        # Add the partial derivatives of both children for each variable
        for name in values:
            result[name] = left_grad[name] + right_grad[name]
        
        return result

# Represents the multiplication of two expressions: f * g
class MultiplicationVariable(Variable):
    def __init__(self, left, right):
        # Store the two expressions being multiplied as attributes
        self.left = left    
        self.right = right 

    # recursively evaluate both sides, then multiply the results
    def evaluate(self, values):
        return self.left.evaluate(values) * self.right.evaluate(values)

    # Use the product rule: the derivative of f * g is f' * g + f * g'
    # Unlike the sum rule, this needs the values of f and g themselves,
    # not just their derivatives
    def grad(self, values):
        # Calculate the values of both children
        f = self.left.evaluate(values)
        g = self.right.evaluate(values)

        # Recursively calculate the gradients of both children
        f_grad = self.left.grad(values) 
        g_grad = self.right.grad(values)  

        result = {} # create a dictionary to store the combined gradients

        # Apply the product rule for each variable
        for name in values:
            result[name] = f_grad[name] * g + f * g_grad[name]
        
        return result

# Represents an expression raised to a constant power: f^p
class PowerVariable(Variable):
    def __init__(self, left, p):
        # Store the base expression and its constant exponent
        self.left = left  
        self.p = p       

    # Recursively evaluate the base, then raise it to p
    def evaluate(self, values):
        return self.left.evaluate(values) ** self.p

    # Use the power rule and chain rule:
    # The derivative of f^p is p * f^(p-1) * f'
    def grad(self, values):
        # Calculate the value of the base expression
        f = self.left.evaluate(values)

        # Recursively calculate the gradient of the base expression
        f_grad = self.left.grad(values) 

        # Apply the power rule to the outer function
        outer = self.p * f ** (self.p - 1)

        result = {} # create a dictionary to store the combined gradients

        # Apply the chain rule by multiplying by f' for each variable
        for name in values:
            result[name] = outer * f_grad[name] 
        
        return result

# Represents the exponential of an expression: e^f
class ExpVariable(Variable):
    def __init__(self, inner):
        # Store the expression in the exponent
        self.inner = inner

    # Recursively evaluate the exponent, then calculate e^f
    def evaluate(self, values):
        return math.exp(self.inner.evaluate(values))

    # Use the chain rule: the derivative of e^f is e^f * f'
    def grad(self, values):
        # Calculate the value of the inner expression
        f = self.inner.evaluate(values)

        # Recursively calculate the gradient of the inner expression
        f_grad = self.inner.grad(values)

        # Calculate the derivative of the outer exponential function
        # Since the derivative of e^x is e^x, this is just e^f
        outer = math.exp(f)
        
        result = {} # create a dictionary to store the combined gradients

        # Apply the chain rule by multiplying by f' for each variable
        for name in values:
            result[name] = outer * f_grad[name]
        
        return result

# Represents the natural logarithm of an expression: ln(f)
class LogVariable(Variable):
    def __init__(self, inner):
        # Store the expression inside the logarithm
        self.inner = inner

    # Recursively evaluate the inner expression, then calculate ln(f)
    # (the inner expression must be positive, otherwise math.log raises an error)
    def evaluate(self, values):
        return math.log(self.inner.evaluate(values))

    # Use the chain rule: the derivative of ln(f) is (1/f) * f'
    def grad(self, values):
        # Calculate the value of the inner expression
        f = self.inner.evaluate(values)

        # Recursively calculate the gradient of the inner expression
        f_grad = self.inner.grad(values) 

        # Apply the derivative rule for the natural logarithm, 1/f
        outer = 1 / f        

        result = {} # create a dictionary to store the combined gradients

        # Apply the chain rule by multiplying by f' for each variable
        for name in values:
            result[name] = outer * f_grad[name]
        
        return result