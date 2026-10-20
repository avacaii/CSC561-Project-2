from variable_class import Variable, Constant

# Values to plug into every expression below
values = {"x_1": 3, "x_2": -4}

# Base variables, each one looks up its value by name in `values`
x_1 = Variable(name="x_1")
x_2 = Variable(name="x_2")

# Print an expression's value and gradient on one line
def show(label, expr):
    print(f"{label:<20} value = {expr.evaluate(values):<10.4g} grad = {expr.grad(values)}")


# Single Variables (base cases)
# A Variable has derivative 1 with respect to itself and 0 for every other variable
print("Single Variables")
show("x_1", x_1) # 3                    
show("x_2", x_2) # -4                   
# A Constant doesn't depend on any variable, so every partial derivative is 0
show("Constant(5)", Constant(5)) # 5      


# Addition
print("\nAddition")
show("x_1 + x_2", x_1 + x_2) # -1        
show("x_1 + 3", x_1 + 3) # 6
show("-1 + x_2", -1 + x_2) # -5         
show("(x_1 + x_2) + x_1", (x_1 + x_2) + x_1) # 2


# Negation
print("\nNegation")
show("-x_1", -x_1) # -3                
show("-(x_1 + x_2)", -(x_1 + x_2)) # 1 


# Multiplication
print("\nMultiplication")
show("x_1 * x_2", x_1 * x_2) # -12 
show("x_1 * 9", x_1 * 9) # 27  
show("5 * x_1", 5 * x_1) # 15             
show("x_1 * x_1", x_1 * x_1) # 9    


# Subtraction
print("\nSubtraction")
show("x_1 - x_2", x_1 - x_2) # 7
show("x_2 - x_1", x_2 - x_1) # -7 
show("x_1 - 3", x_1 - 3) # 0
show("4 - x_1", 4 - x_1) # 1
show("x_1 - x_1", x_1 - x_1) # 0


# Division
print("\nDivision")
show("x_1 / x_2", x_1 / x_2) # -0.75
show("x_2 / x_1", x_2 / x_1) # -1.333
show("x_2 / 2", x_2 / 2) # -2
show("12 / x_1", 12 / x_1) # 4


# Powers
print("\nPowers")
show("x_1 ** 2", x_1 ** 2) # 9
show("x_2 ** 3", x_2 ** 3) # -64
show("x_1 ** 0.5", x_1 ** 0.5) # 1.732
show("x_1 ** -1", x_1 ** -1) # 0.333
show("(x_1 + x_2) ** 2", (x_1 + x_2) ** 2) # 1


# Exponential
print("\nExponential")
show("e^(x_1)", x_1.exp()) # 20.09
show("e^(x_1 + x_2)", (x_1 + x_2).exp()) # 0.368
show("e^0", Variable.exp(0)) # 1


# Natural log
print("\nNatural log")
show("ln(x_1)", x_1.log()) # 1.099
show("ln(2 * x_1)", (2 * x_1).log()) # 1.792
show("ln(e^(x_2))", x_2.exp().log()) # -4
show("ln(x_1^2 + x_2^2)", (x_1 ** 2 + x_2 ** 2).log()) # 3.219


# Order of operations
# Checks if follows Python's normal order of precedence for operations
print("\nOrder of operations")
show("x_1 + x_2 * 2", x_1 + x_2 * 2) # -5
show("(x_1 + x_2) * 2", (x_1 + x_2) * 2) # -2


# Error cases
# These inputs will cause errors to be raised 
print("\nError cases")

# ln is only defined for positive numbers, x_2 has a negative value
try:
    x_2.log().evaluate(values)
except ValueError as e:
    print("ln(x_2):", type(e).__name__, "-", e)

# x_1 - 3 = 0, so this divides by zero
try:
    (x_1 / (x_1 - 3)).evaluate(values)
except ZeroDivisionError as e:
    print("x_1 / (x_1 - 3):", type(e).__name__, "-", e)

# x_3 has no entry in the values dictionary, so its lookup fails
try:
    x_3 = Variable(name="x_3")
    (x_1 + x_3).evaluate(values)
except KeyError as e:
    print("x_1 + x_3:", type(e).__name__, "- no value given for", e)
