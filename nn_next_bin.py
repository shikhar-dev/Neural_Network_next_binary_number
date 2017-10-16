# For Proper Explanation see file 'Descrition and Equations' :)

import numpy as np


# Activation Fucntion
def sigmoid(x, deriv=False):
    if (deriv == True):
        return np.multiply(x, 1 - x)
    return 1 / (1 + np.exp(-x))


# Size of Training Data Set
m = 10001

# Number of features
n = 14

# Generating test data set

X = np.array([i for i in range(m)])  # Generating binary representaions of all numbers from 0 - 16000
X = (((X[:, None] & (1 << np.arange(n)))) > 0).astype(int)
for i in range(len(X)):
    X[i] = X[i][::-1]
X = X.T

# Generating Ground Truth Set

Y = np.array([i for i in range(1, m + 1)])
Y = (((Y[:, None] & (1 << np.arange(n)))) > 0).astype(int)
for i in range(len(Y)):
    Y[i] = Y[i][::-1]
Y = Y.T

# seeding random numbers

np.random.seed(1)

# weight matrix with mean zero
w1 = 2 * np.random.random((4, n)) - 1
b1 = 2 * np.random.random((4, 1)) - 1

w2 = 2 * np.random.random((4, 4)) - 1
b2 = 2 * np.random.random((4, 1)) - 1

w3 = 2 * np.random.random((n, 4)) - 1
b3 = 2 * np.random.random((n, 1)) - 1

# Learning Rate
alpha = 0.01

# Number of iterations
niter = 10000

# Main Loop
for iter in xrange(niter):
    # Foraward Propogation
    z1 = np.dot(w1, X) + b1  # Hidden Layer 1
    a1 = sigmoid(z1)

    z2 = np.dot(w2, a1) + b2  # Hidden Layer 2
    a2 = sigmoid(z2)

    z3 = np.dot(w3, a2) + b3
    a3 = sigmoid(z3)  # Ouptut Layer

    print 'ERROR : ', np.sum(Y - a3, axis=1, keepdims=True).T / m

    # Backward Propogation
    dz3 = a3 - Y  # Output Layer
    dw3 = 1 / m * np.dot(dz3, a2.T)
    db3 = np.sum(dz3, axis=1, keepdims=True) / m

    dz2 = np.multiply(np.dot(w3.T, dz3), sigmoid(sigmoid(z2), True))  # Hidden Layer 2
    dw2 = 1 / m * np.dot(dz2, a1.T)
    db2 = np.sum(dz2, axis=1, keepdims=True) / m

    dz1 = np.multiply(np.dot(w2.T, dz2), sigmoid(sigmoid(z1), True))  # Hidden Layer 1
    dw1 = 1 / m * np.dot(dz1, X.T)
    db1 = np.sum(dz1, axis=1, keepdims=True) / m

    # Gradient Descent 1 Iteration
    w3 = w3 - alpha * dw3  # Layer 3
    b3 = b3 - alpha * db3

    w2 = w2 - alpha * dw2  # layer 2
    b2 = b2 - alpha * db2

    w1 = w1 - alpha * dw1  # Layer 1
    b1 = b1 - alpha * db1

# PREDICTIONS




# Generating Test Case with x representing 10 in binary .
X = np.array([10]) 
X = (((X[:, None] & (1 << np.arange(n)))) > 0).astype(int)
X[0] = X[0][::-1]
X = X.T

# Generating Ground Truth Set where Y represents 11 in binary.

Y = np.array([11])
Y = (((Y[:, None] & (1 << np.arange(n)))) > 0).astype(int)
Y[0] = Y[0][::-1]
Y = Y.T

# Foraward Propogation (Predicting...)
z1 = np.dot(w1, X) + b1  # Hidden Layer 1
a1 = sigmoid(z1)

z2 = np.dot(w2, a1) + b2  # Hidden Layer 2
a2 = sigmoid(z2)

z3 = np.dot(w3, a2) + b3
a3 = sigmoid(z3)  # Ouptut Layer

print a3

a3[a3 >= 0.5] = 1  # any number >= 0.5 is 1 else 0
a3[a3 < 0.5] = 0

print 'Predicted number : ',
for i in a3.T[0]:
    print int(i),
print
print 'Expected number  : ',
for i in Y.T[0]:
    print i,

    # Both expected and predicted output comes to be 11 :)
    # Hence our Neural Network is trained to predict next number with good enough accuracy.
    # We can calculate accuracy by testing this on multiple test cases and averaging the error obtained.
