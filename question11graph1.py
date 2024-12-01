""""
This script implements the solution of an Initial Value Problem using the Explicit Euler
Method using the libraries numpy and matplotlib.
"""

import numpy as np
import matplotlib.pyplot as plt

# Function implementation of Explicit Euler
def explicit_euler(f,x_start,x_end,y0,h):
    # Calculate the size of x and y
    L = int((x_end - x_start)/h + 0.5)

    # Allocate memory
    x = np.zeros(L)
    y = np.zeros(L)

    # Initialize
    x[0] = x_start
    y[0] = y0

    # Explicit Euler iteration
    for i in range(L - 1):
        x[i+1] = x[i] + h
        y[i+1] = y[i] + h*f(x[i],y[i])

    return x,y

if __name__ == "__main__":

    y0 = 1 # Initial value
    f = lambda x,y: 2*x*y - x*y**2 # Derivative of the function

    ## Solution with h = 0.1
    h1 = 0.1 # Discrete step in x

    # Calculate the size of x and y
    L1 = int(5/h1 + 0.5)

    # Allocate memory
    x1 = np.zeros(L1)
    y1 = np.zeros(L1)

    # Initialize
    x1[0] = 0
    y1[0] = y0

    # Explicit Euler iteration
    for i in range(L1 - 1):
        x1[i+1] = x1[i] + h1
        y1[i+1] = y1[i] + h1*f(x1[i],y1[i])
   
    ## Analytical solution
    F = lambda x, y0: 2*y0/(y0 + (2-y0)*np.exp(-x**2))

    ## Solution with h = 0.01
    h2 = 0.01
    x2, y2 = explicit_euler(f,0,5,y0,h2)

    y_analytical = F(x2,y0)

    plt.figure(figsize = (8,6))
    plt.plot(x1,y1,"b-", label = "Explicit Euler $h = 0.1$")
    plt.plot(x2,y2,"r-", label = "Explicit Euler $h = 0.01$")
    plt.plot(x2,y_analytical,"k-", label = "Analytical Solution")
    plt.grid()
    plt.xlabel("x")
    plt.ylabel("y(x)")
    plt.legend()
    plt.savefig("explicit_euler.pdf")
    plt.show()