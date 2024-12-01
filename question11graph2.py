"""
This script solves a given Initial Value Problem for various initial values
"""

import numpy as np
import matplotlib.pyplot as plt

# Custom functions
from question11graph1 import explicit_euler

if __name__ == "__main__":

    f = lambda x,y: 2*x*y - x*y**2 # Derivative of the function

    h = 0.01 # Discrete step in x

    # Range of initial values
    y0_start, y0_end = 0,2
    num_y0_samples = 6
    Y0 = np.linspace(y0_start,y0_end,num_y0_samples)

    # Allocate memory
    x_start, x_end = 0,5
    L = int((x_end - x_start)/h + 0.5)

    X = np.zeros((num_y0_samples,L))
    Y = np.zeros((num_y0_samples,L))

    # Generate curves
    for i in range(num_y0_samples):
        X[i,:], Y[i,:] = explicit_euler(f,x_start,x_end,Y0[i],h)

        
    # Plot curves
    plt.figure(figsize = (8,6))

    for i in range(num_y0_samples):
        plt.plot(X[i,:],Y[i,:], label = f"Explicit Euler $y_0 = {Y0[i]:.4}$")

    plt.grid()
    plt.xlabel("x")
    plt.ylabel("y(x)")
    plt.legend()
    plt.savefig("explicit_euler_vary_initial.pdf")
    plt.show()