"""
This script applies the Explicit Euler method to an equation and calculates
its error in terms of the step h
"""

import numpy as np
import matplotlib.pyplot as plt

# Custom functions
from question11graph1 import explicit_euler
if __name__ == "__main__":

    ## Analytical solution
    F = lambda x, y0: 2*y0/(y0 + (2-y0)*np.exp(-x**2))

    ## Numerical solution, varying h

    f = lambda x,y: 2*x*y - x*y**2 # Derivative of the function
    y0 = 1 # Initial value

    # Various step lengths
    num_h_samples = 50
    H = np.logspace(-6,-1,num_h_samples)

    # Allocate space for the error vector
    e = np.zeros(num_h_samples)

    x_start, x_end = 0,5

    for i in range(num_h_samples):
        x, y = explicit_euler(f,x_start,x_end,y0,H[i])
        y_analytical = F(x,y0)
        ei = np.max(np.abs(y - y_analytical))
        e[i] = ei


    log_h = np.log10(H)
    log_e = np.log10(e)

    plt.figure(figsize = (8,6))
    plt.plot(log_h, log_e, "r-")
    plt.grid()
    plt.xlabel("$\log h$")
    plt.ylabel("$\log e_h$")
    plt.legend()
    plt.savefig("error_h.pdf")
    plt.show()

