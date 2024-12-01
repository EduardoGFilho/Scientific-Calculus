""""
This script implements the solution of an Initial Value Problem using the Explicit Euler
Method using the libraries numpy and matplotlib.
"""

import numpy as np
import matplotlib.pyplot as plt


def q3_explicit_euler(x_start,x_end,y0,h):
    # Calculate the size of x and y
    L = int((x_end - x_start)/h + 0.5)

    # Allocate memory
    x = np.zeros(L)
    y = np.zeros((2,L))

    # Initialize
    x[0] = x_start
    y[:,0] = y0

    # Explicit Euler iteration
    for i in range(L - 1):
        x[i+1] = x[i] + h
        y[:,i+1] = y[:,i] + h*np.array([y[1,i], - np.sin(y[0,i])])

    return x,y

if __name__ == "__main__":

    h = 0.0001
    t_start, t_end = 0, 4*np.pi

    L = int((t_end - t_start)/h + 0.5)

    y0 = np.array([0,1])

    x = np.zeros((L))
    y = np.zeros((2,L))

    x, y = q3_explicit_euler(t_start,t_end,y0,h)

    # Plot curves
    plt.figure(figsize = (6,6))

    y0 = np.round(y0*1.0,2)
    plt.plot(y[0],y[1],"k-", label = f"Explicit Euler $y_0 = ({y0[0]:.2},{y0[1]:.2})$")


    plt.grid()
    plt.xlabel("q(t)")
    plt.ylabel("p(t)")
    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')
    plt.legend(loc='upper right')
    plt.savefig("pendulum.pdf")
    plt.show()