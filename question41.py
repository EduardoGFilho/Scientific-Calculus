""""
This script implements the solution of an Initial Value Problem using the Explicit Euler
Method using the libraries numpy and matplotlib.
"""

import numpy as np
import matplotlib.pyplot as plt

# not generalzied
def q4_explicit_euler(A, x_start,x_end,y0,h):
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
        y[:,i+1] = y[:,i] + h*np.matmul(A,y[:,i])

    return x,y

if __name__ == "__main__":

    h = 0.01
    t_start, t_end = 0,6

    L = int((t_end - t_start)/h + 0.5)

    A = np.array([[-4, -1],[3, 0]])

    num_y0_samples = 6
    angles = np.linspace(0,2*np.pi,num_y0_samples,False)

    Y0 = np.array([np.cos(angles),np.sin(angles)])

    X = np.zeros((num_y0_samples,1,L))
    Y = np.zeros((num_y0_samples,2,L))

    for i in range(num_y0_samples):
        X[i,:], Y[i,:] = q4_explicit_euler(A,t_start,t_end,Y0[:,i],h)

    # Plot curves
    plt.figure(figsize = (6,6))

    for i in range(num_y0_samples):
        y0 = np.round(Y0[:,i],2)
        plt.plot(Y[i,0],Y[i,1], label = f"Explicit Euler $y_0 = ({y0[0]:.2},{y0[1]:.2})$")


    plt.grid()
    plt.xlabel("x")
    plt.ylabel("y(x)")
    plt.legend()
    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')
    plt.savefig("trajectory_vary_initial.pdf")
    plt.show()