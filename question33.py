""""
This script implements the solution of an Initial Value Problem using the Explicit Euler
Method using the libraries numpy and matplotlib.
"""

import numpy as np
import matplotlib.pyplot as plt

from question31 import q3_explicit_euler

if __name__ == "__main__":

    h = 0.0001
    t_start, t_end = 0, 4*np.pi

    L = int((t_end - t_start)/h + 0.5)

    num_y0_samples = 6
    Q0 = np.zeros(num_y0_samples)
    P0 = np.linspace(0,3,num_y0_samples)
    Y0 = np.array([Q0,P0])

    X = np.zeros((num_y0_samples,1,L))
    Y = np.zeros((num_y0_samples,2,L))

    for i in range(num_y0_samples):
        X[i,:], Y[i,:] = q3_explicit_euler(t_start,t_end,Y0[:,i],h)

    # Plot curves
    plt.figure(figsize = (6,6))

    for i in range(num_y0_samples):
        y0 = np.round(Y0[:,i],2)
        plt.plot(Y[i,0],Y[i,1], label = f"Explicits Euler $y_0 = ({y0[0]:.2},{y0[1]:.2})$")


    plt.grid()
    plt.xlabel("q(t)")
    plt.ylabel("p(t)")
    ax = plt.gca()
    #ax.set_aspect('equal', adjustable='box')
    plt.legend()
    plt.savefig("pendulum_many.pdf")
    plt.show()