import numpy as np
import matplotlib.pyplot as plt

# This function is NOT generalized as the previous methods defined here
# and only works for the specific problem in question 2
def q2_implicit_euler(l,x_start,x_end,y0,h):

    # Calculate the size of x and y
    L = int((x_end - x_start)/h + 0.5)

    # Allocate memory
    x = np.zeros(L)
    y = np.zeros(L)

    # Initialize
    x[0] = x_start
    y[0] = y0

    # Implicit Euler iteration
    for i in range(L - 1):
        x[i+1] = x[i] + h
        y[i+1] = y[i]/(1-h*l)

    return x,y

if __name__ == "__main__":

    ## Numeric Solution
    x_start, x_end = 0,50
    h = 0.01
    l = -0.5
    y0 = 1

    x,y = q2_implicit_euler(l,x_start,x_end,y0,h)

    h2 = 5
    x2,y2 = q2_implicit_euler(l,x_start,x_end,y0,h2)

    ## Analytic Solution
    F = lambda x,l,C: C*np.exp(l*x)
    C = 1
    y_analytical = F(x,l,C)
    
    plt.figure(figsize = (8,6))
    plt.plot(x2,y2,"b-", label = "Implicit Euler $h = 5$")
    plt.plot(x,y,"r-", label = "Implicit Euler $h = 0.5$")
    plt.plot(x,y_analytical,"k-", label = "Analytical Solution")
    plt.grid()
    plt.xlabel("x")
    plt.ylabel("y(x)")
    plt.legend()
    plt.savefig("implicit_euler_question21.pdf")
    plt.show()

