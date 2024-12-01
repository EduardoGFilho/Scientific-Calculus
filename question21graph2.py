import numpy as np
import matplotlib.pyplot as plt

from question11graph1 import explicit_euler

if __name__ == "__main__":
    f = lambda l,y: l*y
    y0 = 1

    ## Numeric Solution
    x_start, x_end = 0,50
    h = 0.01
    l = -0.5

    x,y = explicit_euler(lambda x,y : f(l,y),x_start,x_end,y0,h)

    h2 = 5
    x2,y2 = explicit_euler(lambda x,y : f(l,y),x_start,x_end,y0,h2)

    ## Analytic Solution
    F = lambda x,l,C: C*np.exp(l*x)
    C = 1
    y_analytical = F(x,l,C)
    
    plt.figure(figsize = (8,6))
    plt.plot(x2,y2,"b-", label = "Explicit Euler $h = 5$")
    plt.plot(x,y,"r-", label = "Explicit Euler $h = 0.5$")
    plt.plot(x,y_analytical,"k-", label = "Analytical Solution")
    plt.grid()
    plt.xlabel("x")
    plt.ylabel("y(x)")
    plt.legend()
    plt.savefig("explicit_euler_question21_w_divergence.pdf")
    plt.show()