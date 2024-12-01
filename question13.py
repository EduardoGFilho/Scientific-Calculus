import numpy as np
import matplotlib.pyplot as plt

from question11graph1 import explicit_euler

def mid_point(f,x_start,x_end,y0,h):

    # Calculate the size of x and y
    L = int((x_end - x_start)/h + 0.5)

    # Allocate memory
    x = np.zeros(L)
    y = np.zeros(L)

    # Initialize
    x[0] = x_start
    y[0] = y0

    # Midpoint iteration
    for i in range(L - 1):
        x[i+1] = x[i] + h
        y[i+1] = y[i] + h*f(x[i] + h/2, y[i]+ (h/2)*f(x[i],y[i]))

    return x,y


def runge_kutta4(f,x_start,x_end,y0,h):
    # Calculate the size of x and y
    L = int((x_end - x_start)/h + 0.5)

    # Allocate memory
    x = np.zeros(L)
    y = np.zeros(L)

    # Initialize
    x[0] = x_start
    y[0] = y0

    # Order 4 Runge-Kutta iteration
    for i in range(L - 1):

        k1 = f(x[i],y[i])
        k2 = f(x[i] + h/2, y[i] + h*k1/2)
        k3 = f(x[i] + h/2, y[i] + h*k2/2)
        k4 = f(x[i]+ h, y[i] + h*k3)

        x[i+1] = x[i] + h
        y[i+1] = y[i] + (h/6)*(k1 + 2*k2 + 2* k3 + k4)

    return x,y

if __name__ == "__main__":
    f = lambda x,y: 2*x*y - x*y**2 # Derivative of the function
    h = 0.01 # Discrete step in x
    y0 = 1 # Initial value

    x_start, x_end = 0,5

    x_euler, y_euler = explicit_euler(f, x_start, x_end, y0, h)
    x_mp, y_mp = mid_point(f, x_start, x_end, y0, h)
    x_rk4, y_rk4 = runge_kutta4(f, x_start, x_end, y0, h)

    ## Analytical solution
    F = lambda x, y0: 2*y0/(y0 + (2-y0)*np.exp(-x**2))
    x_analytical, y_analytical = x_rk4, F(x_rk4, y0)

    plt.figure(figsize =(8,6))
    #plt.plot(x1,y1,"b-", label = "Explicit Euler $h = 0.1$")
    plt.plot(x_analytical,y_analytical,"k-", label = "Analytical Solution")
    plt.plot(x_euler, y_euler,"r-", label = "Explicit Euler")
    plt.plot(x_mp,y_mp, label = "Midpoint")
    plt.plot(x_rk4,y_rk4, label = "4th Order Runge-Kutta")
    plt.grid()
    plt.xlabel("x")
    plt.ylabel("y(x)")
    plt.legend() 
    plt.savefig("various_methods.pdf")

    #plt.show()

    # Question to professor: How can we calculate the order of these methods, if adjacent results are not related as for
    # iterative methods?

    # Solution estimated from the definition of convergence given in wikipedia

    h2 = h*2
    x2_euler, y2_euler = explicit_euler(f, x_start, x_end, y0, h2)
    x2_mp, y2_mp = mid_point(f, x_start, x_end, y0, h2)
    x2_rk4, y2_rk4 = runge_kutta4(f, x_start, x_end, y0, h2)

    ## Analytical solution
    x2_analytical, y2_analytical = x2_rk4, F(x2_rk4, y0)

    e1_euler = np.max(np.abs(y_euler - y_analytical))
    e1_mp = np.max(np.abs(y_mp - y_analytical))
    e1_rk4 = np.max(np.abs(y_rk4 - y_analytical))

    e2_euler = np.max(np.abs(y2_euler - y2_analytical))
    e2_mp = np.max(np.abs(y2_mp - y2_analytical))
    e2_rk4 = np.max(np.abs(y2_rk4 - y2_analytical))

    order_euler = (np.log(e2_euler) - np.log(e1_euler))/(np.log(h2) - np.log(h))
    order_mp = (np.log(e2_mp) - np.log(e1_mp))/(np.log(h2) - np.log(h))
    order_rk4 = (np.log(e2_rk4) - np.log(e1_rk4))/(np.log(h2) - np.log(h))

    print(f"Estimated order of Euler Method:{order_euler}")
    print(f"Estimated order of RK4 Method:{order_rk4}")
    print(f"Estimated order of Midpoint Method:{order_mp}")



    

