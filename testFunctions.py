import numpy as np
#import numba as nb
import math as mp

#spec = [
#(('x', nb.float64[:, :])          # an array field
#)
#    ]

#@nb.jit(parallel=True)
def himmelblau(x: np.array) -> np.float:
    """
    https://en.wikipedia.org/wiki/Himmelblau%27s_function
    f(3,2)=0
    f(-2.8051, 3.1313)=0
    f(-3.7793,-3.2831)=0
    f(3.5844, -1.8481)=0
    """
    __name__  = 'Himmelblau'
    f = (((x[0]**2) + x[1] - 11)**2) + ((x[0] + x[1]**2) - 7)**2
    assert(type(f) != float)
    return f

def ackley(x: np.array) -> np.float:
    """
    f(0,0)=0
    """
    __name__  = 'Ackley'
    f = -20*np.exp(-0.2*np.sqrt(0.5*(x[0]**2+x[1]**2))) - np.exp(0.5*(np.cos(2*np.pi*x[0]) + np.cos(2*np.pi*x[1])) ) + np.e + 20
    return f