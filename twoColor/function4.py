from math import pi
import time
from mpmath import *
from numpy import frompyfunc
mp.dps = 10; mp.pretty = True

# start = time.time()


def mat_sum(k, mu, T, ux, uxx, uy, uyy, uxy, x, y):
    a0 = (k**2*(-16.0*mu**2 + 4.0*ux + 8.0*uxx*x + 8.0*uy + 8.0*uyy*y)
          + 3.0*k**4.0 - mu**2*(16.0*ux + 32.0*uxx*x + 16.0*uy + 16.0*uyy*y)
          + 16.0*mu**4.0 + 8.0*ux*(uy + uyy*y) + 16.0*uxx*uy*x
          + 16.0*uxx*uyy*x*y - 16.0*uxy**2*x*y + 4.0*uy**2 + 8.0*uy*uyy*y)
    a1 = 6.0*k**2 + 4.0*ux + 8.0*uxx*x + 8.0*uy + 8.0*uyy*y
    a2 = 3.0
    b0 = ((k**2 - 4.0*mu**2 + 2.0*uy)
          * (k**2*(-4.0*mu**2 + 2.0*ux + 4.0*uxx*x + 2.0*uy + 4.0*uyy*y)
             + k**4.0 - mu**2*(8.0*ux + 16.0*uxx*x) + 4.0*ux*(uy + 2.0*uyy*y)
             + 8.0*x*(uxx*(uy + 2.0*uyy*y) - 2.0*uxy**2*y)))
    b1 = (k**2*(4.0*ux + 8.0*uxx*x + 8.0*uy + 8.0*uyy*y) + 3.0*k**4.0
          + mu**2*(16.0*ux + 32.0*uxx*x - 16.0*uy - 16.0*uyy*y) + 16.0*mu**4.0
          + 8.0*ux*(uy + uyy*y) + 16.0*uxx*uy*x + 16.0*uxx*uyy*x*y
          - 16.0*uxy**2*x*y + 4.0*uy**2 + 8.0*uy*uyy*y)
    b2 = 3.0*k**2 + 8.0*mu**2 + 2.0*ux + 4.0*uxx*x + 4.0*uy + 4.0*uyy*y
    w = 2*pi*T

    # print(a0, a1, a2, b0, b1, b2)
    """
    start = time.time()
    print("###################")
    print( nsum(lambda i: (a2*(w*i)**4 + a1*(w*i)**2 + a0)
           /((w*i)**6 + b2*(w*i)**4 + b1*(w*i)**2 + b0), [-inf, inf]))
    print("time: ", time.time() - start)
    start = time.time()
    print( nsum(lambda i: (a2*(w*i)**4 + a1*(w*i)**2 + a0)
           /((w*i)**6 + b2*(w*i)**4 + b1*(w*i)**2 + b0), [-inf, inf], method='shanks'))
    print("time: ", time.time() - start)
    start = time.time()
    print( nsum(lambda i: (a2*(w*i)**4 + a1*(w*i)**2 + a0)
           /((w*i)**6 + b2*(w*i)**4 + b1*(w*i)**2 + b0), [-inf, inf], method='levin'))
    print("time: ", time.time() - start)
    start = time.time()
    print( nsum(lambda i: (a2*(w*i)**4 + a1*(w*i)**2 + a0)
           /((w*i)**6 + b2*(w*i)**4 + b1*(w*i)**2 + b0), [-inf, inf], method='euler-maclaurin'))
    print("time: ", time.time() - start)
    start = time.time()
    print( nsum(lambda i: (a2*(w*i)**4 + a1*(w*i)**2 + a0)
           /((w*i)**6 + b2*(w*i)**4 + b1*(w*i)**2 + b0), [-inf, inf], method='direct'))
    print("time: ", time.time() - start)
    start = time.time()"""

    return nsum(lambda i: (a2*(w*i)**4 + a1*(w*i)**2 + a0)
           /((w*i)**6 + b2*(w*i)**4 + b1*(w*i)**2 + b0), [-inf, inf], method='direct')

def func(k, mu, T, ux, uxx, uy, uyy, uxy, x, y):
    function = frompyfunc(mat_sum, 10, 1)
    return function(k, mu, T, ux, uxx, uy, uyy, uxy, x, y)

# print(mat_sum(200, 10, 10, 100, 100, 100, 100, 100, 100, 100))
# print(time.time() - start)
