from math import pi
import numpy as np
import time

# start = time.time()


def func(k, mu, T, ux, uxx, uy, uyy, uxy, x, y):
    # np.seterr(all='raise')
    nMats = 50
    # print(nMats)
    a0 = (k**2*(-16.0*mu**2 + 4.0*ux + 8.0*uxx*x + 8.0*uy + 8.0*uyy*y)
          + 3.0*k**4.0 - mu**2*(16.0*ux + 32.0*uxx*x + 16.0*uy + 16.0*uyy*y)
          + 16.0*mu**4.0 + 8.0*ux*(uy + uyy*y) + 16.0*uxx*uy*x
          + 16.0*uxx*uyy*x*y - 16.0*uxy**2*x*y + 4.0*uy**2 + 8.0*uy*uyy*y)
    a1 = 6.0*k**2 + 4.0*ux + 8.0*uxx*x + 8.0*uy + 8.0*uyy*y
    a2 = 3.0
    """b0 = (((k**2 - 4.0*mu**2 + 2.0*uy)*np.heaviside(abs(k**2 - 4.0*mu**2 + 2.0*uy), 0)
           + 0.0000000001*(1 - np.heaviside(abs(k**2 - 4.0*mu**2 + 2.0*uy), 0)))
          * (k**2*(-4.0*mu**2 + 2.0*ux + 4.0*uxx*x + 2.0*uy + 4.0*uyy*y)
             + k**4.0 - mu**2*(8.0*ux + 16.0*uxx*x) + 4.0*ux*(uy + 2.0*uyy*y)
             + 8.0*x*(uxx*(uy + 2.0*uyy*y) - 2.0*uxy**2*y)))"""
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
    # print("params f2: ", a0[0], a1[0], b0[0], b1[0], b2[0])
    # print("params f2: ", a0, a1, b0, b1, b2, "...end")
    # print(b0)
    res = 0.
    # Matsubara sum from -n to n symmetric for n -> -n. Calculate from 1 to n
    # then times two and finally add the zero mode n = 0.
    for i in range(1, nMats):
        res = (res + (a2*(w*i)**4 + a1*(w*i)**2 + a0)/((w*i)**6 + b2*(w*i)**4
                                                       + b1*(w*i)**2 + b0))
    # print((mu, T))
    # print("res: ", 2*res + a0/b0)
    """print("a: ", np.array2string(a0, separator=', '))
    print("b: ", np.array2string(b0, separator=', '))
    print((k**2 - 4.0*mu**2 + 2.0*uy)[15])
    print("a/b:")
    print(a0/b0)"""
    return 2*res + a0/b0

# print(time.time() - start)
