from math import pi

def func2(k, mu, T, ux, uxx, uy, uyy, uxy, x, y):
    nMats = 26
    a0 = k**2*(-16.0*mu**2 + 4.0*ux + 8.0*uxx*x + 8.0*uy + 8.0*uyy*y) + 3.0*k**4.0 - mu**2*(16.0*ux + 32.0*uxx*x + 16.0*uy + 16.0*uyy*y) + 16.0*mu**4.0 + 8.0*ux*(uy + uyy*y) + 16.0*uxx*uy*x + 16.0*uxx*uyy*x*y - 16.0*uxy**2*x*y + 4.0*uy**2 + 8.0*uy*uyy*y
    a1 = 6.0*k**2 + 4.0*ux + 8.0*uxx*x + 8.0*uy + 8.0*uyy*y
    a2 = 3.0
    b0 = (k**2 - 4.0*mu**2 + 2.0*uy)*(k**2*(-4.0*mu**2 + 2.0*ux + 4.0*uxx*x + 2.0*uy + 4.0*uyy*y) + k**4.0 - mu**2*(8.0*ux + 16.0*uxx*x) + 4.0*ux*(uy + 2.0*uyy*y) + 8.0*x*(uxx*(uy + 2.0*uyy*y) - 2.0*uxy**2*y))
    b1 = k**2*(4.0*ux + 8.0*uxx*x + 8.0*uy + 8.0*uyy*y) + 3.0*k**4.0 + mu**2*(16.0*ux + 32.0*uxx*x - 16.0*uy - 16.0*uyy*y) + 16.0*mu**4.0 + 8.0*ux*(uy + uyy*y) + 16.0*uxx*uy*x + 16.0*uxx*uyy*x*y - 16.0*uxy**2*x*y + 4.0*uy**2 + 8.0*uy*uyy*y
    b2 = 3.0*k**2 + 8.0*mu**2 + 2.0*ux + 4.0*uxx*x + 4.0*uy + 4.0*uyy*y
    w = 2*pi*T
    res = 0
    # Matsubara sum from -n to n symmetric for n -> -n. Calculate from 1 to n
    # then times two and finally add the zero mode n = 0.
    for i in range(1, nMats):
        res = res + (a2*(w*i)**4 + a1*(w*i)**2 + a0)/((w*i)**6 + b2*(w*i)**4 + b1*(w*i)**2 + b0)
    return 2*res + a0/b0

