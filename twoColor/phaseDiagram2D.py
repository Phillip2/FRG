#!python

# http://modelling3e4.connectmv.com/wiki/Software_tutorial/Integration_of_ODEs#E

from math import pi, floor
import numpy as np
from scipy.integrate import ode
from function5 import func
import time
from joblib import Parallel, delayed
from scipy.optimize import fmin
import multiprocessing

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import MaxNLocator


def Epi(k, ux):
    return np.sqrt(k**2 + 2*ux)


def Esig(k, ux, uxx, xx):
    return np.sqrt(k**2 + 2*ux + 4*xx*uxx)


def Eq(k, g, xx):
    return np.sqrt(k**2 + g**2*xx)


def Ek(k, g, xx, yy, mu, n):
    return np.sqrt(g**2*yy + (np.sqrt(k**2 + g**2*xx) + (-1)**n*mu)**2)


def f(t, y, N, Nd, g, mu, T, gr, dx, xx, yy):
    u = y
    k = t
    uxForw = (-3.0/2.0*u[0] + 2.0*u[1] - 1.0/2.0*u[2])/dx
    uxBack = (3.0/2.0*u[N-1] - 2.0*u[N-2] + 1.0/2.0*u[N-3])/dx
    uxCent = (-1.0/2.0*u[0:N-2] + 1.0/2.0*u[2:N])/dx
    ux = np.append(np.append(uxForw, uxCent), uxBack)
    for i in range(N, N*Nd, N):
        uxForw = (-3.0/2.0*u[i] + 2.0*u[1+i] - 1.0/2.0*u[2+i])/dx
        uxBack = (3.0/2.0*u[i+N-1] - 2.0*u[i+N-2] + 1.0/2.0*u[i+N-3])/dx
        uxCent = (-1.0/2.0*u[i:N+i-2] + 1.0/2.0*u[i+2:N+i])/dx
        ux = np.append(ux, np.append(np.append(uxForw, uxCent), uxBack))
    uxxForw = (-3.0/2.0*ux[0] + 2.0*ux[1] - 1.0/2.0*ux[2])/dx
    uxxBack = (3.0/2.0*ux[N-1] - 2.0*ux[N-2] + 1.0/2.0*ux[N-3])/dx
    uxxCent = (-1.0/2.0*ux[0:N-2] + 1.0/2.0*ux[2:N])/dx
    uxx = np.append(np.append(uxForw, uxCent), uxBack)
    for i in range(N, N*Nd, N):
        uxxForw = (-3.0/2.0*ux[i] + 2.0*ux[1+i] - 1.0/2.0*ux[2+i])/dx
        uxxBack = (3.0/2.0*ux[i+N-1] - 2.0*ux[i+N-2] + 1.0/2.0*ux[i+N-3])/dx
        uxxCent = (-1.0/2.0*ux[i:N+i-2] + 1.0/2.0*ux[i+2:N+i])/dx
        uxx = np.append(uxx, np.append(np.append(uxxForw, uxxCent), uxxBack))

    uyForw = (-3.0/2.0*u[0:N] + 2.0*u[N:2*N] - 1.0/2.0*u[2*N:3*N])/dy
    uyBack = (3.0/2.0*u[(Nd-1)*N:Nd*N] - 2.0*u[(Nd-2)*N:(Nd-1)*N]
              + 1.0/2.0*u[(Nd-3)*N:(Nd-2)*N])/dy
    uyCent = (-1.0/2.0*u[0:(Nd-2)*N] + 1.0/2.0*u[2*N:Nd*N])/dy
    uy = np.append(np.append(uyForw, uyCent), uyBack)

    uyyForw = (-3.0/2.0*uy[0:N] + 2.0*uy[N:2*N] - 1.0/2.0*uy[2*N:3*N])/dy
    uyyBack = (3.0/2.0*uy[(Nd-1)*N:Nd*N] - 2.0*uy[(Nd-2)*N:(Nd-1)*N]
               + 1.0/2.0*uy[(Nd-3)*N:(Nd-2)*N])/dy
    uyyCent = (-1.0/2.0*uy[0:(Nd-2)*N] + 1.0/2.0*uy[2*N:Nd*N])/dy
    uyy = np.append(np.append(uyyForw, uyyCent), uyyBack)

    # Check: define second derivative in terms of first derivative
    # should approx. coincide with uyyCent
    # uyyCent2 = (-1.0/2.0*uy[0:(Nd-2)*N] + 1.0/2.0*uy[2*N:Nd*N])/dy

    uxyForw = (-3.0/2.0*ux[0:N] + 2.0*ux[N:2*N] - 1.0/2.0*ux[2*N:3*N])/dy
    uxyBack = ((3.0/2.0*ux[(Nd-1)*N:Nd*N] - 2.0*ux[(Nd-2)*N:(Nd-1)*N]
               + 1.0/2.0*ux[(Nd-3)*N:(Nd-2)*N])/dy)
    uxyCent = (-1.0/2.0*ux[0:(Nd-2)*N] + 1.0/2.0*ux[2*N:Nd*N])/dy
    uxy = np.append(np.append(uxyForw, uxyCent), uxyBack)

    duk = (k**4/(12*pi**2)*(3.0/Epi(k, ux)*(1.0/np.tanh(Epi(k, ux)/(2*T)))
                            - 8/Ek(k, g, xx, yy, mu, 1)*(1 - mu/(Eq(k, g, xx)))
                            * (np.tanh(Ek(k, g, xx, yy, mu, 1)/(2*T)))
                            - 8/Ek(k, g, xx, yy, mu, 0)*(1 + mu/(Eq(k, g, xx)))
                            * np.tanh(Ek(k, g, xx, yy, mu, 0)/(2*T))
           + func(k, mu, T, ux, uxx, uy, uyy, uxy, xx, yy)))

    return duk


def ode_solve(N, Nd, g, m, t, gr, dx, xx, yy, T_ax, mu_ax):
    ode15s.set_initial_value(u0, k_cutoff).set_f_params(N, Nd, g, mu_ax[t][m],
                                                        T_ax[t][m], gr, dx, xx,
                                                        yy)
    c = 1
    while c < N_k:
        ode15s.integrate(ode15s.t-dk)
        # print(ode15s.t)
        c += 1
    return (ode15s.y, ode15s.t, ode15s.successful())


def solve(N, Nd, g, m, t, gr, dx, xx, yy, T_ax, mu_ax):
    res = ode_solve(N, Nd, g, m, t, gr, dx, xx, yy, T_ax, mu_ax)
    print(res[1], res[2])
    print((mu_ax[t][m], T_ax[t][m]))
    return res[0]


def interpFunc(t, a, b, c):
    return a*t**2 + b*t + c


def interpolate(s, expl_sym_br, dx):
    x1 = np.argmin([s - expl_sym_br])
    x2 = x1 + 1
    x3 = x1 - 1
    print(x1, x2, x3)
    y1 = s[x1] - expl_sym_br[x1]
    y2 = s[x2] - expl_sym_br[x2]
    y3 = s[x3] - expl_sym_br[x3]
    print(y1, y2, y3)
    a = (x1*(y3 - y2) + x2*(y1 - y3) + x3*(y2 - y1))/((x1 - x2)*(x1 - x3) *
                                                      (x2 - x3))
    b = (y2 - y1)/(x2 - x1) - a*(x1 + x2)
    c = y1 - a*x1**2 - b*x1
    min_pos = fmin(interpFunc, x1, args=(a, b, c), xtol=0.001, ftol=1e+6)[0]
    minimize = np.sqrt(min_pos % N*dx)
    print(min_pos, minimize)
    return minimize, b, 2*a


def interpolateDiq(s, expl_sym_br, dy, N):
    argm = np.argmin([s - expl_sym_br])
    x1 = floor(argm/N)
    x2 = x1 + 1
    x3 = x1 - 1
    print("Diquark Minimum Interpolation")
    print(x1, x2, x3)
    y1 = s[argm] - expl_sym_br[argm]
    y2 = s[argm + N] - expl_sym_br[argm + N]
    y3 = s[argm - N] - expl_sym_br[argm - N]
    print(y1, y2, y3)
    print("expl: "+str(expl_sym_br[argm])+"..."+str(expl_sym_br[argm + N]))
    a = (x1*(y3 - y2) + x2*(y1 - y3) + x3*(y2 - y1))/((x1 - x2)*(x1 - x3) *
                                                      (x2 - x3))
    b = (y2 - y1)/(x2 - x1) - a*(x1 + x2)
    c = y1 - a*x1**2 - b*x1
    min_pos = fmin(interpFunc, x1, args=(a, b, c), xtol=0.001, ftol=1e+6)[0]
    print(min_pos, floor(argm/N))
    minimize = np.sqrt(min_pos*dy)
    print(min_pos, minimize)
    return minimize, b, 2*a


if __name__ == "__main__":
    start = time.time()
    lam = 23
    m_lam = 0
    g = 4.8
    k_cutoff = 600
    k_IR = 80
    dk = 1
    N_k = (k_cutoff - k_IR)/(dk) + 1
    L = 140.0**2
    N = 40
    Nd = 40
    dx = L / (N - 1)
    dy = L / (Nd - 1)
    x = np.linspace(0, L, N, dtype=np.float64)
    y = np.linspace(0, L, Nd, dtype=np.float64)
    gr = np.linspace(0, L, N, dtype=np.float64)
    xx = x
    yy = np.linspace(0, 0, N, dtype=np.float64)
    for i in range(Nd - 1):
        # create rho vector for functions that only depend on rho
        xx = np.append(xx, x)
    for i in range(Nd - 1):
        # create Delta vector for functions that only depend on Delta
        yy = np.append(yy, np.linspace(y[i + 1], y[i + 1], N))
    for i in range(Nd - 1):
        gr = np.append(gr, np.linspace(y[i + 1], L + y[i + 1], N))
    # k = np.linspace(k_cutoff, k_IR, N_k)
    u0 = 1.0/2.0*m_lam**2*gr + lam/4.0*gr**2

    T_min = 3
    T_max = 250
    N_T = 40
    mu_min = 0
    mu_max = 150
    N_mu = 40
    T_array = np.linspace(T_min, T_max, N_T)
    mu_array = np.linspace(mu_min, mu_max, N_mu)
    mu_ax, T_ax = np.meshgrid(mu_array, T_array)
    h = 595000
    expl_sym_br = h*np.sqrt(xx)
    sol = [[None for _ in range(len(mu_array))] for _ in range(len(T_array))]
    min_values = np.empty([len(T_array), len(mu_array)])
    min_values_diq = np.empty([len(T_array), len(mu_array)])
    m_sig = np.zeros([len(T_array), len(mu_array)])
    m_pi = np.zeros([len(T_array), len(mu_array)])

    ode_order = 5
    ode_steps = 100000
    ode_integrator = 'vode'
    ode_method = 'bdf'
    r_tol = 1e-13
    a_tol = 1e-13
    ode15s = ode(f)
    ode15s.set_integrator(ode_integrator, method=ode_method, order=ode_order,
                          nsteps=ode_steps, rtol=r_tol, atol=a_tol)

    # ode_integrator = 'dop853'  # dopri5
    # ode15s.set_integrator(ode_integrator, nsteps=ode_steps)

    num_cores = multiprocessing.cpu_count()

    for t in range(len(T_array)):
        sol[t] = Parallel(n_jobs=num_cores)(delayed(solve)(
            N, Nd, g, m, t, gr, dx, xx, yy, T_ax, mu_ax) for m in
            range(len(mu_array)))
    for t in range(len(T_array)):
        for m in range(len(mu_array)):
            s = sol[t][m]
            argm = np.argmin([s - expl_sym_br])
            if argm % N != 0 and argm % N != N - 1:
                minimum = interpolate(s, expl_sym_br, dx)[0]
                min_values[t, m] = minimum
                m_pi[t, m] = np.sqrt(h/min_values[t, m])
                m_sig[t, m] = np.sqrt(4*min_values[t, m]**2 *
                                      interpolate(s, expl_sym_br, dx)[2]/dx**2
                                      + m_pi[t, m]**2)
            else:
                min_values[t, m] = np.sqrt(np.argmin([s - expl_sym_br]) % N*dx)
            if floor(argm/N) != 0 and floor(argm/N) != Nd - 1:
                minimum = interpolateDiq(s, expl_sym_br, dy, N)[0]
                min_values_diq[t, m] = minimum
            else:
                min_values_diq[t, m] = np.sqrt(floor(argm/N)*dy)

    print("chiral condensate: "+str(min_values[0, 0]))
    print("diquark condensate: "+str(min_values_diq[0, -1]))
    print("vacuum pion mass: "+str(m_pi[0, 0]))
    print("vacuum sigma mass: "+str(m_sig[0, 0]))
    print("min values: ", min_values)
    end = time.time()
    print("TIME:", end - start)
    plt.plot(sol[0][0])
    plt.plot(sol[0][-1])
    param_list = np.array([lam, m_lam, g, k_cutoff, k_IR, N_k, L, N, dx,
                           T_min, T_max, N_T, mu_max, N_mu, Nd, mu_min])
    doc = open("optimize.txt", "a")
    doc.write(str((end - start, (ode_integrator, ode_method, ode_order,
                                 ode_steps, lam, k_cutoff, k_IR, dk, N_k, L, N,
                                 Nd, N_T, N_mu, min_values[0, 0])))+"\n")
    dat_name = '2DPhaseDiagramN_T'+str(N_T)+'N_mu'+str(N_mu)
    params_name = dat_name+'params'
    fig_name = dat_name+'.png'
    np.save(dat_name, sol)
    np.save(params_name, param_list)
    fig, ax1 = plt.subplots(nrows=1)
    levels = MaxNLocator(nbins=32).tick_values(min_values.min(),
                                               min_values.max())
    CS = ax1.contourf(mu_ax, T_ax, min_values, levels=levels)
    fig.colorbar(CS, ax=ax1)
    plt.title('Phase Diagram')
    plt.savefig(fig_name)
    fig, ax1 = plt.subplots(nrows=1)
    levels = MaxNLocator(nbins=32).tick_values(min_values_diq.min(),
                                               min_values_diq.max())
    CS = ax1.contourf(mu_ax, T_ax, min_values_diq, levels=levels)
    fig.colorbar(CS, ax=ax1)
    plt.title('Diquark Phase Diagram')
    plt.savefig('Diq'+fig_name)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_wireframe(mu_ax, T_ax, min_values, color='red')
    ax.plot_wireframe(mu_ax, T_ax, min_values_diq)
    plt.savefig('3D'+fig_name)
    plt.show()
