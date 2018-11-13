#!python

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fmin
import tkinter as tk
from tkinter import filedialog
from matplotlib.ticker import MaxNLocator



def interpFunc(t, a, b, c):
    return a*t**2 + b*t + c


def interpolate(s, expl_sym_br, dx):
    # Quadratische Interpolation vielleicht zu ungenau...
    x1 = np.argmin([s - expl_sym_br])
    x2 = x1 + 1
    x3 = x1 - 1
    y1 = s[x1] - expl_sym_br[x1]
    y2 = s[x2] - expl_sym_br[x2]
    y3 = s[x3] - expl_sym_br[x3]
    a = (x1*(y3 - y2) + x2*(y1 - y3) + x3*(y2 - y1))/((x1 - x2)*(x1 - x3) *
                                                      (x2 - x3))
    b = (y2 - y1)/(x2 - x1) - a*(x1 + x2)
    c = y1 - a*x1**2 - b*x1
    minimum = np.sqrt(fmin(interpFunc, x1, args=(a, b, c), xtol=0.001,
                           ftol=1e+6)[0]*dx)
    return minimum, b, 2*a


if __name__ == "__main__":
    # tk.Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
    filename = filedialog.askopenfilename()
    dat_name = filename
    dat = np.load(dat_name)[0:-1]
    sol = np.load(dat_name)[-1]

    lam = dat[0]
    m_lam = dat[1]
    g = dat[2]
    k_cutoff = dat[3]
    k_IR = dat[4]
    N_k = dat[5]
    L = dat[6]
    N = int(dat[7])
    x = np.linspace(0, L, N)
    dx = x[1] - x[0]
    k = np.linspace(k_cutoff, k_IR, N_k)
    u0 = 1.0/2.0*m_lam**2*x + lam/4.0*x**2
    T_min = dat[8]
    T_max = dat[9]
    N_T = int(dat[10])
    mu_min = dat[11]
    mu_max = dat[12]
    N_mu = int(dat[13])
    T_array = np.linspace(T_min, T_max, N_T)
    mu_array = np.linspace(mu_min, mu_max, N_mu)
    h = dat[14]
    expl_sym_br = h*np.sqrt(x)
    min_values = np.zeros([N_T, N_mu])
    m_sig = np.zeros([N_T, N_mu])
    m_pi = np.zeros([N_T, N_mu])

    for t in range(N_T):
        for m in range(N_mu):
            s = sol[t][m]
            argmin = np.argmin([s - expl_sym_br])
            if argmin != 0 and argmin != N - 1:
                min_values[t, m] = interpolate(s, expl_sym_br, dx)[0]
                m_pi[t, m] = np.sqrt(h/min_values[t, m])
                m_sig[t, m] = np.sqrt(4*min_values[t, m]**2 *
                                      interpolate(s, expl_sym_br, dx)[2]/dx**2
                                      + m_pi[t, m]**2)
            else:
                min_values[t, m] = np.sqrt(np.argmin([s
                                                      - expl_sym_br])*dx)

    print("chiral condensate: "+str(min_values[0, 0]))
    print("vacuum pion mass: "+str(m_pi[0, 0]))
    print("vacuum sigma mass: "+str(m_sig[0, 0]))
    plt.plot(sol[0][0] - expl_sym_br, color="r")
    plt.plot(sol[0][-1] - expl_sym_br, color="b")
    mu_ax, T_ax = np.meshgrid(mu_array, T_array)
    fig, ax1 = plt.subplots(nrows=1)
    levels = MaxNLocator(nbins=20).tick_values(min_values.min(),
                                              min_values.max())
    CS = ax1.contourf(mu_ax, T_ax, min_values, levels=levels)
    fig.colorbar(CS, ax=ax1)
    plt.title('Phase Diagram')
    plt.savefig('PhaseDiagramN_T'+str(N_T)+'N_mu'+str(N_mu)+'.png')
    plt.show()
