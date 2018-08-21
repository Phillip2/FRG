#!python

import numpy as np

import matplotlib.pyplot as plt

data = np.load('TripoltPhaseDiagramN_T2N_mu2.npy')
p = np.load('TripoltPhaseDiagramN_T2N_mu2params.npy')
lam, m_lam, g, k_cutoff, k_IR, N_k, L, N, dx, T_max, N_T, mu_max, N_mu =\
    p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], p[9], p[10], p[11],\
    p[12]

x = np.linspace(0, L, N)
k = np.linspace(k_cutoff, k_IR, N_k)
u0 = 1.0/2.0*m_lam**2*x + lam/4.0*x**2
T_array = np.linspace(5, T_max, N_T)
mu_array = np.linspace(0, mu_max, N_mu)
expl_sym_br = 1750000*np.sqrt(x)
min_values = np.empty([len(mu_array), len(T_array)])

for m in range(len(mu_array)):
    for t in range(len(T_array)):
        min_values[m, t] = np.sqrt(np.argmin([data[m][t][-1, :] - expl_sym_br])
                                   * dx)

mu_ax, T_ax = np.meshgrid(mu_array, T_array)
fig = plt.figure()
CS = plt.contourf(mu_ax, T_ax, min_values.T, 15)
plt.title('Phase Diagram')
plt.show()
