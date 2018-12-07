#!python

import numpy as np
from scipy.integrate import dblquad
import time
from joblib import Parallel, delayed
import multiprocessing

import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import MaxNLocator


def E(k, th, q, M, m):
    return np.sqrt(k**2 + M**2 + q**2/4 +
                   (-1)**m*np.sqrt((q*k*np.cos(th))**2 + (M*q)**2))


def eps(k, th, q, M, d, mu, n, m):
    return np.sqrt((mu + (-1)**n*E(k, th, q, M, m))**2 + d**2)

# TODO: Remove E0 term from Taylor approximation and just add quadratic term in
# q
def integrand(k, th, q, M, d, mu, T):
    return (-k**2*np.sin(th)*(T*2*np.log(1 + np.exp(-eps(k, th, q, M, d, mu, 0, 0)/T)) + T*np.log(1 + np.exp(-eps(k, th, q, M, 0, mu, 0, 0)/T)) +
                              T*2*np.log(1 + np.exp(-eps(k, th, q, M, d, mu, 0, 1)/T)) + T*np.log(1 + np.exp(-eps(k, th, q, M, 0, mu, 0, 1)/T)) +
                              T*2*np.log(1 + np.exp(-eps(k, th, q, M, d, mu, 1, 0)/T)) + T*np.log(1 + np.exp(-eps(k, th, q, M, 0, mu, 1, 0)/T)) +
                              T*2*np.log(1 + np.exp(-eps(k, th, q, M, d, mu, 1, 1)/T)) + T*np.log(1 + np.exp(-eps(k, th, q, M, 0, mu, 1, 1)/T)) +
                              eps(k, th, q, M, d, mu, 0, 0) + 6*E(k, th, 0, M, 0) +
                              eps(k, th, q, M, d, mu, 0, 1) +
                              eps(k, th, q, M, d, mu, 1, 0) - 2*E(k, th, q, M, 0) +
                              eps(k, th, q, M, d, mu, 1, 1) - 2*E(k, th, q, M, 1))
            + k**2*np.sin(th)*((E(k, th, q, M, 0) - mu)*np.heaviside(mu - E(k, th, q, M, 0), 0) + (E(k, th, q, M, 1) - mu)*np.heaviside(mu - E(k, th, q, M, 1), 0)))


def integrandd(k, th, q, M, d, mu, T):
    return (-k**2*np.sin(th)*(T*2*np.log(1 + np.exp(-eps(k, th, q, M, d, mu, 0, 0)/T)) + T*np.log(1 + np.exp(-eps(k, th, q, M, 0, mu, 0, 0)/T)) +
                              T*2*np.log(1 + np.exp(-eps(k, th, q, M, d, mu, 0, 1)/T)) + T*np.log(1 + np.exp(-eps(k, th, q, M, 0, mu, 0, 1)/T)) +
                              T*2*np.log(1 + np.exp(-eps(k, th, q, M, d, mu, 1, 0)/T)) + T*np.log(1 + np.exp(-eps(k, th, q, M, 0, mu, 1, 0)/T)) +
                              T*2*np.log(1 + np.exp(-eps(k, th, q, M, d, mu, 1, 1)/T)) + T*np.log(1 + np.exp(-eps(k, th, q, M, 0, mu, 1, 1)/T)) +
                              eps(k, th, q, M, d, mu, 0, 0) +
                              eps(k, th, q, M, d, mu, 0, 1) +
                              eps(k, th, q, M, d, mu, 1, 0) + E(k, th, q, M, 0) +
                              eps(k, th, q, M, d, mu, 1, 1) + E(k, th, q, M, 1))
            + k**2*np.sin(th)*((E(k, th, q, M, 0) - mu)*np.heaviside(mu - E(k, th, q, M, 0), 0) + (E(k, th, q, M, 1) - mu)*np.heaviside(mu - E(k, th, q, M, 1), 0)))


# Try Buballas expression
def integrandd(k, th, q, M, d, mu, T):
    return (-k**2*np.sin(th)*(2*T*np.log(1 + np.exp(-eps(k, th, q, M, d, mu, 0, 0)/T)) + T*np.log(1 + np.exp(-eps(k, th, q, M, 0, mu, 0, 0)/T)) +
                              2*T*np.log(1 + np.exp(-eps(k, th, q, M, d, mu, 0, 1)/T)) + T*np.log(1 + np.exp(-eps(k, th, q, M, 0, mu, 0, 1)/T)) +
                              2*T*np.log(1 + np.exp(-eps(k, th, q, M, d, mu, 1, 0)/T)) + T*np.log(1 + np.exp(-eps(k, th, q, M, 0, mu, 1, 0)/T)) +
                              2*T*np.log(1 + np.exp(-eps(k, th, q, M, d, mu, 1, 1)/T)) + T*np.log(1 + np.exp(-eps(k, th, q, M, 0, mu, 1, 1)/T)) +
                              eps(k, th, q, M, d, mu, 0, 0) + eps(k, th, q, M, 0, mu, 0, 0)/2 +
                              eps(k, th, q, M, d, mu, 0, 1) + eps(k, th, q, M, 0, mu, 0, 1)/2 +
                              eps(k, th, q, M, d, mu, 1, 0) + eps(k, th, q, M, 0, mu, 1, 0)/2 +
                              eps(k, th, q, M, d, mu, 1, 1) + eps(k, th, q, M, 0, mu, 1, 1)/2))


def f(q, M, d, mu, T, cutoff, G, Gd):
    integral = dblquad(integrand, 0, np.pi, 0, cutoff, args=(q, M, d, mu, T), epsabs=0.1, epsrel=0.1)
    result = 1/((2*np.pi)**2)*2*integral[0] + M**2/(4*G) + d**2/(4*Gd) + M**2*(93)**2*q**2/(2*301**2)
    return result


if __name__ == "__main__":
    start = time.time()
    G = 5.01/(1000**2)
    Gd = 3*G/4
    k_cutoff = 650

    T_min = 125
    T_max = 180
    N_T = 10
    mu_min = 0
    mu_max = 400
    N_mu = 30
    N_cond = 20
    cond_array = np.linspace(0, 500, N_cond)
    N_d = 20
    N_q = 12
    d_array = np.linspace(0, 200, N_d)
    q_array = np.linspace(0, 400, N_q)
    T_array = np.linspace(T_min, T_max, N_T)
    mu_array = np.linspace(mu_min, mu_max, N_mu)
    mu_ax, T_ax = np.meshgrid(mu_array, T_array)
    min_values = np.zeros([N_T, N_mu])
    min_values_cond = np.zeros([N_T, N_mu])
    min_values_diq = np.zeros([N_T, N_mu])
    min_values_inh = np.zeros([N_T, N_mu])
    sol = [[[[[None for _ in range(N_q)] for _ in range(N_d)] for _ in range(N_cond)] for _ in range(N_mu)] for _ in range(N_T)]
    num_cores = multiprocessing.cpu_count()
    print("number of cores: ", num_cores)
    for i in range(N_T):
        for j in range(N_mu):
            for k in range(N_cond):
                for l in range(N_d):
                    sol[i][j][k][l] = Parallel(n_jobs=num_cores)(delayed(f)(q_array[m], cond_array[k], d_array[l], mu_array[j], T_array[i], k_cutoff, G, Gd)\
                                                                 for m in range(N_q))
            s = np.array(sol[i][j])
            #print(s)
            min_values_cond[i, j] = np.unravel_index(np.argmin(s, axis=None), s.shape)[0]
            min_values_diq[i, j] = np.unravel_index(np.argmin(s, axis=None), s.shape)[1]
            min_values_inh[i, j] = np.unravel_index(np.argmin(s, axis=None), s.shape)[2]
            print("mu, T: "+str((mu_array[j], T_array[i])))
            print("min: "+str(np.unravel_index(np.argmin(s, axis=None), s.shape)))

    print(min_values_cond)
    print(min_values_diq)
    print(min_values_inh)
    print("TIME: ", time.time()-start)

    param_list = np.array([G, Gd, k_cutoff, T_min, T_max, N_T, mu_min, mu_max, N_mu, N_cond, N_d, N_q, sol], dtype=object)
    dat_name = '2DPhaseDiagram_T_max'+str(T_max)+'N_T'+str(N_T)+'N_mu'+str(N_mu)+'N_cond'+str(N_cond)+'N_d'+str(N_d)+'N_q'+str(N_q)
    fig_name = dat_name+'.png'
    np.save(dat_name, param_list)

    fig, ax1 = plt.subplots(nrows=1)
    fig2, ax2 = plt.subplots(nrows=1)
    fig3, ax3 = plt.subplots(nrows=1)
    levels = MaxNLocator(nbins=16).tick_values(min_values_cond.min(),
                                               min_values_cond.max())
    CS = ax1.contourf(mu_ax, T_ax, min_values_cond, levels=levels)
    levels2 = MaxNLocator(nbins=16).tick_values(min_values_diq.min(),
                                               min_values_diq.max())
    CS2 = ax2.contourf(mu_ax, T_ax, min_values_diq, levels=levels2)
    levels3 = MaxNLocator(nbins=16).tick_values(min_values_inh.min(),
                                               min_values_inh.max())
    CS3 = ax3.contourf(mu_ax, T_ax, min_values_inh, levels=levels3)
    fig.colorbar(CS, ax=ax1)
    fig.colorbar(CS2, ax=ax2)
    fig.colorbar(CS3, ax=ax3)
    plt.show()
