#!python

import numpy as np
from scipy.integrate import quad
import time
from joblib import Parallel, delayed
import multiprocessing

import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import MaxNLocator


def integrand(E, M, mu, cutoff):
    C_list = [1, -3, 3, -1]
    fPV = 0
    for e in C_list: fPV += e*np.sqrt(E**2 + C_list.index(e)*cutoff**2)
    return 1/(np.pi**2)*E*np.sqrt(E**2 - M**2)*(fPV + np.heaviside(mu - E, 0)*(mu - E))


def f(M, mu, cutoff, G):
    inf = np.inf
    integral = quad(integrand, M, inf, args=(M, mu, cutoff), epsabs=0.1, epsrel=0.001)
    result = -6*integral[0] + M**2/(4*G)
    return result


if __name__ == "__main__":
    start = time.time()
    k_cutoff = 757.048
    G = 6.002/k_cutoff**2

    N_mu = 40
    mu_min = 0
    mu_max = 400
    mu_array = np.linspace(mu_min, mu_max, N_mu)

    N_cond = 165
    cond_max = 330
    cond_array = np.linspace(0, cond_max, N_cond)
    dcond = cond_max/N_cond
    min_values_cond = np.zeros(N_mu)
    min_values_cond[0] = 1
    sol = [[None for _ in range(N_cond)] for _ in range(N_mu)]
    num_cores = 2
    print("number of cores: ", num_cores)
    for i in range(N_mu):
        sol[i] = Parallel(n_jobs=num_cores)(delayed(f)(cond_array[k], mu_array[i], k_cutoff, G)\
                                                            for k in range(N_cond))
        s = np.array(sol[i])
        min_values_cond[i] = np.unravel_index(np.argmin(s, axis=None), s.shape)[0]*dcond
        print("mu: "+str(mu_array[i]))
        print("min: "+str(np.unravel_index(np.argmin(s, axis=None), s.shape)))
    print("TIME: ", time.time()-start)

    print(min_values_cond)
    param_list = np.array([G, k_cutoff, mu_min, mu_max, N_mu, N_cond, cond_max, sol], dtype=object)
    dat_name = 'PhaseDiagramHomZeroTN_mu'+str(N_mu)+'N_cond'+str(N_cond)
    fig_name = dat_name+'.png'
    np.save(dat_name, param_list)
    plt.plot(mu_array, min_values_cond)
    plt.show()
