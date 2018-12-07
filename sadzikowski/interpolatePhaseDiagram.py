#!python

import numpy as np
from scipy.integrate import dblquad
import time
from scipy.interpolate import interp2d, griddata, RectBivariateSpline
from scipy.optimize import minimize, brute, basinhopping
from scipy.interpolate import RegularGridInterpolator as rgi
from scipy.optimize import fmin
from joblib import Parallel, delayed
import multiprocessing

import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import MaxNLocator
import tkinter as tk
from tkinter import filedialog

if __name__ == "__main__":
    filename = filedialog.askopenfilename()
    filename1 = filedialog.askopenfilename()
    filename2 = filedialog.askopenfilename()
    dat_name = filename
    dat1 = np.load(dat_name)[0:-1]
    sol1 = np.load(dat_name)[-1]
    G, Gd, k_cutoff, T_min1, T_max1, N_T1, mu_min, mu_max, N_mu, N_cond, N_d, N_q = dat1

    dat_name = filename1
    dat2 = np.load(dat_name)[0:-1]
    sol2 = np.load(dat_name)[-1]
    G, Gd, k_cutoff, T_min2, T_max2, N_T2, mu_min, mu_max, N_mu, N_cond, N_d, N_q = dat2

    dat_name = filename2
    dat3 = np.load(dat_name)[0:-1]
    sol3 = np.load(dat_name)[-1]
    G, Gd, k_cutoff, T_min3, T_max3, N_T3, mu_min, mu_max, N_mu, N_cond, N_d, N_q = dat3

    print(G, Gd, k_cutoff, T_min1, T_max1, N_T1, mu_min, mu_max, N_mu, N_cond, N_d, N_q)
    print(T_min1, T_max1, N_T1)
    print(T_min2, T_max2, N_T2)
    print(T_min3, T_max3, N_T3)
    start = time.time()
    T_array1 = np.linspace(T_min1, T_max1, N_T1)
    T_array2 = np.linspace(T_min2, T_max2, N_T2)
    T_array3 = np.linspace(T_min3, T_max3, N_T3)
    T_array = np.append(np.append(T_array1, T_array2), T_array3)
    mu_array = np.linspace(mu_min, mu_max, N_mu)

    sol = np.append(np.append(sol1, sol2, axis=0), sol3, axis=0)

    cond_max = 500
    d_max = 200
    q_max = 400
    cond_array = np.linspace(0, cond_max, N_cond)
    d_array = np.linspace(0, d_max, N_d)
    q_array = np.linspace(0, q_max, N_q)

    N_T = N_T1 + N_T2 + N_T3
    mu_ax, T_ax = np.meshgrid(mu_array, T_array)
    min_values_cond = np.zeros([N_T, N_mu])
    min_values_diq = np.zeros([N_T, N_mu])
    min_values_inh = np.zeros([N_T, N_mu])

    def func(params):
        # print(params)
        a, b, c = params[0], params[1], params[2]
        if a < 0 or b < 0 or c < 0 or a > cond_max or b > d_max or c > q_max: return abs(a*b*c)*1e+10 # Ensure bound by cost function
        return f((a, b, c))

    for i in range(N_T):
        for j in range(N_mu):
            s = sol[i][j]
            f = rgi((cond_array, d_array, q_array), s)
            # print(f((cond_array[11], 0, 0)), f((0, 50, 40)), f((200, 50, 40)))
            rranges = (slice(0, cond_max, 20), slice(0, d_max, 10), slice(0, q_max, 10))
            resbrute = brute(func, rranges, finish=fmin)
            #basin_min = basinhopping(func, np.array([cond_max/2, d_max/2, q_max/2]), niter=200, stepsize=10)
            #print(basin_min.x, basin_min.fun)
            #input(basin_min)
            print(resbrute[0], resbrute[1], resbrute[2])
            #input("continue...")
            min_values_cond[i, j] = resbrute[0]
            min_values_diq[i, j] = resbrute[1]
            if resbrute[0] > 1: min_values_inh[i, j] = resbrute[2]
            else: min_values_inh[i, j] = 0
            print("mu, T: "+str((mu_array[j], T_array[i])))

    print(min_values_cond)
    print(min_values_diq)
    print(min_values_inh)
    print("TIME: ", time.time()-start)

    # TODO DETECT PHASE BOUNDARIES

    dat_name = '2DPhaseDiagramN_T'+str(N_T)+'N_mu'+str(N_mu)+'N_cond'+str(N_cond)+'N_d'+str(N_d)+'N_q'+str(N_q)
    fig_name = dat_name+'.png'
    min_list = np.array([min_values_cond, min_values_diq, min_values_inh], dtype=object)
    np.save(dat_name+'minima', min_list)

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
