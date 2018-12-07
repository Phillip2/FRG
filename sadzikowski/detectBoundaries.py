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

    min_values_cond, min_values_diq, min_values_inh = np.load(filename)[0], \
        np.load(filename)[1], np.load(filename)[2]

    print(min_values_cond)
    #print(min_values_diq[0,-1])
    #print(min_values_inh[0,0])

    """plt.scatter(1, 1, c="b")
    plt.scatter(2, 2, c="r")
    plt.show()"""

    N_T = 30
    N_mu = 30
    cond_max = 500
    d_max = 200
    q_max = 400
    N_cond = 20
    N_d = 20
    N_q = 12

    T_min1, T_min2, T_min3 = 5, 65, 125
    T_max1, T_max2, T_max3 = 60, 120, 180

    mu_min, mu_max = 0, 400

    T_array1 = np.linspace(T_min1, T_max1, N_T/3)
    T_array2 = np.linspace(T_min2, T_max2, N_T/3)
    T_array3 = np.linspace(T_min3, T_max3, N_T/3)
    T_array = np.append(np.append(T_array1, T_array2), T_array3)
    mu_array = np.linspace(mu_min, mu_max, N_mu)
    mu_ax, T_ax = np.meshgrid(mu_array, T_array)

    for i in range(N_T):
        for j in range(N_mu - 1):
            if min_values_cond[i, j+1] - min_values_cond[i, j] > 2*cond_max/N_cond\
                    or (min_values_cond[j+1, i] > 1 and min_values_cond[j, i] < 1):
                plt.scatter(mu_array[j+1], T_array[i], c="black")
            if min_values_cond[i, j] - min_values_cond[i, j+1] > 2*cond_max/N_cond\
                    or (min_values_cond[j+1, i] > 1 and min_values_cond[j, i] < 1):
                plt.scatter(mu_array[j], T_array[i], c="black")
            if (min_values_inh[i, j] < 1 and min_values_inh[i, j+1] > 1):
                plt.scatter(mu_array[j+1], T_array[i], c="r")
            if (min_values_inh[i, j] > 1 and min_values_inh[i, j+1] < 1):
                plt.scatter(mu_array[j], T_array[i], c="r")
            if (min_values_diq[i, j] < 1 and min_values_diq[i, j+1] > 1):
                plt.scatter(mu_array[j+1], T_array[i], c="b")
            if (min_values_diq[i, j] > 1 and min_values_diq[i, j+1] < 1):
                plt.scatter(mu_array[j], T_array[i], c="b")

    plt.show()

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
