#!python

import numpy as np
from math import floor
import matplotlib.pyplot as plt
from scipy.optimize import fmin
from scipy.interpolate import interp2d, griddata, RectBivariateSpline
from scipy.interpolate import Rbf
from scipy.optimize import minimize, brute
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import MaxNLocator


n_T = input("Enter N_T: ")
n_mu = input("Enter N_mu: ")

sol = np.load('2DPhaseDiagramN_T'+str(n_T)+'N_mu'+str(n_mu)+'.npy')
p = np.load('2DPhaseDiagramN_T'+str(n_T)+'N_mu'+str(n_mu)+'params.npy')
lam, m_lam, g, k_cutoff, k_IR, N_k, L, N, dx, T_min, T_max, N_T, mu_max, N_mu,\
    Nd = p[0], p[1], p[2], p[3], p[4], int(p[5]), p[6], int(p[7]), p[8], p[9], p[10],\
    int(p[11]), p[12], int(p[13]), int(p[14])

x = np.linspace(0, L, N)
y = np.linspace(0, L, Nd)
T_array = np.linspace(T_min, T_max, N_T)
mu_array = np.linspace(0, mu_max, N_mu)
mu_ax, T_ax = np.meshgrid(mu_array, T_array)
xx = x
for i in range(int(Nd) - 1):
    xx = np.append(xx, x)
h = 595000
expl_sym_br = h*np.sqrt(xx)
min_values = np.empty([N_T, N_mu])
min_values_diq = np.empty([N_T, N_mu])
m_sig = np.zeros([len(T_array), len(mu_array)])
m_pi = np.zeros([len(T_array), len(mu_array)])
XX, YY = np.meshgrid(x, y)
dy = L / (Nd)

def interp_func(arg, f):
    return f(arg[0], arg[1])[0]


def func(params):
    a, b = params[0], params[1]
    if a < 0 or b < 0: return f(a, b) + 1e+10 # Ensure bound by cost function
    return f(a, b)


for t in range(len(T_array)):
    for m in range(len(mu_array)):
        s = sol[t][m] - expl_sym_br
        grid2D = np.array([s[0:N]])
        for i in range(Nd-1):
            grid2D = np.concatenate((grid2D, np.array([s[N*(i+1):(i+2)*N]])), axis=0)
        f = RectBivariateSpline(y, x, grid2D)
        # local minimum:
        # minimum = minimize(func, np.array([5*dy,5*dx]), args=(f), method='Nelder-Mead', options={'xtol': 1e-6, 'disp': True})
        # print("minimum: ", minimum.x)
        # print("f at min: ", f(minimum.x[0], minimum.x[1]))
        # print("chiral condensate interpolated: ", np.sqrt(minimum.x[1]))
        # global minimum
        rranges = (slice(0, Nd*dy, 0.5*dy), slice(0, N*dx, 0.5*dx))
        resbrute = brute(func, rranges, finish=fmin)
        # minimum = np.unravel_index(np.argmin(grid2D, axis=None), grid2D.shape)
        # print(minimum)
        print("(T, mu):", str((T_array[t], mu_array[m])))
        #print("Chiral condensate:"+str(np.sqrt(resbrute[1])))
        #print("Diquark condensate:"+str(np.sqrt(resbrute[0])))
        min_values[t, m] = np.sqrt(resbrute[1])
        min_values_diq[t, m] = np.sqrt(resbrute[0])

print(min_values[0,0])
dat_name = '2DPhaseDiagramN_T'+str(N_T)+'N_mu'+str(N_mu)
fig_name = dat_name+'.png'
min_name = dat_name+'minima'
np.save(min_name, (min_values, min_values_diq))
####
fig, ax1 = plt.subplots(nrows=1)
levels = MaxNLocator(nbins=32).tick_values(min_values.min(),
                                            min_values.max())
CS = ax1.contourf(mu_ax, T_ax, min_values, levels=levels)
fig.colorbar(CS, ax=ax1)
plt.title('Phase Diagram')
plt.savefig(fig_name)
####
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
