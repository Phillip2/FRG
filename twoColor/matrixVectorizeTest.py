import numpy as np

L = 5
N = 5
Nd = 5
dx = 1
dy = 1
x = np.linspace(0, L, N)
y = np.linspace(0, L, Nd)
u = np.linspace(0, L, N)
for i in range(Nd - 1):
    u = np.append(u, np.linspace((i+2)*L, (i+3)*L, N))

print(x, y, u)

uxForw = (-3.0/2.0*u[0] + 2.0*u[1] - 1.0/2.0*u[2])/dx
uxBack = (3.0/2.0*u[N-1] - 2.0*u[N-2] + 1.0/2.0*u[N-3])/dx
uxCent = (-1.0/2.0*u[0:N-2] + 1.0/2.0*u[2:N])/dx
ux = np.append(np.append(uxForw, uxCent), uxBack)
for i in range(N, N*Nd, N):
    uxForw = (-3.0/2.0*u[0 + i] + 2.0*u[1 + i] - 1.0/2.0*u[2 + i])/dx
    uxBack = (3.0/2.0*u[i + N-1] - 2.0*u[i + N-2] + 1.0/2.0*u[i + N-3])/dx
    uxCent = (-1.0/2.0*u[i+0:N+i-2] + 1.0/2.0*u[i+2:N+i])/dx
    ux = np.append(ux, np.append(np.append(uxForw, uxCent), uxBack))
uxxForw = (2.0*u[0] - 5.0*u[1] + 4.0*u[2] - 1.0*u[3])/dx**2
uxxBack = (2.0*u[N-1] - 5.0*u[N-2] + 4.0*u[N-3] - 1.0*u[N-4])/dx**2
uxxCent = (u[0:N-2] - 2*u[1:N-1] + u[2:N])/dx**2
uxx = np.append(np.append(uxxForw, uxxCent), uxxBack)
for i in range(N, N*Nd, N):
    uxxForw = (2.0*u[i] - 5.0*u[i+1] + 4.0*u[i+2] - 1.0*u[i+3])/dx**2
    uxxBack = (2.0*u[i+N-1] - 5.0*u[i+N-2] + 4.0*u[i+N-3]
                - 1.0*u[i+N-4])/dx**2
    uxxCent = (u[i+0:i+N-2] - 2*u[i+1:i+N-1] + u[i+2:i+N])/dx**2
    uxx = np.append(uxx, np.append(np.append(uxxForw, uxxCent), uxxBack))

print(len(ux), ux)
print(len(uxx), uxx)

uyForw = (-3.0/2.0*u[0:N] + 2.0*u[N:2*N] - 1.0/2.0*u[2*N:3*N])/dy
uyBack = (3.0/2.0*u[(Nd-1)*N:Nd*N] - 2.0*u[(Nd-2)*N:(Nd-1)*N]
          + 1.0/2.0*u[(Nd-3)*N:(Nd-2)*N])/dy
uyCent = (-1.0/2.0*u[0:(Nd-2)*N] + 1.0/2.0*u[2*N:Nd*N])/dy
uy = np.append(np.append(uyForw, uyCent), uyBack)

print(len(uy), uy)

uyyForw = (2.0*u[0:N] - 5.0*u[N:2*N] + 4.0*u[2*N:3*N] - 1.0*u[3*N:4*N])/dy**2
uyyBack = (2.0*u[(Nd-1)*N:Nd*N] - 5.0*u[(Nd-2)*N:(Nd-1)*N]
           + 4.0*u[(Nd-3)*N:(Nd-2)*N] - 1.0*u[(Nd-4)*N:(Nd-3)*N])/dy**2
uyyCent = (u[0:(Nd-2)*N] - 2*u[N:(Nd-1)*N] + u[2*N:Nd*N])/dy**2
uyy = np.append(np.append(uyyForw, uyyCent), uyyBack)

print(len(uyy), uyy)
