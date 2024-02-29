import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors

def fullGrid(grid):
    mirrorX = np.flip(grid,axis=0)
    mirrorY = np.flip(grid,axis=1)
    flip = np.flip(grid)
    top = np.concatenate((flip,mirrorX),axis=1)
    bottom = np.concatenate((mirrorY,grid),axis=1)
    total = np.concatenate((top,bottom),axis=0)
    return total

mlow = np.load('divided_mlow.npy')
mveryhigh = np.load('divided_mveryhigh.npy')

fig, ax = plt.subplots(figsize=(8, 6))
ax.set_xlabel(r'$k = \sqrt{k_x^2 + k_y^2}$')
ax.set_ylabel(r'$k_z$')
ax.set_title('Pos with M_gal between 2e10 and 3e10 Msun, no SN, Npix=64')
ax.grid(visible=True)
cax = ax.imshow(fullGrid(mlow), origin='lower',extent=(-1,1,-1,1),cmap='nipy_spectral',vmin=0,vmax=5)
fig.colorbar(cax,label=r'$P_{RSD}(k)/P_{Pos}(k)$')
plt.show()

#First, lets look at the Kaiser effect, which dominates at small scales k

def compute_mu(k_trans,k_z):
    """
    Compute the value of mu = cos(theta) for each grid point, taking theta to be the angle with the k_z axis 
    """
    k_total = np.sqrt(k_z**2 + k_trans**2)
    if k_total < 1e-5:
        return 1
    else:
        mu = k_z / k_total
    return mu

kmax = 1
N = 64
k_idx = np.linspace(0,kmax-0.01,N)
k_idxFac = np.floor(N*k_idx/kmax).astype(int)
print(k_idxFac)

mlow_mu = np.zeros_like(mlow)
mveryhigh_mu = np.zeros_like(mveryhigh)
for i in range(N):
    kz_ = k_idxFac[i]
    for j in range(N):
        kt_ = k_idxFac[j]
        #print(kt_,kz_)
        mlow_mu[i,j] = compute_mu(kt_,kz_)
        mveryhigh_mu[i,j] = compute_mu(kt_,kz_)

mlow_mu = np.flip(mlow_mu,axis=0)
mveryhigh_mu = np.flip(mveryhigh_mu,axis=0)
#Now we have an array with the mu value at each point in the grid! 

plt.imshow(mlow_mu)
plt.title("mu values")
plt.colorbar()
plt.show()

#Lets look at the bottom left quadrant (k 0 to 0.25)
mlow_mu_lowk = mlow_mu[int(0.75*N):,:int(0.25*N)] #xdata
mlow_lowk = mlow[:int(0.25*N),:int(0.25*N)] #ydata

#Now get on with fitting: 
from scipy.optimize import curve_fit

def kaiser(mu,f):
    """
    This is the function we want to fit to the data to measure the Kaiser effect by determining f 
    Assume the ratio of P(k)s is given by this function: P_RSD/P = (1+f*mu**2)**2
    """
    return (1+f*(mu**2))**2

print(mlow_mu_lowk)
print(mlow_lowk)


fig, ax = plt.subplots(figsize=(8, 6))
ax.set_xlabel(r'$k = \sqrt{k_x^2 + k_y^2}$')
ax.set_ylabel(r'$k_z$')
ax.set_title('Divided RSD of low mass galaxies at low k')
ax.grid(visible=True)
cax = ax.imshow(mlow_lowk, origin='lower',extent=(0,0.25,0,0.25),cmap='nipy_spectral',vmin=0,vmax=5)
fig.colorbar(cax,label=r'$P_{RSD}(k)/P_{Pos}(k)$')
plt.show()

f_low,f_lowStd = curve_fit(kaiser, mlow_mu_lowk.flatten(), mlow_lowk.flatten())
#print(f_low[0],u"\u00B1",f_lowStd[0,0])
print(u'The fitted value f to quantify the Kaiser effect for high mass galaxies is: {:.3f} \u00B1 {:.5f}'.format(f_low[0],f_lowStd[0,0]))

#Lets look at the bottom left quadrant (k 0 to 0.25)
mveryhigh_mu_lowk = mveryhigh_mu[int(0.75*N):,:int(0.25*N)] #xdata
mveryhigh_lowk = mveryhigh[:int(0.25*N),:int(0.25*N)] #ydata

f_veryhigh,f_veryhighStd = curve_fit(kaiser, mveryhigh_mu_lowk.flatten(), mveryhigh_lowk.flatten())
print(u'The fitted value f to quantify the Kaiser effect for high mass galaxies is: {:.3f} \u00B1 {:.5f}'.format(f_veryhigh[0],f_veryhighStd[0,0]))
#print(f_veryhigh[0],u"\u00B1",f_veryhighStd[0,0])

#Done without outlier-analysis: maybe for the future? 




