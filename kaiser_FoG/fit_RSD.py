import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
from scipy.optimize import curve_fit

def fullGrid(grid):
    mirrorX = np.flip(grid,axis=0)
    mirrorY = np.flip(grid,axis=1)
    flip = np.flip(grid)
    top = np.concatenate((flip,mirrorX),axis=1)
    bottom = np.concatenate((mirrorY,grid),axis=1)
    total = np.concatenate((top,bottom),axis=0)
    return total

def load_files(file):
    path = '/net/draco/data2/vanveenhuyzen/rsd_project/nbodykit_sourcecode/griddata/'
    m_pos = np.load(path+'grid_pos_{}.npy'.format(file))
    m_rsd = np.load(path+'grid_rsd_{}.npy'.format(file))
    return m_pos,m_rsd,(m_rsd/m_pos)

#Load in the data: 
mall16_pos,mall16_rsd,mall16_div = load_files('mall_N16')

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

def compute_ktotal(k_trans,k_z):
    k_total = np.sqrt(k_z**2 + k_trans**2)
    if k_total < 1e-5:
        k_total = 1e-5
    return k_total

kmax = 1 
print(mall16_div.shape)

def mu_grid(grid):
    
    N = np.shape(grid)[0]

    k_index = np.linspace(0,kmax-0.01,N)
    k_indexFac = np.floor(N*k_index/kmax).astype(int)

    mu_grid = np.zeros_like(grid)
    ktotal_grid = np.zeros_like(grid)
    for i in range(N):
        kz = k_indexFac[i] #radial k 

        kz_alt = k_index[i]

        for j in range(N):
            kt = k_indexFac[j] #transversal k

            kt_alt = k_index[j]

            mu_grid[i,j] = compute_mu(kt,kz)
            ktotal_grid[i,j] = compute_ktotal(kt_alt,kz_alt)

    mu_grid = np.flip(mu_grid,axis=0)
    ktotal_grid = np.flip(ktotal_grid,axis=0)
    return mu_grid,ktotal_grid

def kaiser(mu,f):
    """
    This is the function we want to fit to the data to measure the Kaiser effect by determining f 
    Assume the ratio of P(k)s is given by this function: P_RSD/P = (1+f*mu**2)**2
    """
    return (1+f*(mu**2))**2

def compute_growthfactor(divided_grid,kmax_kaiser,description):

    #Get the pixel resolution of the grid 
    N = np.shape(divided_grid)[0]

    #Get the limits we want to index to 
    M = int(kmax_kaiser*N)
    L = int((1-kmax_kaiser)*N)

    #Pk is easy, but for mu we need to calculate the mu values first
    pk = divided_grid[:M,:M]

    mu_,ktotal_ = mu_grid(divided_grid)
    mu = mu_[L:,:M]

    f,f_std = curve_fit(kaiser, mu.flatten(), pk.flatten())
    print(u'The fitted value f to quantify the Kaiser effect for {} galaxies is: {:.5f} \u00B1 {:.5f}'.format(description,f[0],f_std[0,0]))
    return f[0],f_std[0,0], mu, ktotal_

f,ferr,mu16,ktotal16 = compute_growthfactor(mall16_div,0.25,'test')

mall16_div_lowk = mall16_div[:4,:4]
print(mall16_div_lowk.shape)

mall16_div_highk = mall16_div[:,int(0.85*16):]
print(mall16_div_highk)

mall_highk = np.mean(mall16_div_highk,axis=1)
print(mall_highk)

k_values = np.linspace(0,1,16)
for i in range(mall16_div_highk.shape[1]):
    plt.plot(k_values,mall16_div_highk[:,i],linestyle='dashed',alpha=0.8)
plt.plot(k_values,mall_highk,color='black')
plt.show()

fig, ax = plt.subplots(figsize=(8, 6))
ax.set_xlabel(r'$k = \sqrt{k_x^2 + k_y^2}$')
ax.set_ylabel(r'$k_z$')
ax.set_title(r'$P_{RSD}/P_{real}$ for second highest Log mass bin, no SN, Npix=64')
ax.grid(visible=True)
cax = ax.imshow(fullGrid(mall16_div_lowk), origin='lower',extent=(-0.25,0.25,-0.25,0.25),cmap='nipy_spectral',vmin=0,vmax=5)
fig.colorbar(cax,label=r'$P_{RSD}(k)/P_{Pos}(k)$')
#plt.savefig('mhighLog.png')
plt.show()
plt.close()

fig, ax = plt.subplots(figsize=(8, 6))
ax.set_xlabel(r'$k = \sqrt{k_x^2 + k_y^2}$')
ax.set_ylabel(r'$k_z$')
ax.set_title(r'$P_{RSD}/P_{real}$ for second highest Log mass bin, no SN, Npix=64')
ax.grid(visible=True)
cax = ax.imshow(fullGrid(mall16_div), origin='lower',extent=(-1,1,-1,1),cmap='nipy_spectral',vmin=0,vmax=5)
fig.colorbar(cax,label=r'$P_{RSD}(k)/P_{Pos}(k)$')
#plt.savefig('mhighLog.png')
plt.show()
plt.close()

fig, ax = plt.subplots(figsize=(8, 6))
ax.set_xlabel(r'$k = \sqrt{k_x^2 + k_y^2}$')
ax.set_ylabel(r'$k_z$')
ax.set_title(r'$P_{RSD}/P_{real}$ for second highest Log mass bin, no SN, Npix=64')
ax.grid(visible=True)
cax = ax.imshow(mall16_div_highk, origin='lower',extent=(0.75,1,0,1),cmap='nipy_spectral',vmin=0.5,vmax=1.05)
fig.colorbar(cax,label=r'$P_{RSD}(k)/P_{Pos}(k)$')
#plt.savefig('mhighLog.png')
plt.show()
plt.close()

mu_mall,ktotal_mall = mu_grid(mall16_div)

print(ktotal_mall.flatten())

def rsd_fit(mu,f,a,b,c):
    """
    Product of the Kaiser and F.o.G. effects in the RSD plot, note that we divide mu by ktotal_mall
    """
    return ((1+f*(mu**2))**2) * (-a*(mu/ktotal_mall.flatten())**2 + b*(mu/ktotal_mall.flatten()) + c)

params,pcov = curve_fit(rsd_fit, mu_mall.flatten(), mall16_div.flatten(),p0=[0.26,-0.1,0.5,1])
print(u'The fitted parameters to quantify the Kaiser effect for {} galaxies is: {} \u00B1 {}'.format('all',params,pcov[0]))