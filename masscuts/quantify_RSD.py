import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors

#import sys
#sys.path.append('/net/draco/data2/vanveenhuyzen/rsd_project/nbodykit_sourcecode/griddata')

def fullGrid(grid):
    mirrorX = np.flip(grid,axis=0)
    mirrorY = np.flip(grid,axis=1)
    flip = np.flip(grid)
    top = np.concatenate((flip,mirrorX),axis=1)
    bottom = np.concatenate((mirrorY,grid),axis=1)
    total = np.concatenate((top,bottom),axis=0)
    return total

#####################
#Load in the data:
#####################

def load_files(file):
    path = '/net/draco/data2/vanveenhuyzen/rsd_project/nbodykit_sourcecode/griddata/'
    m_pos = np.load(path+'grid_pos_{}.npy'.format(file))
    m_rsd = np.load(path+'grid_rsd_{}.npy'.format(file))
    return m_pos,m_rsd,(m_rsd/m_pos)

mlowest_pos,mlowest_rsd,mlowest_div = load_files('mlowest')
mlow_pos,mlow_rsd,mlow_div = load_files('mlow')
mmid_pos,mmid_rsd,mmid_div = load_files('mmid')
mhigh_pos,mhigh_rsd,mhigh_div = load_files('mhigh')
mhighest_pos,mhighest_rsd,mhighest_div = load_files('mhighest')

print(mlowest_div)

fig, ax = plt.subplots(figsize=(8, 6))
ax.set_xlabel(r'$k = \sqrt{k_x^2 + k_y^2}$')
ax.set_ylabel(r'$k_z$')
ax.set_title(r'$P_{RSD}/P_{real}$ for m_lowest, no SN, Npix=64')
ax.grid(visible=True)
cax = ax.imshow(fullGrid(mlowest_div), origin='lower',extent=(-1,1,-1,1),cmap='nipy_spectral',vmin=0,vmax=5)
fig.colorbar(cax,label=r'$P_{RSD}(k)/P_{Pos}(k)$')
plt.savefig('mlowest.png')
#plt.show()
plt.close()

fig, ax = plt.subplots(figsize=(8, 6))
ax.set_xlabel(r'$k = \sqrt{k_x^2 + k_y^2}$')
ax.set_ylabel(r'$k_z$')
ax.set_title(r'$P_{RSD}/P_{real}$ for m_low, no SN, Npix=64')
ax.grid(visible=True)
cax = ax.imshow(fullGrid(mlow_div), origin='lower',extent=(-1,1,-1,1),cmap='nipy_spectral',vmin=0,vmax=5)
fig.colorbar(cax,label=r'$P_{RSD}(k)/P_{Pos}(k)$')
plt.savefig('mlow.png')
#plt.show()
plt.close()

fig, ax = plt.subplots(figsize=(8, 6))
ax.set_xlabel(r'$k = \sqrt{k_x^2 + k_y^2}$')
ax.set_ylabel(r'$k_z$')
ax.set_title(r'$P_{RSD}/P_{real}$ for m_mid, no SN, Npix=64')
ax.grid(visible=True)
cax = ax.imshow(fullGrid(mmid_div), origin='lower',extent=(-1,1,-1,1),cmap='nipy_spectral',vmin=0,vmax=5)
fig.colorbar(cax,label=r'$P_{RSD}(k)/P_{Pos}(k)$')
plt.savefig('mmid.png')
#plt.show()
plt.close()

fig, ax = plt.subplots(figsize=(8, 6))
ax.set_xlabel(r'$k = \sqrt{k_x^2 + k_y^2}$')
ax.set_ylabel(r'$k_z$')
ax.set_title(r'$P_{RSD}/P_{real}$ for m_high, no SN, Npix=64')
ax.grid(visible=True)
cax = ax.imshow(fullGrid(mhigh_div), origin='lower',extent=(-1,1,-1,1),cmap='nipy_spectral',vmin=0,vmax=5)
fig.colorbar(cax,label=r'$P_{RSD}(k)/P_{Pos}(k)$')
plt.savefig('mhigh.png')
#plt.show()
plt.close()

fig, ax = plt.subplots(figsize=(8, 6))
ax.set_xlabel(r'$k = \sqrt{k_x^2 + k_y^2}$')
ax.set_ylabel(r'$k_z$')
ax.set_title(r'$P_{RSD}/P_{real}$ for m_highest, no SN, Npix=64')
ax.grid(visible=True)
cax = ax.imshow(fullGrid(mhighest_div), origin='lower',extent=(-1,1,-1,1),cmap='nipy_spectral',vmin=0,vmax=5)
fig.colorbar(cax,label=r'$P_{RSD}(k)/P_{Pos}(k)$')
plt.savefig('mhighest.png')
#plt.show()
plt.close()


mlow = np.load('divided_mlow.npy')
mveryhigh = np.load('divided_mveryhigh.npy')

fig, ax = plt.subplots(figsize=(8, 6))
ax.set_xlabel(r'$k = \sqrt{k_x^2 + k_y^2}$')
ax.set_ylabel(r'$k_z$')
ax.set_title(r'$P_{RSD}/P_{real}$ for m_lowest, no SN, Npix=64')
ax.grid(visible=True)
cax = ax.imshow(fullGrid(mlow), origin='lower',extent=(-1,1,-1,1),cmap='nipy_spectral',vmin=0,vmax=5)
fig.colorbar(cax,label=r'$P_{RSD}(k)/P_{Pos}(k)$')
#plt.show()
plt.close()

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
#plt.show()
plt.close()

#Lets look at the bottom left quadrant (k 0 to 0.25), since the Kaiser effect occurs on large scales ---> small k 
mlow_mu_lowk = mlow_mu[int(0.75*N):,:int(0.25*N)] #xdata
mlow_lowk = mlow[:int(0.25*N),:int(0.25*N)] #ydata

fig, ax = plt.subplots(figsize=(12,3))
ax.set_xlabel(r'$k = \sqrt{k_x^2 + k_y^2}$')
ax.set_ylabel(r'$k_z$')
ax.set_title('Divided RSD of low mass galaxies at low k')
ax.grid(visible=True)
cax = ax.imshow(mlow_lowk, origin='lower',extent=(0.75,1,0,1),cmap='nipy_spectral',vmin=0,vmax=5)
fig.colorbar(cax,label=r'$P_{RSD}(k)/P_{Pos}(k)$')
fig.savefig('low_kabs.png')
plt.show()

#Now get on with fitting: 
from scipy.optimize import curve_fit

def kaiser(mu,f):
    """
    This is the function we want to fit to the data to measure the Kaiser effect by determining f 
    Assume the ratio of P(k)s is given by this function: P_RSD/P = (1+f*mu**2)**2
    """
    return (1+f*(mu**2))**2

#print(mlow_mu_lowk)
print(mlow_lowk)

f_low,f_lowStd = curve_fit(kaiser, mlow_mu_lowk.flatten(), mlow_lowk.flatten())
#print(f_low[0],u"\u00B1",f_lowStd[0,0])
print(u'The fitted value f to quantify the Kaiser effect for low mass galaxies is: {:.3f} \u00B1 {:.5f}'.format(f_low[0],f_lowStd[0,0]))

#Lets look at the bottom left quadrant (k 0 to 0.25)
mveryhigh_mu_lowk = mveryhigh_mu[int(0.75*N):,:int(0.25*N)] #xdata
mveryhigh_lowk = mveryhigh[:int(0.25*N),:int(0.25*N)] #ydata

#print(mveryhigh_lowk) #this is very close to 1 and not too noisy...

fig, ax = plt.subplots(figsize=(8, 6))
ax.set_xlabel(r'$k = \sqrt{k_x^2 + k_y^2}$')
ax.set_ylabel(r'$k_z$')
ax.set_title('Divided RSD of very high mass galaxies at all k')
ax.grid(visible=True)
cax = ax.imshow(mveryhigh, origin='lower',extent=(0,1,0,1),cmap='nipy_spectral',vmin=0,vmax=5)
fig.colorbar(cax,label=r'$P_{RSD}(k)/P_{Pos}(k)$')
plt.show()
plt.close()

fig, ax = plt.subplots(figsize=(8, 6))
ax.set_xlabel(r'$k = \sqrt{k_x^2 + k_y^2}$')
ax.set_ylabel(r'$k_z$')
ax.set_title('Divided RSD of very high mass galaxies at low k')
ax.grid(visible=True)
cax = ax.imshow(mveryhigh_lowk, origin='lower',extent=(0,0.25,0,0.25),cmap='nipy_spectral',vmin=0,vmax=5)
fig.colorbar(cax,label=r'$P_{RSD}(k)/P_{Pos}(k)$')
plt.show()
plt.close()

f_veryhigh,f_veryhighStd = curve_fit(kaiser, mveryhigh_mu_lowk.flatten(), mveryhigh_lowk.flatten())
print(u'The fitted value f to quantify the Kaiser effect for high mass galaxies is: {:.3f} \u00B1 {:.5f}'.format(f_veryhigh[0],f_veryhighStd[0,0]))
#print(f_veryhigh[0],u"\u00B1",f_veryhighStd[0,0])

def compute_growthfactor(divided_grid,kmax_kaiser):
    N = np.shape(divided_grid)[0]

    #Get the limits we want to index to 
    M = int(kmax_kaiser*N)
    L = int((1-kmax_kaiser)*N)

    #print(M)
    #print(L)
    
    k_idx = np.linspace(0,kmax-0.01,N)
    k_idxFac = np.floor(N*k_idx/kmax).astype(int)

    #Pk is easy, but for mu we need to calculate the mu values first
    pk = divided_grid[:M,:M]
    #Calculate the mu values
    mu_ = np.zeros_like(divided_grid)
    for i in range(N):
        kz_ = k_idxFac[i] #radial k 
        for j in range(N):
            kt_ = k_idxFac[j] #transversal k
            mu_[i,j] = compute_mu(kt_,kz_)
    mu_ = np.flip(mu_,axis=0)
    mu = mu_[L:,:M]

    # fig, ax = plt.subplots(figsize=(8, 6))
    # ax.set_xlabel(r'$k = \sqrt{k_x^2 + k_y^2}$')
    # ax.set_ylabel(r'$k_z$')
    # ax.set_title('Divided RSD at low k')
    # ax.grid(visible=True)
    # cax = ax.imshow(pk, origin='lower',extent=(0,0.25,0,0.25),cmap='nipy_spectral',vmin=0,vmax=5)
    # fig.colorbar(cax,label=r'$P_{RSD}(k)/P_{Pos}(k)$')
    # plt.show()
    # plt.close()

    #print(pk)

    f,f_std = curve_fit(kaiser, mu.flatten(), pk.flatten())
    print(u'The fitted value f to quantify the Kaiser effect for high mass galaxies is: {:.5f} \u00B1 {:.5f}'.format(f[0],f_std[0,0]))
    return f[0],f_std[0,0]

print('-'*100)
compute_growthfactor(mlowest_div,0.25)
compute_growthfactor(mlow_div,0.25)
compute_growthfactor(mmid_div,0.25)
compute_growthfactor(mhigh_div,0.25)
compute_growthfactor(mhighest_div,0.25)

#Done without outlier-analysis: maybe for the future? 

#########################################################################
#Next, lets look at the F.o.G. effect, dominant at the highest k values 

#Slice out only higher sqrt(kx^2 + ky^2), mind the axis (reversed)
mveryhigh_highk = mveryhigh[:,int(0.5*N):int(0.9*N)]

"""
fig, ax = plt.subplots(figsize=(8, 6))
ax.set_xlabel(r'$k = \sqrt{k_x^2 + k_y^2}$')
ax.set_ylabel(r'$k_z$')
ax.set_title('Divided RSD of very high mass galaxies')
ax.grid(visible=True)
cax = ax.imshow(mveryhigh, origin='lower',extent=(0,1,0,1),cmap='nipy_spectral',vmin=0,vmax=5)
fig.colorbar(cax,label=r'$P_{RSD}(k)/P_{Pos}(k)$')
fig.savefig('full.png')
plt.show()

fig, ax = plt.subplots(figsize=(12,3))
ax.set_xlabel(r'$k = \sqrt{k_x^2 + k_y^2}$')
ax.set_ylabel(r'$k_z$')
ax.set_title('Divided RSD of very high mass galaxies at high k')
ax.grid(visible=True)
cax = ax.imshow(mveryhigh_highk, origin='lower',extent=(0.75,1,0,1),cmap='nipy_spectral',vmin=0,vmax=5)
fig.colorbar(cax,label=r'$P_{RSD}(k)/P_{Pos}(k)$')
fig.savefig('high_kabs.png')
plt.show()
"""
print(np.shape(mveryhigh_highk))

k_lists = k_idx[int(0.5*N):int(0.9*N)]

pk_lists = np.shape(mveryhigh_highk)[1]
k_range = np.linspace(int(0.5*N),int(0.9*N),N)/N
for i in range(pk_lists):
    plt.plot(k_range,mveryhigh_highk[:,i],label='k_ = {:.2f}'.format(k_lists[i]))
plt.xlabel(r'k_z')
plt.ylabel(r'$P_{RSD}(k)/P_{REAL}(k)$')
plt.title(r'Divided power values as a function of k_z at high k_transversal')
plt.legend()
#plt.show()
plt.close()

#Unfortunately, plotting them all is not very informative, so lets take the mean of the largest 0.1 k values

pk_mean = np.mean(mveryhigh_highk,axis=1)
print(np.shape(pk_mean))
pk_fit = np.polyfit(k_idx,pk_mean,5)

x = k_idx
pk_polynomial = x**5 * pk_fit[0] + x**4 * pk_fit[1] + x**3 * pk_fit[2] + x**2 * pk_fit[3] + x * pk_fit[4] + pk_fit[5]
print(pk_fit[-1])

plt.plot(k_range,pk_mean,label='0.9 < k_ <= 1')
plt.plot(k_range,pk_polynomial,label='polynomial fit')
plt.xlabel(r'k_z')
plt.ylabel(r'$P_{RSD}(k)/P_{REAL}(k)$')
plt.title(r'Mean divided power values as a function of k_z at high k_transversal')
plt.legend()
#plt.show()
plt.close()


