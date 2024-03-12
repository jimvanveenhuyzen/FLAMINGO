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

z0_5_pos, z0_5_rsd, z0_5_div = load_files('z0_5')
z0_8_pos, z0_8_rsd, z0_8_div = load_files('z0_8')
z1_pos,z1_rsd,z1_div = load_files('z1')
z2_pos,z2_rsd,z2_div = load_files('z2')

weak_pos,weak_rsd,weak_div = load_files('WEAK_AGN')
strongest_pos,strongest_rsd,strongest_div = load_files('STRONGEST_AGN')

mall_pos,mall_rsd,mall_div = load_files('mall')
mall16_pos,mall16_rsd,mall16_div = load_files('mall_N16')

#print('The strongest positions')
print(strongest_pos)
print(strongest_rsd)

fig, ax = plt.subplots(figsize=(8, 6))
ax.set_xlabel(r'$k = \sqrt{k_x^2 + k_y^2}$')
ax.set_ylabel(r'$k_z$')
ax.set_title(r'$P_{RSD}$ for all galaxies with STRONGEST_AGN, no SN, Npix=64')
ax.grid(visible=True)
cax = ax.imshow(fullGrid(strongest_rsd), origin='lower',extent=(-1,1,-1,1),cmap='nipy_spectral',norm=matplotlib.colors.LogNorm(vmin=50,vmax=1.1e5))
fig.colorbar(cax,label=r'$P_{RSD}(k)/P_{Pos}(k)$')
plt.savefig('strongest_rsd.png')
#plt.show()
plt.close()

fig, ax = plt.subplots(figsize=(8, 6))
ax.set_xlabel(r'$k = \sqrt{k_x^2 + k_y^2}$')
ax.set_ylabel(r'$k_z$')
ax.set_title(r'$P_{RSD}/P_{real}$ for all galaxies with STRONGEST_AGN, no SN, Npix=64')
ax.grid(visible=True)
cax = ax.imshow(fullGrid(strongest_div), origin='lower',extent=(-1,1,-1,1),cmap='nipy_spectral',vmin=0,vmax=5)
fig.colorbar(cax,label=r'$P_{RSD}(k)/P_{Pos}(k)$')
plt.savefig('strongest.png')
#plt.show()
plt.close()

fig, ax = plt.subplots(figsize=(8, 6))
ax.set_xlabel(r'$k = \sqrt{k_x^2 + k_y^2}$')
ax.set_ylabel(r'$k_z$')
ax.set_title(r'$P_{RSD}/P_{real}$ for all galaxies with WEAK_AGN, no SN, Npix=64')
ax.grid(visible=True)
cax = ax.imshow(fullGrid(weak_div), origin='lower',extent=(-1,1,-1,1),cmap='nipy_spectral',vmin=0,vmax=5)
fig.colorbar(cax,label=r'$P_{RSD}(k)/P_{Pos}(k)$')
plt.savefig('weak.png')
#plt.show()
plt.close()

fig, ax = plt.subplots(figsize=(8, 6))
ax.set_xlabel(r'$k = \sqrt{k_x^2 + k_y^2}$')
ax.set_ylabel(r'$k_z$')
ax.set_title(r'$P_{RSD}/P_{real}$ for all galaxies at z=0.5, no SN, Npix=64')
ax.grid(visible=True)
cax = ax.imshow(fullGrid(z0_5_div), origin='lower',extent=(-1,1,-1,1),cmap='nipy_spectral',vmin=0,vmax=5)
fig.colorbar(cax,label=r'$P_{RSD}(k)/P_{Pos}(k)$')
plt.savefig('z0_5.png')
#plt.show()
plt.close()

fig, ax = plt.subplots(figsize=(8, 6))
ax.set_xlabel(r'$k = \sqrt{k_x^2 + k_y^2}$')
ax.set_ylabel(r'$k_z$')
ax.set_title(r'$P_{RSD}/P_{real}$ for all galaxies at z=0.8, no SN, Npix=64')
ax.grid(visible=True)
cax = ax.imshow(fullGrid(z0_8_div), origin='lower',extent=(-1,1,-1,1),cmap='nipy_spectral',vmin=0,vmax=5)
fig.colorbar(cax,label=r'$P_{RSD}(k)/P_{Pos}(k)$')
plt.savefig('z0_8.png')
#plt.show()
plt.close()

fig, ax = plt.subplots(figsize=(8, 6))
ax.set_xlabel(r'$k = \sqrt{k_x^2 + k_y^2}$')
ax.set_ylabel(r'$k_z$')
ax.set_title(r'$P_{RSD}/P_{real}$ for all galaxies at z=1.0, no SN, Npix=64')
ax.grid(visible=True)
cax = ax.imshow(fullGrid(z1_div), origin='lower',extent=(-1,1,-1,1),cmap='nipy_spectral',vmin=0,vmax=5)
fig.colorbar(cax,label=r'$P_{RSD}(k)/P_{Pos}(k)$')
plt.savefig('z1.png')
#plt.show()
plt.close()

fig, ax = plt.subplots(figsize=(8, 6))
ax.set_xlabel(r'$k = \sqrt{k_x^2 + k_y^2}$')
ax.set_ylabel(r'$k_z$')
ax.set_title(r'$P_{RSD}/P_{real}$ for all galaxies at z=2.0, no SN, Npix=64')
ax.grid(visible=True)
cax = ax.imshow(fullGrid(z2_div), origin='lower',extent=(-1,1,-1,1),cmap='nipy_spectral',vmin=0,vmax=5)
fig.colorbar(cax,label=r'$P_{RSD}(k)/P_{Pos}(k)$')
plt.savefig('z2.png')
#plt.show()
plt.close()

fig, ax = plt.subplots(figsize=(8, 6))
ax.set_xlabel(r'$k = \sqrt{k_x^2 + k_y^2}$')
ax.set_ylabel(r'$k_z$')
ax.set_title(r'$P_{RSD}/P_{real}$ for all galaxies, no SN, Npix=64')
ax.grid(visible=True)
cax = ax.imshow(fullGrid(mall_div), origin='lower',extent=(-1,1,-1,1),cmap='nipy_spectral',vmin=0,vmax=5)
fig.colorbar(cax,label=r'$P_{RSD}(k)/P_{Pos}(k)$')
plt.savefig('mall.png')
#plt.show()
plt.close()

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


def mu_grid(grid):
    N = np.shape(grid)[0]
    k_index = np.linspace(0,kmax-0.01,N)
    k_indexFac = np.floor(N*k_index/kmax).astype(int)
    mu_grid = np.zeros_like(grid)

    for i in range(N):
        kz = k_indexFac[i] #radial k 
        for j in range(N):
            kt = k_idxFac[j] #transversal k
            mu_grid[i,j] = compute_mu(kt,kz)
    mu_grid = np.flip(mu_grid,axis=0)
    return mu_grid

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
#plt.show()
plt.close()

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
#plt.show()
plt.close()

fig, ax = plt.subplots(figsize=(8, 6))
ax.set_xlabel(r'$k = \sqrt{k_x^2 + k_y^2}$')
ax.set_ylabel(r'$k_z$')
ax.set_title('Divided RSD of very high mass galaxies at low k')
ax.grid(visible=True)
cax = ax.imshow(mveryhigh_lowk, origin='lower',extent=(0,0.25,0,0.25),cmap='nipy_spectral',vmin=0,vmax=5)
fig.colorbar(cax,label=r'$P_{RSD}(k)/P_{Pos}(k)$')
#plt.show()
plt.close()

f_veryhigh,f_veryhighStd = curve_fit(kaiser, mveryhigh_mu_lowk.flatten(), mveryhigh_lowk.flatten())
print(u'The fitted value f to quantify the Kaiser effect for high mass galaxies is: {:.3f} \u00B1 {:.5f}'.format(f_veryhigh[0],f_veryhighStd[0,0]))
#print(f_veryhigh[0],u"\u00B1",f_veryhighStd[0,0])

def compute_growthfactor(divided_grid,kmax_kaiser,description):

    #Get the pixel resolution of the grid 
    N = np.shape(divided_grid)[0]

    #Get the limits we want to index to 
    M = int(kmax_kaiser*N)
    L = int((1-kmax_kaiser)*N)

    #The total k numbers
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

    mu_ = mu_grid(divided_grid)
    mu = mu_[L:,:M]

    f,f_std = curve_fit(kaiser, mu.flatten(), pk.flatten())
    print(u'The fitted value f to quantify the Kaiser effect for {} galaxies is: {:.5f} \u00B1 {:.5f}'.format(description,f[0],f_std[0,0]))
    return f[0],f_std[0,0], mu

print('-'*100)
compute_growthfactor(mlowest_div,0.25,'lowest mass')
compute_growthfactor(mlow_div,0.25,'lower mass')
compute_growthfactor(mmid_div,0.25,'middle mass')
compute_growthfactor(mhigh_div,0.25,'higher mass')
compute_growthfactor(mhighest_div,0.25,'highest mass')

print('Now finally for all galaxies we get:')
compute_growthfactor(mall_div,0.25,'all')

compute_growthfactor(mall16_div,0.25,'all (N=16)')

compute_growthfactor(z0_5_div,0.25,'z=0.5')
compute_growthfactor(z0_8_div,0.25,'z=0.8')
compute_growthfactor(z1_div,0.25,'z=1.0')
compute_growthfactor(z2_div,0.25,'z=2.0')


compute_growthfactor(strongest_div,0.25,'strongest')
compute_growthfactor(weak_div,0.25,'weak')

#Done without outlier-analysis: maybe for the future? 

#########################################################################
#Next, lets look at the F.o.G. effect, dominant at the highest k values 

def fog(mu,k,f,sigma):
    """
    This is the function we want to fit to the data to measure the Kaiser effect by determining f 
    Assume the ratio of P(k)s is given by this function: P_RSD/P = (1+f*mu**2)**2
    """
    return kaiser(mu,f) * (1 + 0.5*k**2 * mu**2 * sigma**2) **(-2)

def smooth_grid(grid,factor):
    """
    Function that smoothes over a pixel grid by taking the mean of square pixel areas
    Arguments:
        -grid: the pixel grid of shape Nup by Nup we want to smooth over
        -factor: the compression factor, e.g. factor=2 compresses the grid to shapes (N/2,N/2)
    Return:
        -grid_smoothed: the smoothed grid with resolution (N/factor,N/factor)
    """
    #Set up the variables
    grid_copy = np.copy(grid)
    Nup = np.shape(grid)[0]
    Ndown = int(Nup//factor)

    #Do the compression using numpy boradcasting
    grid_copy = grid_copy.reshape(Ndown,Ndown,factor,factor)
    grid_smoothed = np.mean(grid_copy,axis=(2,3))
    return grid_smoothed

def fit_fog(grid,kaiser_maxk,fog_mink,smoothing_factor):

    #Get the resolution of the new pixel grid
    Ndown = int(np.shape(grid)[0]//smoothing_factor)

    #First smooth over the grid
    grid_smoothed = np.copy(grid)
    grid_smoothed = smooth_grid(grid_smoothed,smoothing_factor)

    #Compute the kaiser effect/growth factor: 
    f,f_std,mu = compute_growthfactor(grid_smoothed,kaiser_maxk,'test')

    #Convert to mu coordinates 
    grid_mu = mu_grid(grid_smoothed)
    mu_val = grid_mu[:,int(fog_mink * Ndown):]
    kaiser_term = kaiser(mu_val,f)

    #Now finally get the ratio P_RSD/P_real divided by the kaiser term, which is (1 + fmu**2)**2
    rsd_over_real = grid_smoothed[:,int(fog_mink * Ndown):] / kaiser_term

    return rsd_over_real,f

fog_test,f_test = fit_fog(mall_div,0.25,0.85,4)
 
f_all,f_all_err,_ = compute_growthfactor(mall_div,0.25,'all')

#First, we want to smooth the array by a factor 2 since the grid is too noisy for N=64
Nbig = 64
Nsmall = 32
cf = int(Nbig//Nsmall) #compression factor

mall_div_resh = np.copy(mall_div)
mall_div_resh = mall_div.reshape(Nsmall,cf,Nsmall,cf)
mall_div_compress = np.mean(mall_div_resh, axis=(1,3))

# mu_1 = np.zeros_like(mall_div_compress)
# for i in range(Nsmall):
#     kz_1 = k_idxFac[i] #radial k 
#     for j in range(Nsmall):
#         kt_1 = k_idxFac[j] #transversal k
#         mu_1[i,j] = compute_mu(kt_1,kz_1)
# mu_1 = np.flip(mu_1,axis=0)

Nsmaller = 16 
cf_2 = int(Nbig//Nsmaller)
mall_div_resh_ = np.copy(mall_div)
mall_div_resh_ = mall_div.reshape(Nsmaller,cf_2,Nsmaller,cf_2)
mall_div_compress_ = np.mean(mall_div_resh_, axis=(1,3))
print(np.shape(mall_div_compress_))
f_all_,_,_ = compute_growthfactor(mall_div_compress_,0.25,'all (compressed)')
mu_2 = mu_grid(mall_div_compress_)
mu_all_ = mu_2[:,int(0.85*Nsmaller):]
kaiser_all_ = kaiser(mu_all_,f_all_)
highk_ = mall_div_compress_[:,int(0.85*Nsmaller):]  * kaiser_all_
#k_range_ = np.linspace(int(0.85*Nsmaller),Nsmaller*kmax,Nsmaller)/Nsmaller

#True factor? 
highk_alt = mall_div_compress_[:,int(0.85*Nsmaller):]  / kaiser_all_
highk_alt_mean = np.mean(highk_alt,axis=1)

k_range_ = np.linspace(0,kmax-0.01,Nsmaller)

print(int(0.85*Nsmaller))
print(kmax)
print(np.linspace(int(0.85*Nsmaller),kmax,Nsmaller)/Nsmaller)

print(np.shape(highk_))
highk_mean_ = np.mean(highk_,axis=1)
pk_lists = np.shape(highk_)[1]
k_val_ = k_idx[int(0.85*N):]
print(k_range_)
for i in range(pk_lists):
    plt.plot(k_range_,highk_[:,i],label='k_ = {:.2f}'.format(k_val_[i]),linestyle='dashed',alpha=0.5)
plt.plot(k_range_,highk_mean_,label='mean k_',color='black',linewidth=3)
plt.xlabel(r'$k_z$')
plt.ylabel(r'$P_{RSD}(k)/P_{REAL}(k)$')
plt.title(r'Divided power values as a function of k_z at high k_transversal')
plt.legend()
plt.savefig('fog_N16.png')
#plt.show()
plt.close()

for i in range(pk_lists):
    plt.plot(k_range_,highk_alt[:,i],label='k_ = {:.2f}'.format(k_val_[i]),linestyle='dashed',alpha=0.5)
plt.plot(k_range_,highk_alt_mean,label='mean k_',color='black',linewidth=3)
plt.xlabel(r'$k_z$')
plt.ylabel(r'$P_{RSD}(k)/P_{REAL}(k)$')
plt.title(r'Divided power values as a function of k_z at high k_transversal (Alternate)')
plt.legend()
plt.savefig('fog_N16_alt.png')
#plt.show()
plt.close()

def gaussian_func(x,a,b,x0,sigma):
    """
    Define a Gaussian function with a vertical shift factor b and scale factor a
    """
    return b + a * np.exp(-(x-x0)**2/(2*sigma**2)) 

def lorenztian_func(x,a,b,x0,gamma):
    return b + a * (1/np.pi) * gamma / ((x-x0)**2 + gamma**2)

k_new = np.concatenate((-1*np.flip(k_range_),k_range_))
print(k_new)
highk_mean_new = np.concatenate((np.flip(highk_mean_),highk_mean_))

popt,pcov = curve_fit(gaussian_func,k_new,highk_mean_new,p0=[1,1,0,1])
lorentz,_ = curve_fit(lorenztian_func,k_new,highk_mean_new,p0=[1,1,0,1])
fog_fac = gaussian_func(k_new,*popt)
fog_lorentz = gaussian_func(k_new,*lorentz)

print('the parameters are:',popt)
print('the parameters are:',lorentz)

plt.plot(k_new,highk_mean_new,label='mean k_',color='black',linewidth=3)
plt.plot(k_new,fog_fac,color='red',linewidth=2,linestyle='dashed',label='Gaussian fit')
plt.plot(k_new,fog_lorentz,color='green',linewidth=2,linestyle='dashed',label='Lorentzian fit')
plt.xlabel(r'$k_z$')
plt.ylabel(r'$P_{RSD}(k)/P_{REAL}(k)$')
plt.title(r'Divided power values as a function of k_z at high k_transversal')
plt.legend()
plt.savefig('fog_N16_gaussian.png')
#plt.show()
plt.close()

mu_test = mu_grid(mall_div_compress)

f_all,f_all_err,_ = compute_growthfactor(mall_div_compress,0.25,'all (compressed)')

mu_all = mu_test[:,int(0.9*Nsmall):] 

kaiser_all = kaiser(mu_all,f_all)
print(kaiser_all)
print(np.shape(kaiser_all))

highk = mall_div_compress[:,int(0.9*Nsmall):]  * kaiser_all
k_val = k_idx[int(0.9*N):]
print(highk.shape)

pk_lists = np.shape(highk)[1]
#k_range = np.linspace(int(0.9*Nsmall),Nsmall*kmax,Nsmall)/Nsmall
k_range = np.linspace(0,kmax-0.01,Nsmall)

highk_mean = np.mean(highk,axis=1)

for i in range(pk_lists):
    plt.plot(k_range,highk[:,i],label='k_ = {:.2f}'.format(k_val[i]),linestyle='dashed',alpha=0.5)
plt.plot(k_range,highk_mean,label='mean k_',color='black',linewidth=3)
plt.xlabel(r'$k_z$')
plt.ylabel(r'$P_{RSD}(k)/P_{REAL}(k)$')
plt.title(r'Divided power values as a function of k_z at high k_transversal')
plt.legend()
plt.savefig('fog_N32.png')
#plt.show()
plt.close()


fig, ax = plt.subplots(figsize=(8, 6))
ax.set_xlabel(r'$k = \sqrt{k_x^2 + k_y^2}$')
ax.set_ylabel(r'$k_z$')
ax.set_title(r'$P_{RSD}/P_{real}$ for all galaxies, no SN, Npix=64')
ax.grid(visible=True)
cax = ax.imshow(fullGrid(mall_div), origin='lower',extent=(-1,1,-1,1),cmap='nipy_spectral',vmin=0,vmax=5)
fig.colorbar(cax,label=r'$P_{RSD}(k)/P_{Pos}(k)$')
#plt.savefig('mall.png')
#plt.show()
plt.close()

fig, ax = plt.subplots(figsize=(8, 6))
ax.set_xlabel(r'$k = \sqrt{k_x^2 + k_y^2}$')
ax.set_ylabel(r'$k_z$')
ax.set_title(r'$P_{RSD}/P_{real}$ for all galaxies, no SN, Npix=16')
ax.grid(visible=True)
cax = ax.imshow(fullGrid(mall_div_compress_), origin='lower',extent=(-1,1,-1,1),cmap='nipy_spectral',vmin=0,vmax=5)
fig.colorbar(cax,label=r'$P_{RSD}(k)/P_{Pos}(k)$')
plt.savefig('smoothed_mall.png')
#plt.show()
plt.close()

fig, ax = plt.subplots(figsize=(8, 6))
ax.set_xlabel(r'$k = \sqrt{k_x^2 + k_y^2}$')
ax.set_ylabel(r'$k_z$')
ax.set_title(r'$P_{RSD}/P_{real}$ for all galaxies without smoothing, no SN, Npix=16')
ax.grid(visible=True)
cax = ax.imshow(fullGrid(mall16_div), origin='lower',extent=(-1,1,-1,1),cmap='nipy_spectral',vmin=0,vmax=5)
fig.colorbar(cax,label=r'$P_{RSD}(k)/P_{Pos}(k)$')
plt.savefig('mall_N16.png')
#plt.show()
plt.close()

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

#Slice out only higher sqrt(kx^2 + ky^2), mind the axis (reversed)
mveryhigh_highk = mhighest_div[:,int(0.9*N):]

print(np.shape(mveryhigh_highk))

k_lists = k_idx[int(0.9*N):]

pk_lists = np.shape(mveryhigh_highk)[1]
k_range = np.linspace(int(0.9*N),N*kmax,N)/N
for i in range(pk_lists):
    plt.plot(k_range,mveryhigh_highk[:,i],label='k_ = {:.2f}'.format(k_lists[i]))
plt.xlabel(r'$k_z$')
plt.ylabel(r'$P_{RSD}(k)/P_{REAL}(k)$')
plt.title(r'Divided power values as a function of k_z at high k_transversal')
plt.legend()
#plt.show()
plt.close()


#The code below tries to fit a straight line but this is outdated for now 

"""
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

mhighest_highk = mhighest_div[:,int(0.9*N):]
mhigh_highk = mhigh_div[:,int(0.9*N):]
mmid_highk = mmid_div[:,int(0.9*N):]
mlow_highk = mlow_div[:,int(0.9*N):]
mlowest_highk = mlowest_div[:,int(0.9*N):]

mhighest_mean = np.mean(mhighest_highk,axis=1)
mhigh_mean = np.mean(mhigh_highk,axis=1)
mmid_mean = np.mean(mmid_highk,axis=1)
mlow_mean = np.mean(mlow_highk,axis=1)
mlowest_mean = np.mean(mlowest_highk,axis=1)

print(np.mean(mhighest_mean))
print(np.mean(mhigh_mean))
print(np.mean(mmid_mean))
print(np.mean(mlow_mean))
print(np.mean(mlowest_mean))

mhighest_fit = np.polyfit(k_idx,mhighest_mean,1)
print(mhighest_fit)
mhigh_fit = np.polyfit(k_idx,mhigh_mean,1)
mmid_fit = np.polyfit(k_idx,mmid_mean,1)
mlow_fit = np.polyfit(k_idx,mlow_mean,1)
mlowest_fit = np.polyfit(k_idx,mlowest_mean,1)
print(mhigh_fit)
print(mmid_fit)
print(mlow_fit)
print(mlowest_fit)

plt.plot(k_range,mhighest_mean,label='highest M')
plt.plot(k_range,mhigh_mean,label='high M')
plt.plot(k_range,mmid_mean,label='mid M')
plt.plot(k_range,mlow_mean ,label='low M')
plt.plot(k_range,mlowest_mean,label='lowest M')
plt.xlabel(r'k_z')
plt.ylabel(r'$P_{RSD}(k)/P_{REAL}(k)$')
plt.title(r'Mean divided power values as a function of k_z with 0.9 <= k <= 1.0')
plt.legend()
plt.savefig('fog_compared.png')
#plt.show()
plt.close()
"""




