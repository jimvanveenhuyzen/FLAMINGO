import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
from scipy.optimize import minimize

kmax=1

def load_files(file):
    path = '/net/draco/data2/vanveenhuyzen/rsd_project/nbodykit_sourcecode/griddata/'
    m_pos = np.load(path+'grid_pos_{}.npy'.format(file))
    m_rsd = np.load(path+'grid_rsd_{}.npy'.format(file))
    return m_pos,m_rsd,(m_rsd/m_pos)

#Load in the data: 
mall_pos,mall_rsd,mall_div = load_files('mall')

#Load in an alternate for mall_pos without any noise 
mall_pos_nonoise = np.load('/net/draco/data2/vanveenhuyzen/rsd_project/data_analysis/Preal_0506.npy')

#Load in the 'weights**2' to correct for the shotnoise 
mall_numval = np.load('/net/draco/data2/vanveenhuyzen/rsd_project/nbodykit_sourcecode/griddata/number_of_values/numval_mall.npy')
#Seems we have some weird artefact so lets remove it
mall_numvalTrue = np.ones_like(mall_numval) * mall_numval[-1,:]

def compute_mu(k_trans,k_z):
    """Use k_transerval and k_z to compute the angle mu=cos(theta), with theta
    being the angle of a vector k_total with the k_z axis (y-axis!)
    """
    k_total = np.sqrt(k_z**2 + k_trans**2)
    if k_total < 1e-5:
        if k_z < 1e-5:
            return 0
        else:
            return 1
    else:
        mu = k_z / k_total
    return mu

def compute_ktotal(k_trans,k_z):
    k_total = np.sqrt(k_z**2 + k_trans**2)
    if k_total < 1e-5:
        k_total = 1e-5
    return k_total

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
    """Compute the functional influence of the Kaiser effect
    """
    return (1+f*(mu**2))**2

# Create meshgrid
x = np.linspace(0, 1, 64)  # X coordinates range from 0 to 1
y = np.linspace(0, 1, 64)  # Y coordinates range from 0 to 1
# Adjust y values to be bottom-up
y = np.flipud(y)
# Create meshgrid
k_trans,k_z = np.meshgrid(x, y)

print('The values of kz are\n')
print(k_z)

mu_mall64,_ = mu_grid(mall_div)

#Flip the row order of k_z and mu since mall_div is also flipped
k_z = np.flip(k_z,axis=0)
mu_mall64 = np.flip(mu_mall64,axis=0)

#Compute the theoretical kaiser effect
theoretical_kaiser = kaiser(mu_mall64,f=0.524467)

#Import the computed weights
weights_smooth = np.load('weights_smooth.npy')
weights_malldiv = np.load('weights_malldiv.npy')

def weights_comb(grid,weights,shotnoise,mu):
    weights_poisson = weights 
    weights_sn = (grid/shotnoise)**(1) * (((1-mu**2))**0.5)
    #weights_sn = weights_custom
    weights_poisson = np.ones_like(grid)
    #weights_sn = 1

    #weights_tot = weights_poisson*weights_sn
    #weights_tot = weights_var
    #weights_tot = weights_smooth
    weights_tot = weights_malldiv
    return weights_tot
weights = np.sqrt(mall_numvalTrue)
power_rsd = np.genfromtxt('/net/draco/data2/vanveenhuyzen/rsd_project/nbodykit_sourcecode/power_spectra/power_rsd_0122.txt')
sn = power_rsd[0,2]

weights_total = weights_comb(mall_rsd,weights,sn,mu_mall64)
plt.imshow(weights_total,origin='lower',extent=(0,1,0,1))
plt.title('Weights = (w_SN * w_numpoints)**2, using RSD/SN * (1-mu^2)^0.5')
plt.xlabel(r'$k = \sqrt{k_x^2 + k_y^2}$')
plt.ylabel(r'$k_z$')
plt.grid(visible=True)
plt.colorbar()
#plt.show()
plt.close()

def residuals(ydata,model):
    diff_sq = (ydata-model)**2
    diff_sq_weighted = weights_total**2 * diff_sq / np.sum(weights_total)
    return diff_sq,diff_sq_weighted

def omega_matter(f):
    #Sargent and Turner (1977)
    return f**(1/0.545)

omega_m = 0.306
f_true = omega_m**0.545
print('The true value of Omega_m is {}'.format(omega_m))
print('The true value of f is {:.5f}'.format(f_true))

def lorentz_pow(x,mu,kz,power):
    f = x[0]
    sigma_v = x[1]

    model = ((1+(f/1.04)*(mu)**2)**2) * 1/(1 + (kz*sigma_v)**2)

    diff_sqr_weighted = weights_total**2 * (model-power)**2 / np.sum(weights_total)
    return np.sqrt(np.mean(diff_sqr_weighted))

def model_lorentz_pow(x,mu,kz):
    f = x[0]
    sigma_v = x[1]

    model = ((1+(f/1.04)*(mu)**2)**2) * 1/(1 + (kz*sigma_v)**2)
    return model 

result_lorentz_pow = minimize(lorentz_pow,[0.5,1],args=(mu_mall64,k_z,mall_div),method='L-BFGS-B',bounds=[(0,1),(0,3)])
print('Model by Kwan (2022), using free parameters growth factor,dispersion: f, sigma_v',result_lorentz_pow.x)

model_gaussian = model_lorentz_pow(result_lorentz_pow.x,mu_mall64,k_z)

res,res_weighted = residuals(mall_div,model_gaussian)

fig, ax = plt.subplots(2,2,figsize=(10,10))
cax1 = ax[0,0].imshow(mall_div, origin='lower',extent=(0,1,0,1),cmap='nipy_spectral',vmin=0.5,vmax=2.5)
#ax[0,0].set_xlabel(r'$k = \sqrt{k_x^2 + k_y^2}$')
ax[0,0].set_ylabel(r'$k_z$')
ax[0,0].set_title(r'$P_{RSD}/P_{real}$ for all galaxies')
ax[0,0].grid(visible=True)
fig.colorbar(cax1,ax=ax[0,0],label=r'$P_{RSD}(k)/P_{Pos}(k)$')

cax2 = ax[0,1].imshow(model_gaussian,origin='lower',extent=(0,1,0,1),cmap='nipy_spectral',vmin=0.5,vmax=2.5)
#ax[0,1].set_xlabel(r'$k = \sqrt{k_x^2 + k_y^2}$')
ax[0,1].set_ylabel(r'$k_z$')
ax[0,1].set_title(r'$P_{RSD}/P_{real}$ all galaxies using the model')
ax[0,1].grid(visible=True)
fig.colorbar(cax2,ax=ax[0,1],label=r'$P_{RSD}(k)/P_{Pos}(k)$')

cax3 = ax[1,0].imshow(res, origin='lower',extent=(0,1,0,1),cmap='nipy_spectral',vmin=0,vmax=2)
ax[1,0].set_xlabel(r'$k = \sqrt{k_x^2 + k_y^2}$')
ax[1,0].set_ylabel(r'$k_z$')
ax[1,0].set_title(r'Residuals')
ax[1,0].grid(visible=True)
fig.colorbar(cax3,ax=ax[1,0],label=r'(data-model)^2')

cax4 = ax[1,1].imshow(res_weighted, origin='lower',extent=(0,1,0,1),cmap='nipy_spectral',norm=matplotlib.colors.LogNorm(vmin=5e-6,vmax=1e-3))
ax[1,1].set_xlabel(r'$k = \sqrt{k_x^2 + k_y^2}$')
ax[1,1].set_ylabel(r'$k_z$')
ax[1,1].set_title(r'Weighted residuals')
ax[1,1].grid(visible=True)
fig.colorbar(cax4,ax=ax[1,1],label=r'w^2 * (data-model)^2 / sum(w)')

fig.suptitle('Comparison of true and modelled RSD/Real grid ({})'.format('Lorentzian_Power'))
fig.savefig('fitting/model_kwan2022_reducednoise.png')
#plt.show()
plt.close()

def byrohl(x,mu,kz,power):
    f = x[0]
    sigma_v = x[1]
    #b = x[2]

    model = ((1+(f/1.04)*(mu)**2)**2) * 1/(1 + (kz*sigma_v)**2)

    diff_sqr_weighted = weights_total[:32,:32]**2 * (model-power)**2 / np.sum(weights_total[:32,:32])
    #diff_sqr_weighted = (model-power)**2
    return np.sqrt(np.mean(diff_sqr_weighted))

def model_byrohl(x,mu,kz):
    f = x[0]
    sigma_v = x[1]
    #b = x[2]

    model = ((1+(f/1.04)*(mu)**2)**2) * 1/(1 + (kz*sigma_v)**2)
    return model 

#result_byrohl = minimize(byrohl,[0.5,1],args=(mu_mall64,k_z,mall_div),method='L-BFGS-B',bounds=[(0,1),(0,3)])
#print('Byrohl model: Using free parameters growth factor, disp: f, sigma_v',result_byrohl.x)

#print(weights_smooth)

plt.imshow(weights_smooth,origin='lower')
plt.show()

kz_reduced = k_z[:32,:32]
mu_reduced = mu_mall64[:32,:32]
mall_div_reduced = mall_div[:32,:32]

fig, ax = plt.subplots(figsize=(10,10))
cax1 = ax.imshow(mall_div_reduced, origin='lower',extent=(0,0.75,0,0.75),cmap='nipy_spectral',vmin=0.5,vmax=2.5)
#ax[0,0].set_xlabel(r'$k = \sqrt{k_x^2 + k_y^2}$')
ax.set_ylabel(r'$k_z$')
ax.set_title(r'$P_{RSD}/P_{real}$ for all galaxies')
ax.grid(visible=True)
fig.colorbar(cax1,ax=ax,label=r'$P_{RSD}(k)/P_{Pos}(k)$')

plt.show()
plt.close()

result_lorentz_pow = minimize(byrohl,[0.5,1],args=(mu_reduced,kz_reduced,mall_div_reduced),method='L-BFGS-B',bounds=[(0,1),(0,3)])
print('Model by Kwan up to k=0.75 using free parameters growth factor,dispersion: f, sigma_v',result_lorentz_pow.x)

model_gaussian = model_byrohl(result_lorentz_pow.x,mu_reduced,kz_reduced)

fig, ax = plt.subplots(figsize=(10,10))
cax1 = ax.imshow(model_gaussian, origin='lower',extent=(0,0.75,0,0.75),cmap='nipy_spectral',vmin=0.5,vmax=2.5)
#ax[0,0].set_xlabel(r'$k = \sqrt{k_x^2 + k_y^2}$')
ax.set_ylabel(r'$k_z$')
ax.set_title(r'$P_{RSD}/P_{real}$ for all galaxies')
ax.grid(visible=True)
fig.colorbar(cax1,ax=ax,label=r'$P_{RSD}(k)/P_{Pos}(k)$')

plt.savefig('model_k0p75.png')
plt.show()
plt.close()
