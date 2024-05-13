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

fig, ax = plt.subplots(figsize=(8, 6))
ax.set_xlabel(r'$k_x$')
ax.set_ylabel(r'$k_y$')
ax.set_title('RSD 1D grid: LoS = [0,0,1], Ngrid = 64')
ax.grid(visible=True)
cax = ax.imshow(mall_pos,extent=(-1, 1, -1, 1), origin='lower', cmap='nipy_spectral',norm=matplotlib.colors.LogNorm(vmin=50,vmax=1.1e5))
fig.colorbar(cax,label='P(k)')
plt.savefig('mall_pos.png')
plt.close()

fig, ax = plt.subplots(figsize=(8, 6))
ax.set_xlabel(r'$k_x$')
ax.set_ylabel(r'$k_y$')
ax.set_title('RSD 1D grid: LoS = [0,0,1], Ngrid = 64')
ax.grid(visible=True)
cax = ax.imshow(mall_pos_nonoise,extent=(-1, 1, -1, 1), origin='lower', cmap='nipy_spectral',norm=matplotlib.colors.LogNorm(vmin=50,vmax=1.1e5))
fig.colorbar(cax,label='P(k)')
plt.savefig('mall_pos_nonoise.png')
plt.close()

mall_div_realnoiseless = mall_rsd/mall_pos_nonoise

fig, ax = plt.subplots(figsize=(10,10))
cax1 = ax.imshow(mall_div, origin='lower',extent=(0,1,0,1),cmap='nipy_spectral',vmin=0.5,vmax=2.5)
ax.set_xlabel(r'$k = \sqrt{k_x^2 + k_y^2}$')
ax.set_ylabel(r'$k_z$')
ax.set_title(r'$P_{RSD}/P_{real}$ for all galaxies')
ax.grid(visible=True)
fig.colorbar(cax1,label=r'$P_{RSD}(k)/P_{Pos}(k)$')
plt.savefig('mdiv.png')
#plt.show()
plt.close()

fig, ax = plt.subplots(figsize=(10,10))
cax1 = ax.imshow(mall_div_realnoiseless, origin='lower',extent=(0,1,0,1),cmap='nipy_spectral',vmin=0.5,vmax=2.5)
ax.set_xlabel(r'$k = \sqrt{k_x^2 + k_y^2}$')
ax.set_ylabel(r'$k_z$')
ax.set_title(r'$P_{RSD}/P_{real}$ for all galaxies')
ax.grid(visible=True)
fig.colorbar(cax1,label=r'$P_{RSD}(k)/P_{Pos}(k)$')
plt.savefig('mdiv_realnoiseless.png')
#plt.show()
plt.close()

#Load in the 'weights**2' to correct for the shotnoise 
mall_numval = np.load('/net/draco/data2/vanveenhuyzen/rsd_project/nbodykit_sourcecode/griddata/number_of_values/numval_mall.npy')
#Seems we have some weird artefact so lets remove it
mall_numvalTrue = np.ones_like(mall_numval) * mall_numval[-1,:]

def compute_mu(k_trans,k_z):
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
    """
    This is the function we want to fit to the data to measure the Kaiser effect by determining f 
    Assume the ratio of P(k)s is given by this function: P_RSD/P = (1+f*mu**2)**2
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
np.save('kaiser.npy',theoretical_kaiser)

weights_custom = np.ones((64,64))
weights_custom[:35,:25] = 5 
plt.imshow(weights_custom,origin='lower',extent=(0,1,0,1))
plt.colorbar()
#plt.show()
plt.close()

weights_var = np.load('fitting/weights_0506.npy')
weights_nokaiser = np.load('weights_nokaiser.npy')
weights_smooth = np.load('weights_smooth.npy')
weights_malldiv = np.load('weights_malldiv.npy')

def weights_comb(grid,weights,shotnoise,mu):
    weights_poisson = weights 
    weights_sn = (grid/shotnoise)**(1) * (((1-mu**2))**0.5)
    #weights_sn = weights_custom
    weights_poisson = np.ones_like(grid)
    #weights_sn = 1

    #weights_tot = weights_poisson*weights_sn
    #weights_tot[:15,:5] += 50
    #weights_tot[:35,:20] += 10
    #weights_tot = np.log10(weights_var)
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

def lorentz(x,mu,kz,power):
    f = x[0]
    #gamma = x[1]
    #model = ((1+f*(mu**2))**2) * (gamma/((gamma)**2 + (kz)**2))
    model = ((1+f*(mu**2))**2) * 1/(1+(kz)**2)

    diff_sqr_weighted = weights_total**2 * (model-power)**2 / np.sum(weights_total)
    #Finally, compute the Root-Mean-Square Error (RMSE), which is what we will try to minimize! 
    return np.sqrt(np.mean(diff_sqr_weighted))

#result_lorentz = minimize(lorentz,[0.5,1.5],args=(mu_mall64,k_z,mall_div),method='L-BFGS-B',bounds=[(0,1),(0,3)])
result_lorentz = minimize(lorentz,0.5,args=(mu_mall64,k_z,mall_div),method='L-BFGS-B',bounds=[(0,1)])
print('Using L-BFGS-B minimization and a Lorentzian profile:',result_lorentz.x)

def model_lorentz(x,mu,kz):
    f = x[0]
    #gamma = x[1]
    #model = ((1+f*(mu**2))**2) * (gamma/((gamma)**2 + (kz)**2))
    model = ((1+f*(mu**2))**2) * (1/(1+(kz)**2))
    return model

model = model_lorentz(result_lorentz.x,mu_mall64,k_z)

def residuals(ydata,model):
    diff_sq = (ydata-model)**2
    diff_sq_weighted = weights_total**2 * diff_sq / np.sum(weights_total)
    return diff_sq,diff_sq_weighted

res,res_weighted = residuals(mall_div,model)

fig, ax = plt.subplots(2,2,figsize=(10,10))
cax1 = ax[0,0].imshow(mall_div, origin='lower',extent=(0,1,0,1),cmap='nipy_spectral',vmin=0.5,vmax=2.5)
#ax[0,0].set_xlabel(r'$k = \sqrt{k_x^2 + k_y^2}$')
ax[0,0].set_ylabel(r'$k_z$')
ax[0,0].set_title(r'$P_{RSD}/P_{real}$ for all galaxies')
ax[0,0].grid(visible=True)
fig.colorbar(cax1,ax=ax[0,0],label=r'$P_{RSD}(k)/P_{Pos}(k)$')

cax2 = ax[0,1].imshow(model,extent=(0,1,0,1),cmap='nipy_spectral',vmin=0.5,vmax=2.5)
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

fig.suptitle('Comparison of true and modelled RSD/Real grid ({})'.format('Lorentzian'))
fig.savefig('fitting/lorentz_0506.png')
#plt.show()
plt.close()

def lorentz_pow(x,mu,kz,power):
    f = x[0]
    beta = x[1]

    model = ((1+f*(mu)**2)**2) * (1/(1+(kz)**beta))

    diff_sqr_weighted = weights_total**2 * (model-power)**2 / np.sum(weights_total)
    return np.sqrt(np.mean(diff_sqr_weighted))

def model_lorentz_pow(x,mu,kz):
    f = x[0]
    beta = x[1]

    model = ((1+f*(mu)**2)**2) * (1/(1 + (kz)**beta))
    return model 

result_lorentz_pow = minimize(lorentz_pow,[0.5,2],args=(mu_mall64,k_z,mall_div),method='L-BFGS-B',bounds=[(0,1),(0,3)])
print('Using a free power law',result_lorentz_pow.x)

model_pow = model_lorentz_pow(result_lorentz_pow.x,mu_mall64,k_z)

res,res_weighted = residuals(mall_div,model_pow)

fig, ax = plt.subplots(2,2,figsize=(10,10))
cax1 = ax[0,0].imshow(mall_div, origin='lower',extent=(0,1,0,1),cmap='nipy_spectral',vmin=0.5,vmax=2.5)
#ax[0,0].set_xlabel(r'$k = \sqrt{k_x^2 + k_y^2}$')
ax[0,0].set_ylabel(r'$k_z$')
ax[0,0].set_title(r'$P_{RSD}/P_{real}$ for all galaxies')
ax[0,0].grid(visible=True)
fig.colorbar(cax1,ax=ax[0,0],label=r'$P_{RSD}(k)/P_{Pos}(k)$')

cax2 = ax[0,1].imshow(model_pow,extent=(0,1,0,1),cmap='nipy_spectral',vmin=0.5,vmax=2.5)
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
fig.savefig('fitting/lorentz_powerfree.png')
#plt.show()
plt.close()

def omega_matter(f):
    #Sargent and Turner (1977)
    return f**(1/0.545)

omega_lorentz = omega_matter(result_lorentz.x[0])
omega_lorentz_pow = omega_matter(result_lorentz_pow.x[0])
print(omega_lorentz)
print('The value of Omega_m is {:.3f}'.format(omega_lorentz_pow))

omega_m = 0.306
f_true = omega_m**0.545
print('The true value of Omega_m is {}'.format(omega_m))
print('The true value of f is {:.5f}'.format(f_true))

kz = np.linspace(0,1,100)
def prof(x,gamma):
    return 1/(1+(x)**gamma)

func1 = prof(kz,0.701)
func2 = prof(kz,2)

plt.plot(kz,func1)
plt.plot(kz,func2)
plt.xlabel('kz')
#plt.show()
plt.close()

def lorentz_pow(x,mu,kz,power):
    f = x[0]
    beta = x[1]

    model = ((1+f*(mu)**2)**2) * (1/(1+(kz)**beta))

    diff_sqr_weighted = weights_total**2 * (model-power)**2 / np.sum(weights_total)
    return np.sqrt(np.mean(diff_sqr_weighted))

def model_lorentz_pow(x,mu,kz):
    f = x[0]
    beta = x[1]

    model = ((1+f*(mu)**2)**2) * (1/(1 + (kz)**beta))
    return model 

result_lorentz_pow = minimize(lorentz_pow,[0.5,2],args=(mu_mall64,k_z,mall_div),method='L-BFGS-B',bounds=[(0,1),(0,3)])
print('Using a free power law',result_lorentz_pow.x)

model_pow = model_lorentz_pow(result_lorentz_pow.x,mu_mall64,k_z)

res,res_weighted = residuals(mall_div,model_pow)

fig, ax = plt.subplots(2,2,figsize=(10,10))
cax1 = ax[0,0].imshow(mall_div, origin='lower',extent=(0,1,0,1),cmap='nipy_spectral',vmin=0.5,vmax=2.5)
#ax[0,0].set_xlabel(r'$k = \sqrt{k_x^2 + k_y^2}$')
ax[0,0].set_ylabel(r'$k_z$')
ax[0,0].set_title(r'$P_{RSD}/P_{real}$ for all galaxies')
ax[0,0].grid(visible=True)
fig.colorbar(cax1,ax=ax[0,0],label=r'$P_{RSD}(k)/P_{Pos}(k)$')

cax2 = ax[0,1].imshow(model_pow,extent=(0,1,0,1),cmap='nipy_spectral',vmin=0.5,vmax=2.5)
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
fig.savefig('fitting/lorentz_powerfree.png')
#plt.show()
plt.close()

def lorentz_pow(x,mu,kz,power):
    f = x[0]
    a = x[1]
    b = x[2]

    model = ((1+f*(mu)**2)**2) * (a/(a+ (kz)**b))

    diff_sqr_weighted = weights_total**2 * (model-power)**2 / np.sum(weights_total)
    return np.sqrt(np.mean(diff_sqr_weighted))

def model_lorentz_pow(x,mu,kz):
    f = x[0]
    a = x[1]
    b = x[2]

    model = ((1+f*(mu)**2)**2) * (a/(a+ (kz)**b))
    return model 

result_lorentz_pow = minimize(lorentz_pow,[0.5,1,1],args=(mu_mall64,k_z,mall_div),method='L-BFGS-B',bounds=[(0,1),(0,5),(0,3)])
print('Using free parameters: f,a',result_lorentz_pow.x)

model_pow = model_lorentz_pow(result_lorentz_pow.x,mu_mall64,k_z)

res,res_weighted = residuals(mall_div,model_pow)

fig, ax = plt.subplots(2,2,figsize=(10,10))
cax1 = ax[0,0].imshow(mall_div, origin='lower',extent=(0,1,0,1),cmap='nipy_spectral',vmin=0.5,vmax=2.5)
#ax[0,0].set_xlabel(r'$k = \sqrt{k_x^2 + k_y^2}$')
ax[0,0].set_ylabel(r'$k_z$')
ax[0,0].set_title(r'$P_{RSD}/P_{real}$ for all galaxies')
ax[0,0].grid(visible=True)
fig.colorbar(cax1,ax=ax[0,0],label=r'$P_{RSD}(k)/P_{Pos}(k)$')

cax2 = ax[0,1].imshow(model_pow,extent=(0,1,0,1),cmap='nipy_spectral',vmin=0.5,vmax=2.5)
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
fig.savefig('fitting/lorentz_afree2204.png')
#plt.show()
plt.close()

def gaussian(x,mu,kz,power):
    A = x[0]
    B = x[1]
    C = x[2]

    model = A/(1 + B*kz**2) + C*kz**2

    #model = ((1+f*(mu)**2)**2) * np.exp(-(f*kz*sigma_v)**2)

    diff_sqr_weighted = weights_total**2 * (model-power)**2 / np.sum(weights_total)
    return np.sqrt(np.mean(diff_sqr_weighted))

def model_gaussian(x,mu,kz):
    A = x[0]
    B = x[1]
    C = x[2]

    model = A/(1 + B*kz**2) + C*kz**2

    #model = ((1+f*(mu)**2)**2) * np.exp(-(f*kz*sigma_v)**2)
    return model 

result_gaussian = minimize(gaussian,[1,1,1],args=(mu_mall64,k_z,mall_div),method='L-BFGS-B',bounds=[(-100,100),(-100,100),(-100,100)])
print('Using free parameters for a gaussian: f,sigma_v',result_gaussian.x)

model_gaussian = model_gaussian(result_gaussian.x,mu_mall64,k_z)

res,res_weighted = residuals(mall_div,model_gaussian)

fig, ax = plt.subplots(2,2,figsize=(10,10))
cax1 = ax[0,0].imshow(mall_div, origin='lower',extent=(0,1,0,1),cmap='nipy_spectral',vmin=0.5,vmax=2.5)
#ax[0,0].set_xlabel(r'$k = \sqrt{k_x^2 + k_y^2}$')
ax[0,0].set_ylabel(r'$k_z$')
ax[0,0].set_title(r'$P_{RSD}/P_{real}$ for all galaxies')
ax[0,0].grid(visible=True)
fig.colorbar(cax1,ax=ax[0,0],label=r'$P_{RSD}(k)/P_{Pos}(k)$')

cax2 = ax[0,1].imshow(model_gaussian,extent=(0,1,0,1),cmap='nipy_spectral',vmin=0.5,vmax=2.5)
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
fig.savefig('fitting/model_kwan2022.png')
#plt.show()
plt.close()

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
print('Model by Kwan (2022, using free parameters growth factor,dispersion: f, sigma_v',result_lorentz_pow.x)

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

    diff_sqr_weighted = weights_total[:48,:48]**2 * (model-power)**2 / np.sum(weights_total[:48,:48])
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

kz_reduced = k_z[:48,:48]
mu_reduced = mu_mall64[:48,:48]
mall_div_reduced = mall_div[:48,:48]

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

plt.show()
plt.close()
