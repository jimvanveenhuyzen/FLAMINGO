import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
from scipy.optimize import minimize

kmax=1

def load_files(file):
    """Quick function to load in the real and redshift-space grids created using FFTPower_custom

    Parameters: 
    ----------
    file : string
        Name of the grid, specified by a main characteristic like mass cut, redshift or AGN feedback

    Returns:
    -------
    m_pos : 2D numpy array (Npixel,Npixel)
        The gridded power values in real space
    m_rsd : 2D numpy array (Npixel,Npixel)
        The gridded power values in redshift space
    m_div : 2D numpy array (Npixel,Npixel)
        The gridded divided power values
    """
    path = '/net/draco/data2/vanveenhuyzen/rsd_project/nbodykit_sourcecode/griddata/'
    m_pos = np.load(path+'grid_pos_{}.npy'.format(file))
    m_rsd = np.load(path+'grid_rsd_{}.npy'.format(file))
    m_div = m_rsd/m_pos
    return m_pos,m_rsd,m_div

#Load in the data: 
mall_pos,mall_rsd,mall_div = load_files('mall')

#Load in an alternate for mall_pos without any noise, unused
mall_pos_nonoise = np.load('/net/draco/data2/vanveenhuyzen/rsd_project/data_analysis/Preal_0506.npy')

#Load in the 'weights**2' to correct for the shotnoise, unused
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

def mu_grid(grid):
    """Uses the compute_mu function to construct a grid with at each point the mu (angular) value 
    """
    
    N = np.shape(grid)[0] #Get the axis size 

    k_index = np.linspace(0,kmax-0.01,N)
    k_indexFac = np.floor(N*k_index/kmax).astype(int) #Like before, use integers to avoid numerical errors

    mu_grid = np.zeros_like(grid)
    ktotal_grid = np.zeros_like(grid)
    #Loop through the whole grid to compute mu everywhere
    for i in range(N):
        kz = k_indexFac[i] #radial k (y-axis)

        for j in range(N):
            kt = k_indexFac[j] #transversal k (x_axis)

            mu_grid[i,j] = compute_mu(kt,kz)

    mu_grid = np.flip(mu_grid,axis=0)
    ktotal_grid = np.flip(ktotal_grid,axis=0)
    return mu_grid,ktotal_grid

def kaiser(mu,f):
    """Compute the functional influence of the Kaiser effect
    """
    return (1+f*(mu**2))**2

#We need the k_z and mu coordinates for the model, so create a meshgrid and use the mu_grid function

x = np.linspace(0, 0.99, 64)  
y = np.linspace(0, 0.99, 64) 
#Adjust y values to be bottom-up
y = np.flipud(y)
k_trans,k_z = np.meshgrid(x, y)

mu_mall64,_ = mu_grid(mall_div)

#Flip the row order of k_z and mu since mall_div is also flipped
k_z = np.flip(k_z,axis=0)
mu_mall64 = np.flip(mu_mall64,axis=0)

#Compute the theoretical kaiser effect
theoretical_kaiser = kaiser(mu_mall64,f=0.524467)

#Import the computed weights
weights_rsd = np.load('weights_rsd.npy')
weights_div = np.load('weights_div.npy')

def weights_comb(grid,weights,shotnoise,mu):
    """Ad-hoc solution to the weights, unused
    """
    weights_poisson = weights 
    weights_sn = (grid/shotnoise)**(1) * (((1-mu**2))**0.5)
    weights_tot = weights_div
    return weights_tot

weights = np.sqrt(mall_numvalTrue)
power_rsd = np.genfromtxt('/net/draco/data2/vanveenhuyzen/rsd_project/nbodykit_sourcecode/power_spectra/power_rsd_0122.txt')
sn = power_rsd[0,2]

#weights_total = weights_comb(mall_rsd,weights,sn,mu_mall64) #unused

#The current weights used 
weights_total = weights_div

plt.imshow(weights_total,origin='lower',extent=(0,1,0,1))
plt.title('Weights = (w_SN * w_numpoints)**2, using RSD/SN * (1-mu^2)^0.5')
plt.xlabel(r'$k = \sqrt{k_x^2 + k_y^2}$')
plt.ylabel(r'$k_z$')
plt.grid(visible=True)
plt.colorbar()
#plt.show()
plt.close()

def residuals(ydata,model):
    """This function computes the residual between model and data, useful for 
    eyeballing a 'goodness-of-fit'
    """
    diff_sq = (ydata-model)**2
    diff_sq_weighted = weights_total**2 * diff_sq / np.sum(weights_total)
    return diff_sq,diff_sq_weighted

def omega_matter(f):
    #Sargent and Turner (1977), computes a cosmological parameter 
    return f**(1/0.545)

omega_m = 0.306 #Omega_matter of the D3A cosmology we used 
f_true = omega_m**0.545 
print('The true value of Omega_m is {}'.format(omega_m))
print('The true value of f is {:.5f}'.format(f_true))

def lorentz(x,mu,kz,power):
    #The fit parameters: 
    f = x[0]
    sigma_v = x[1]

    #The model of the form P_rsd/P_real = (kaiser) * (Fingers-of-God), where the latter is described by a lorentzian
    model = ((1+(f/1.04)*(mu)**2)**2) * 1/(1 + (kz*sigma_v)**2)

    #Incorporate the weights to (positively) bias the minimization routine 
    diff_sqr_weighted = weights_total**2 * (model-power)**2 / np.sum(weights_total)
    #Return the root-mean square error
    return np.sqrt(np.mean(diff_sqr_weighted))

def model_lorentz(x,mu,kz):
    #Model, used to fill in the fitted parameters to compare with the actual data 
    f = x[0]
    sigma_v = x[1]

    model = ((1+(f/1.04)*(mu)**2)**2) * 1/(1 + (kz*sigma_v)**2)
    return model 

#We use scipy.minimize and more specifically BFGS to minimize this two-dimensional grid 
result_lorentz = minimize(lorentz,[0.5,1],args=(mu_mall64,k_z,mall_div),method='L-BFGS-B',bounds=[(0,1),(0,3)])
print('Standard lorenz model, using free parameters growth factor, velocity dispersion: f, sigma_v',result_lorentz.x)

#Compute the modelled result 
model_lorentz = model_lorentz(result_lorentz.x,mu_mall64,k_z)

model_omega = omega_matter(result_lorentz.x[0])
print('The model finds an Omega_matter of {:.5f}'.format(model_omega))

#Obtain the residuals
res_lorentz,res_weighted_lorentz = residuals(mall_div,model_lorentz)

fig, ax = plt.subplots(2,2,figsize=(10,10))
cax1 = ax[0,0].imshow(mall_div, origin='lower',extent=(0,1,0,1),cmap='nipy_spectral',vmin=0.5,vmax=2.5)
#ax[0,0].set_xlabel(r'$k = \sqrt{k_x^2 + k_y^2}$')
ax[0,0].set_ylabel(r'$k_z$')
ax[0,0].set_title(r'$P_{RSD}/P_{real}$ for all galaxies')
ax[0,0].grid(visible=True)
fig.colorbar(cax1,ax=ax[0,0],label=r'$P_{RSD}(k)/P_{Pos}(k)$')

cax2 = ax[0,1].imshow(model_lorentz,origin='lower',extent=(0,1,0,1),cmap='nipy_spectral',vmin=0.5,vmax=2.5)
#ax[0,1].set_xlabel(r'$k = \sqrt{k_x^2 + k_y^2}$')
ax[0,1].set_ylabel(r'$k_z$')
ax[0,1].set_title(r'$P_{RSD}/P_{real}$ all galaxies using the model')
ax[0,1].grid(visible=True)
fig.colorbar(cax2,ax=ax[0,1],label=r'$P_{RSD}(k)/P_{Pos}(k)$')

cax3 = ax[1,0].imshow(res_lorentz, origin='lower',extent=(0,1,0,1),cmap='nipy_spectral',vmin=0,vmax=2)
ax[1,0].set_xlabel(r'$k = \sqrt{k_x^2 + k_y^2}$')
ax[1,0].set_ylabel(r'$k_z$')
ax[1,0].set_title(r'Residuals')
ax[1,0].grid(visible=True)
fig.colorbar(cax3,ax=ax[1,0],label=r'(data-model)^2')

cax4 = ax[1,1].imshow(res_weighted_lorentz, origin='lower',extent=(0,1,0,1),cmap='nipy_spectral',norm=matplotlib.colors.LogNorm(vmin=5e-6,vmax=1e-3))
ax[1,1].set_xlabel(r'$k = \sqrt{k_x^2 + k_y^2}$')
ax[1,1].set_ylabel(r'$k_z$')
ax[1,1].set_title(r'Weighted residuals')
ax[1,1].grid(visible=True)
fig.colorbar(cax4,ax=ax[1,1],label=r'w^2 * (data-model)^2 / sum(w)')

fig.suptitle('Comparison of true and modelled RSD/Real grid ({})'.format('Lorentzian_Power'))
fig.savefig('fitting/model_lorentz_weightsdiv.png')
#plt.show()
plt.close()
