import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
from scipy.optimize import curve_fit
from scipy.optimize import least_squares
from scipy.optimize import minimize
from scipy.stats import chisquare

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
mall_pos,mall_rsd,mall_div = load_files('mall')

mall_numval = np.load('/net/draco/data2/vanveenhuyzen/rsd_project/nbodykit_sourcecode/griddata/number_of_values/numval_mall.npy')

plt.imshow(mall_numval)
plt.colorbar()
#plt.show()
plt.close()

#Seems we have some weird artefact so lets remove it
mall_numvalTrue = np.ones_like(mall_numval) * mall_numval[-1,:]

plt.imshow(mall_numvalTrue)
plt.colorbar()
plt.show()
plt.close()


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
#print(mall16_div.shape)

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
    return f[0],f_std[0,0], mu_, ktotal_

f,ferr,mu16,ktotal16 = compute_growthfactor(mall16_div,0.25,'test')

mall16_div_lowk = mall16_div[:4,:4]
#print(mall16_div_lowk.shape)

mall16_div_highk = mall16_div[:,int(0.85*16):]
#print(mall16_div_highk)

mall_highk = np.mean(mall16_div_highk,axis=1)
#print(mall_highk)

k_values = np.linspace(0,1,16)
for i in range(mall16_div_highk.shape[1]):
    plt.plot(k_values,mall16_div_highk[:,i],linestyle='dashed',alpha=0.8)
plt.plot(k_values,mall_highk,color='black')
plt.savefig('fog_18032024.png')
#plt.show()
plt.close()

fig, ax = plt.subplots(figsize=(8, 6))
ax.set_xlabel(r'$k = \sqrt{k_x^2 + k_y^2}$')
ax.set_ylabel(r'$k_z$')
ax.set_title(r'$P_{RSD}/P_{real}$ for second highest Log mass bin, no SN, Npix=64')
ax.grid(visible=True)
cax = ax.imshow(fullGrid(mall16_div_lowk), origin='lower',extent=(-0.25,0.25,-0.25,0.25),cmap='nipy_spectral',vmin=0,vmax=5)
fig.colorbar(cax,label=r'$P_{RSD}(k)/P_{Pos}(k)$')
plt.savefig('mall_lowk.png')
#plt.show()
plt.close()

#Plot the F.o.G. effect for non-high k 
mall_lowk = np.mean(mall16_div[:,0:int(0.25*16)],axis=1)
mall_midk = np.mean(mall16_div[:,int(0.25*16):int(0.5*16)],axis=1)
mall_midhighk = np.mean(mall16_div[:,int(0.5*16):int(0.85*16)],axis=1)

plt.plot(k_values,mall_lowk,label='k=0 to k=0.25')
plt.plot(k_values,mall_midk,label='k=0.25 to k=0.5')
plt.plot(k_values,mall_midhighk,label='k=0.5 to k=0.85')
plt.plot(k_values,mall_highk,color='black',label='k=0.85 to k=1')
plt.xlabel(r'$k_z$')
plt.ylabel(r'$P_{RSD}/P_{Real}$')
plt.title('The value of the divided spectrum at various sqrt(kx^2+ky^2)')
plt.grid()
plt.legend()
plt.savefig('fog_otherk_25032024.png')
#plt.show()
plt.close()

print(30*'-','Testing mu:')
mu16_lowk = np.mean(mu16[:,0:int(0.25*16)],axis=1)
mu16_midk = np.mean(mu16[:,int(0.25*16):int(0.5*16)],axis=1)
mu16_midhighk = np.mean(mu16[:,int(0.5*16):int(0.85*16)],axis=1)
mu16_highk = np.mean(mu16[:,int(0.85*16):],axis=1)
#print(mu16[:,0:int(0.25*16)])
#print(mu16_lowk)

#print(mu16.shape)

kaiser_lowk = (1 + f*mu16_lowk**2)**2
kaiser_midk = (1 + f*mu16_midk**2)**2
kaiser_midhighk = (1 + f*mu16_midhighk**2)**2
kaiser_highk = (1 + f*mu16_highk**2)**2

plt.plot(k_values,mall_lowk/kaiser_lowk,label='k=0 to k=0.25')
plt.plot(k_values,mall_midk/kaiser_midk,label='k=0.25 to k=0.5')
plt.plot(k_values,mall_midhighk/kaiser_midhighk,label='k=0.5 to k=0.85')
plt.plot(k_values,mall_highk/kaiser_highk,color='black',label='k=0.85 to k=1')
plt.xlabel(r'$k_z$')
plt.ylabel(r'$P_{RSD}/P_{Real}$')
plt.title('The value of the divided spectrum without Kaiser effect')
plt.grid()
plt.legend()
plt.savefig('fog_otherk_noKaiser_25032024.png')
#plt.show()
plt.close()


fig, ax = plt.subplots(figsize=(8, 6))
ax.set_xlabel(r'$k = \sqrt{k_x^2 + k_y^2}$')
ax.set_ylabel(r'$k_z$')
ax.set_title(r'$P_{RSD}/P_{real}$ for second highest Log mass bin, no SN, Npix=64')
ax.grid(visible=True)
cax = ax.imshow(fullGrid(mall16_div), origin='lower',extent=(-1,1,-1,1),cmap='nipy_spectral',vmin=0,vmax=5)
fig.colorbar(cax,label=r'$P_{RSD}(k)/P_{Pos}(k)$')
plt.savefig('mall_N16_18032024.png')
#plt.show()
plt.close()

fig, ax = plt.subplots(figsize=(8, 6))
ax.set_xlabel(r'$k = \sqrt{k_x^2 + k_y^2}$')
ax.set_ylabel(r'$k_z$')
ax.set_title(r'$P_{RSD}/P_{real}$ for second highest Log mass bin, no SN, Npix=64')
ax.grid(visible=True)
cax = ax.imshow(mall16_div_highk, origin='lower',extent=(0.75,1,0,1),cmap='nipy_spectral',vmin=0.5,vmax=1.05)
fig.colorbar(cax,label=r'$P_{RSD}(k)/P_{Pos}(k)$')
plt.savefig('mall_highk.png')
#plt.show()
plt.close()

mu_mall,ktotal_mall = mu_grid(mall16_div)

ktotal_flat = ktotal_mall.flatten()

#print(ktotal_mall.flatten())

def rsd_fit(mu,f,a,b,c):
    """
    Product of the Kaiser and F.o.G. effects in the RSD plot, note that we multiply mu by ktotal_mall
    """
    return ((1+f*(mu**2))**2) * (-a*(mu*ktotal_flat)**2 + b*(mu*ktotal_flat) + c)

def rsd_fit_v2(mu,f,b,c):
    """
    Product of the Kaiser and F.o.G. effects in the RSD plot, note that we multiply mu by ktotal_mall
    """
    return ((1+f*(mu**2))**2) * (b*(mu*ktotal_flat) + c)

def rsd_fit_v3(mu,f,a,b,x0,sigma):
    """
    Product of the Kaiser and F.o.G. effects in the RSD plot, note that we multiply mu by ktotal_mall
    """
    return ((1+f*(mu**2))**2) * (a * np.exp(-((mu*ktotal_flat)-x0)**2/(2*sigma**2)) + b)


params,pcov = curve_fit(rsd_fit, mu_mall.flatten(), mall16_div.flatten(),p0=[0.26,0.1,0.5,1])
print(u'The fitted parameters to quantify the Kaiser effect for {} galaxies is: {} \u00B1 {}'.format('all',params,pcov[0]))

params,pcov = curve_fit(rsd_fit_v2, mu_mall.flatten(), mall16_div.flatten(),p0=[0.26,0.5,1])
print(u'The fitted parameters to quantify the Kaiser effect for {} galaxies is: {} \u00B1 {}'.format('all',params,pcov[0]))

params,pcov = curve_fit(rsd_fit_v3, mu_mall.flatten(), mall16_div.flatten(),p0=[0.26,1,1,0,1])
print(u'The fitted parameters to quantify the Kaiser effect for {} galaxies is: {} \u00B1 {}'.format('all',params,pcov[0]))

#Try fitting a function: the product of the Kaiser formula and a Gaussian 

def func(xdata,f,b,x0,sigma):
    """
    Product of the Kaiser and F.o.G. effects in the RSD plot, note that we multiply mu by ktotal_mall
    """
    mu,kz = xdata

    return ((1+f*(mu**2))**2) * (1/np.sqrt(2*np.pi*sigma**2) * np.exp(-(kz-x0)**2/(2*sigma**2)) + b)

# Create meshgrid
x = np.linspace(0, 1, 64)  # X coordinates range from 0 to 1
y = np.linspace(0, 1, 64)  # Y coordinates range from 0 to 1

# Adjust y values to be bottom-up
y = np.flipud(y)

# Create meshgrid
X, Y = np.meshgrid(x, y)

print(Y)

mu_mall64,_ = mu_grid(mall_div)

numval_max = np.max(mall_numvalTrue)
numval_norm = mall_numvalTrue/numval_max

func_var = np.vstack((mu_mall64.ravel(), Y.ravel()))
func_values = mall_div.ravel()
func_sigma = numval_norm.ravel()

print(mu_mall64.ravel())
print(func_values)

par,cov = curve_fit(func,func_var,func_values,sigma=func_sigma,p0=[0.25,0.5,0.,1.],bounds=([0.2,0.5,-0.1,0],[0.3,1,0.1,10]))

print('gaussian params and covariance:',par,np.diag(cov))

fit_result = func(func_var,*par).reshape(64,64)

fig, ax = plt.subplots(figsize=(8, 6))
cax = ax.imshow(mall_div, origin='lower',extent=(0,1,0,1),cmap='nipy_spectral',vmin=0.5,vmax=1.5)
ax.set_xlabel(r'$k = \sqrt{k_x^2 + k_y^2}$')
ax.set_ylabel(r'$k_z$')
ax.set_title(r'$P_{RSD}/P_{real}$ for all galaxies')
ax.grid(visible=True)
fig.colorbar(cax,label=r'$P_{RSD}(k)/P_{Pos}(k)$')
fig.savefig('fitting/nofit_BFGS.png')
#plt.show()
plt.close()

fig, ax = plt.subplots(figsize=(8, 6))
cax = ax.imshow(fit_result, origin='lower',extent=(0,1,0,1),cmap='nipy_spectral',vmin=0,vmax=5)
fig.colorbar(cax,label=r'$P_{RSD}(k)/P_{Pos}(k)$')
#fig.savefig('fitting/fit.png')
#plt.show()
plt.close()

print('N=64',mall_numvalTrue[-1,:])
weights = np.sqrt(mall_numvalTrue)

print('N=64 weights',weights[-1,:])

def func_minimize(x,mu,kz,power):
    f = x[0]
    a = x[1]
    kz0 = x[2]
    sigma = x[3]

    model = ((1+f*(mu**2))**2) * (1/np.sqrt(2*np.pi*sigma**2) * np.exp(-(kz-kz0)**2/(2*sigma**2)) + a)
    #Now, compute the sum-squared error between the model and divided power 
    diff_squared = (model-power)**2 
    return np.sqrt(np.mean(diff_squared))
    return np.sum(diff_squared)

result = least_squares(func_minimize,[0.25,1,0,1],args=(mu_mall64,Y,mall_div))

print('Using least-squares')
print(result.x)

result_alt = minimize(func_minimize,[0.25,1,0,1],args=(mu_mall64,Y,mall_div),method='BFGS')

print('Using BFGS minimization')
print(result_alt.x)

print(50*'-'+'Relevant fitting starts here:'+50*'-')

def func_gaussian(x,mu,kz,power):
    f = x[0]
    sigma = x[1]
    #b = x[2]
    #sigma = x[2]

    model = ((1+f*(mu**2))**2) * (np.exp(-(kz)**2/(2*sigma**2)))
    #Now, compute the sum-squared error between the model and divided power 
    diff_squared = (model-power)**2 
    #As an extra step, apply the weights computed beforehand: 
    diff_sqr_weighted = weights**2 * (model-power)**2 * (1/np.sum(weights))

    #Finally, compute the Root-Mean-Square Error (RMSE), which is what we will try to minimize! 
    return np.sqrt(np.mean(diff_sqr_weighted))

def func_lorentzian(x,mu,kz,power):
    f = x[0]
    gamma = x[1]

    model = ((1+f*(mu**2))**2) * (gamma/((gamma)**2 + (kz)**2))
    #As an extra step, apply the weights computed beforehand: 
    diff_sqr_weighted = weights * (model-power)**2 * (1/np.sum(weights))

    #Finally, compute the Root-Mean-Square Error (RMSE), which is what we will try to minimize! 
    return np.sqrt(np.mean(diff_sqr_weighted))

def func_lorentzian_basic(x,mu,kz,power):
    f = x[0]
    t = kz * f * 1 

    model = ((1+f*(mu**2))**2) * 1/(1 + t**2)

    diff_sqr_weighted = weights * (model-power)**2 * (1/np.sum(weights))

    #Finally, compute the Root-Mean-Square Error (RMSE), which is what we will try to minimize! 
    return np.sqrt(np.mean(diff_sqr_weighted))

def func_lorentzian_basic2(x,mu,kz,power):
    f = x[0]
    sigma = x[1]
    t = kz * f * sigma

    model = ((1+f*(mu**2))**2) * 1/(1 + t**2)

    diff_sqr_weighted = weights * (model-power)**2 * (1/np.sum(weights))

    #Finally, compute the Root-Mean-Square Error (RMSE), which is what we will try to minimize! 
    return np.sqrt(np.mean(diff_sqr_weighted))

def func_gaussian_basic(x,mu,kz,power):
    f = x[0]
    t = kz * f

    model = ((1+f*(mu**2))**2) * np.exp(-t**2)

    diff_sqr_weighted = weights * (model-power)**2 * (1/np.sum(weights))

    #Finally, compute the Root-Mean-Square Error (RMSE), which is what we will try to minimize! 
    return np.sqrt(np.mean(diff_sqr_weighted))


def func_gaussian_basic2(x,mu,kz,power):
    f = x[0]
    sigma = x[1]
    t = kz * f * sigma

    model = ((1+f*(mu**2))**2) * np.exp(-t**2)

    diff_sqr_weighted = weights * (model-power)**2 * (1/np.sum(weights))

    #Finally, compute the Root-Mean-Square Error (RMSE), which is what we will try to minimize! 
    return np.sqrt(np.mean(diff_sqr_weighted))

#Use L-BFGS-B such that we can use bounds
result_gaussian = minimize(func_gaussian,[0.5,1],args=(mu_mall64,Y,mall_div),method='L-BFGS-B',bounds=((0,1),(-10,10)))

print('Using L-BFGS-B minimization and a Gaussian profile')
print(result_gaussian.x)

result_lorentzian = minimize(func_lorentzian,[0.5,1],args=(mu_mall64,Y,mall_div),method='L-BFGS-B',bounds=((0,1),(0,10)))

print('Using L-BFGS-B minimization and a Lorentzian profile')
print(result_lorentzian.x)

result_lorentzian_basic = minimize(func_lorentzian_basic,0.5,args=(mu_mall64,Y,mall_div),method='L-BFGS-B',bounds=[(0,1)])

print('Using L-BFGS-B minimization and a 1/(1+x**2) Lorentzian profile')
print(result_lorentzian_basic.x)

result_lorentzian_basic2 = minimize(func_lorentzian_basic2,[0.5,1],args=(mu_mall64,Y,mall_div),method='L-BFGS-B',bounds=[(0,1),(0,10)])

print('Using L-BFGS-B minimization and a 1/(1+x**2) Lorentzian profile and sigma_v as free parameter')
print(result_lorentzian_basic2.x)

result_gaussian_basic = minimize(func_gaussian_basic,0.5,args=(mu_mall64,Y,mall_div),method='L-BFGS-B',bounds=[(0,1)])

print('Using L-BFGS-B minimization and a np.exp(-x**2) Gaussian profile')
print(result_gaussian_basic.x)

result_gaussian_basic2 = minimize(func_gaussian_basic2,[0.5,1],args=(mu_mall64,Y,mall_div),method='L-BFGS-B',bounds=[(0,1),(0,10)])

print('Using L-BFGS-B minimization and a np.exp(-x**2) Gaussian profile and sigma_v as free parameter')
print(result_gaussian_basic2.x)

#result_NM = minimize(func_minimize2,[0.25,1],args=(mu_mall64,Y,mall_div),method='Nelder-Mead')

#print('Using Nelder-Mead minimization')
#print(result_NM.x)

def model(x,mu,kz):
    f = x[0]
    #f = 0.25
    #a = x[1]
    sigma = x[1]
    #sigma = x[2]
    #b = x[2]
    model = ((1+f*(mu**2))**2) * (np.exp(-(kz)**2/(2*sigma**2)))
    return model

def model_lorenztian(x,mu,kz):
    f = x[0]
    gamma = x[1]
    model = ((1+f*(mu**2))**2) * (gamma**2/((gamma)**2 + (kz)**2))
    return model

def model_lorentzian_basic(x,mu,kz):
    f = x[0]
    t = kz * f * 1 

    model = ((1+f*(mu**2))**2) * 1/(1 + t**2)
    return model

#Print statements to check whether the shapes of Y (kz) and mu are correctly oriented! 
print(mu_mall64)
print(Y)

mall_div_model = model(result_gaussian.x,mu_mall64,Y)
mall_div_lorentzian = model_lorenztian(result_lorentzian.x,mu_mall64,Y)
mall_div_lorentzian_basic = model_lorentzian_basic(result_lorentzian_basic.x,mu_mall64,Y)

print(40*'-'+'Chi-square test')
chi2_gaussian = chisquare(f_obs=mall_div.flatten(),f_exp=mall_div_model.flatten())
chi2_lorentzian = chisquare(f_obs=mall_div.flatten(),f_exp=mall_div_lorentzian.flatten())

#Compute the degrees of freedom: N*N - 2 
dof = (len(mall_div)-2)**2
print(chi2_gaussian.statistic/dof)
print(chi2_lorentzian.statistic/dof)

test = chisquare(mall_div.flatten(),mall_div.flatten())
print(test.statistic)

fig, ax = plt.subplots(figsize=(8, 6))
cax = ax.imshow(mall_div_model, extent=(0,1,0,1),cmap='nipy_spectral',vmin=0.5,vmax=1.5)
ax.set_xlabel(r'$k = \sqrt{k_x^2 + k_y^2}$')
ax.set_ylabel(r'$k_z$')
ax.set_title(r'$P_{RSD}/P_{real}$ all galaxies using the model')
ax.grid(visible=True)
fig.colorbar(cax,label=r'$P_{RSD}(k)/P_{Pos}(k)$')
fig.savefig('fitting/fit_BFGS.png')
#plt.show()
plt.close()

fig, ax = plt.subplots(figsize=(8, 6))
cax = ax.imshow(mall_div_lorentzian, extent=(0,1,0,1),cmap='nipy_spectral',vmin=0.5,vmax=1.5)
ax.set_xlabel(r'$k = \sqrt{k_x^2 + k_y^2}$')
ax.set_ylabel(r'$k_z$')
ax.set_title(r'$P_{RSD}/P_{real}$ all galaxies using the Lorentzian model')
ax.grid(visible=True)
fig.colorbar(cax,label=r'$P_{RSD}(k)/P_{Pos}(k)$')
fig.savefig('fitting/fit_Lorentzian.png')
#plt.show()
plt.close()

fig, ax = plt.subplots(figsize=(8, 6))
cax = ax.imshow(fullGrid(mall_div), extent=(0,1,0,1),cmap='nipy_spectral',vmin=0,vmax=1.5)
ax.set_xlabel(r'$k = \sqrt{k_x^2 + k_y^2}$')
ax.set_ylabel(r'$k_z$')
ax.set_title(r'$P_{RSD}/P_{real}$ for all galaxies (full)')
ax.grid(visible=True)
fig.colorbar(cax,label=r'$P_{RSD}(k)/P_{Pos}(k)$')
fig.savefig('fitting/nofit_BFGS_full.png')
#plt.show()
plt.close()

fig, ax = plt.subplots(figsize=(8, 6))
cax = ax.imshow(fullGrid(mall_div_model), extent=(0,1,0,1),cmap='nipy_spectral',vmin=0,vmax=1.5)
ax.set_xlabel(r'$k = \sqrt{k_x^2 + k_y^2}$')
ax.set_ylabel(r'$k_z$')
ax.set_title(r'$P_{RSD}/P_{real}$ for all galaxies using the model (full)')
ax.grid(visible=True)
fig.colorbar(cax,label=r'$P_{RSD}(k)/P_{Pos}(k)$')
fig.savefig('fitting/fit_BFGS_full.png')
#plt.show()
plt.close()

#Try to relate sigma to the velocity dispersions of galaxies
mall_disp = np.load('/net/draco/data2/vanveenhuyzen/rsd_project/particle_data/galaxy_filterDisp.npy')
mall_disp_z = mall_disp[:,2]
print(np.mean(mall_disp_z))

def residual(ydata,model):
    diff_sq = (ydata-model)**2
    diff_sq_weighted = weights**2 * diff_sq / np.sum(weights)
    return diff_sq,diff_sq_weighted

res,res_weighted = residual(mall_div,mall_div_lorentzian)

fig, ax = plt.subplots(2,2,figsize=(10,10))
cax1 = ax[0,0].imshow(mall_div, origin='lower',extent=(0,1,0,1),cmap='nipy_spectral',vmin=0.5,vmax=1.5)
#ax[0,0].set_xlabel(r'$k = \sqrt{k_x^2 + k_y^2}$')
ax[0,0].set_ylabel(r'$k_z$')
ax[0,0].set_title(r'$P_{RSD}/P_{real}$ for all galaxies')
ax[0,0].grid(visible=True)
fig.colorbar(cax1,ax=ax[0,0],label=r'$P_{RSD}(k)/P_{Pos}(k)$')

cax2 = ax[0,1].imshow(mall_div_lorentzian,extent=(0,1,0,1),cmap='nipy_spectral',vmin=0.5,vmax=1.5)
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

cax4 = ax[1,1].imshow(res_weighted, origin='lower',extent=(0,1,0,1),cmap='nipy_spectral',vmin=0,vmax=0.005)
ax[1,1].set_xlabel(r'$k = \sqrt{k_x^2 + k_y^2}$')
ax[1,1].set_ylabel(r'$k_z$')
ax[1,1].set_title(r'Weighted residuals')
ax[1,1].grid(visible=True)
fig.colorbar(cax4,ax=ax[1,1],label=r'w^2 * (data-model)^2 / sum(w)')

fig.suptitle('Comparison of true and modelled RSD/Real grid (Lorentzian)')
fig.savefig('fitting/lorentzian.png')
plt.show()
plt.close()

#Now try it for the Gaussian

res_gaussian,res_weighted_gaussian = residual(mall_div,mall_div_model)

fig, ax = plt.subplots(2,2,figsize=(10,10))
cax1 = ax[0,0].imshow(mall_div, origin='lower',extent=(0,1,0,1),cmap='nipy_spectral',vmin=0.5,vmax=1.5)
#ax[0,0].set_xlabel(r'$k = \sqrt{k_x^2 + k_y^2}$')
ax[0,0].set_ylabel(r'$k_z$')
ax[0,0].set_title(r'$P_{RSD}/P_{real}$ for all galaxies')
ax[0,0].grid(visible=True)
fig.colorbar(cax1,ax=ax[0,0],label=r'$P_{RSD}(k)/P_{Pos}(k)$')

cax2 = ax[0,1].imshow(mall_div_model,extent=(0,1,0,1),cmap='nipy_spectral',vmin=0.5,vmax=1.5)
#ax[0,1].set_xlabel(r'$k = \sqrt{k_x^2 + k_y^2}$')
ax[0,1].set_ylabel(r'$k_z$')
ax[0,1].set_title(r'$P_{RSD}/P_{real}$ all galaxies using the model')
ax[0,1].grid(visible=True)
fig.colorbar(cax2,ax=ax[0,1],label=r'$P_{RSD}(k)/P_{Pos}(k)$')

cax3 = ax[1,0].imshow(res_gaussian, origin='lower',extent=(0,1,0,1),cmap='nipy_spectral',vmin=0,vmax=2)
ax[1,0].set_xlabel(r'$k = \sqrt{k_x^2 + k_y^2}$')
ax[1,0].set_ylabel(r'$k_z$')
ax[1,0].set_title(r'Residuals')
ax[1,0].grid(visible=True)
fig.colorbar(cax3,ax=ax[1,0],label=r'(data-model)^2')

cax4 = ax[1,1].imshow(res_weighted_gaussian, origin='lower',extent=(0,1,0,1),cmap='nipy_spectral',vmin=0,vmax=0.005)
ax[1,1].set_xlabel(r'$k = \sqrt{k_x^2 + k_y^2}$')
ax[1,1].set_ylabel(r'$k_z$')
ax[1,1].set_title(r'Weighted residuals')
ax[1,1].grid(visible=True)
fig.colorbar(cax4,ax=ax[1,1],label=r'w^2 * (data-model)^2 / sum(w)')

fig.suptitle('Comparison of true and modelled RSD/Real grid (Gaussian)')
fig.savefig('fitting/gaussian.png')

#plt.tight_layout()
plt.show()
plt.close()

fig, ax = plt.subplots(2,2,figsize=(10,10))
cax1 = ax[0,0].imshow(mall_div, origin='lower',extent=(0,1,0,1),cmap='nipy_spectral',vmin=0.5,vmax=1.5)
#ax[0,0].set_xlabel(r'$k = \sqrt{k_x^2 + k_y^2}$')
ax[0,0].set_ylabel(r'$k_z$')
ax[0,0].set_title(r'$P_{RSD}/P_{real}$ for all galaxies')
ax[0,0].grid(visible=True)
fig.colorbar(cax1,ax=ax[0,0],label=r'$P_{RSD}(k)/P_{Pos}(k)$')

cax2 = ax[0,1].imshow(mall_div_lorentzian_basic,extent=(0,1,0,1),cmap='nipy_spectral',vmin=0.5,vmax=1.5)
#ax[0,1].set_xlabel(r'$k = \sqrt{k_x^2 + k_y^2}$')
ax[0,1].set_ylabel(r'$k_z$')
ax[0,1].set_title(r'$P_{RSD}/P_{real}$ all galaxies using the model')
ax[0,1].grid(visible=True)
fig.colorbar(cax2,ax=ax[0,1],label=r'$P_{RSD}(k)/P_{Pos}(k)$')

cax3 = ax[1,0].imshow(res_gaussian, origin='lower',extent=(0,1,0,1),cmap='nipy_spectral',vmin=0,vmax=2)
ax[1,0].set_xlabel(r'$k = \sqrt{k_x^2 + k_y^2}$')
ax[1,0].set_ylabel(r'$k_z$')
ax[1,0].set_title(r'Residuals')
ax[1,0].grid(visible=True)
fig.colorbar(cax3,ax=ax[1,0],label=r'(data-model)^2')

cax4 = ax[1,1].imshow(res_weighted_gaussian, origin='lower',extent=(0,1,0,1),cmap='nipy_spectral',vmin=0,vmax=0.005)
ax[1,1].set_xlabel(r'$k = \sqrt{k_x^2 + k_y^2}$')
ax[1,1].set_ylabel(r'$k_z$')
ax[1,1].set_title(r'Weighted residuals')
ax[1,1].grid(visible=True)
fig.colorbar(cax4,ax=ax[1,1],label=r'w^2 * (data-model)^2 / sum(w)')

fig.suptitle('Comparison of true and modelled RSD/Real grid (Gaussian)')
fig.savefig('fitting/lorentzian_basic.png')

#plt.tight_layout()
plt.show()
plt.close()
