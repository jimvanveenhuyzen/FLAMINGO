import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
from scipy.optimize import curve_fit

def load_files(file):
    path = '/net/draco/data2/vanveenhuyzen/rsd_project/nbodykit_sourcecode/griddata/'
    m_pos = np.load(path+'grid_pos_{}.npy'.format(file))
    m_rsd = np.load(path+'grid_rsd_{}.npy'.format(file))
    return m_pos,m_rsd,(m_rsd/m_pos)

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

def mu_grid(grid):
    
    N = np.shape(grid)[0]

    kmax = 1
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

#f,ferr,mu16,ktotal16 = compute_growthfactor(mall16_div,0.25,'test')

#Load in the data: 
mall16_pos,mall16_rsd,mall16_div = load_files('mall_N16')
mall_pos,mall_rsd,mall_div = load_files('mall')

mall_numval = np.load('/net/draco/data2/vanveenhuyzen/rsd_project/nbodykit_sourcecode/griddata/number_of_values/numval_mall.npy')
mall_numval16 = np.load('/net/draco/data2/vanveenhuyzen/rsd_project/nbodykit_sourcecode/griddata/number_of_values/numval_mall16.npy')
mall_numval32 = np.load('/net/draco/data2/vanveenhuyzen/rsd_project/nbodykit_sourcecode/griddata/number_of_values/numval_mall32.npy')

plt.imshow(mall_numval16)
plt.colorbar()
plt.show()
plt.close()

plt.imshow(mall_numval32)
plt.colorbar()
plt.show()
plt.close()

plt.imshow(mall_numval)
plt.colorbar()
plt.show()
plt.close()

#Seems we have some weird artefact so lets remove it
mall_numvalTrue = np.ones_like(mall_numval) * mall_numval[-1,:]

plt.imshow(mall_numvalTrue)
plt.colorbar()
plt.show()
plt.close()

print("The amount of points per pixel point as a func of k_perpendicular per pixel size is:")
print('N=16',mall_numval16[-1,:])
print('N=32',mall_numval32[-1,:])
print('N=64',mall_numvalTrue[-1,:])

mall_lowk = np.mean(mall_div[:,0:int(0.25*64)],axis=1)
mall_midk = np.mean(mall_div[:,int(0.25*64):int(0.5*64)],axis=1)
mall_midhighk = np.mean(mall_div[:,int(0.5*64):int(0.85*64)],axis=1)
mall_highk = np.mean(mall_div[:,int(0.85*64):],axis=1)

k_values = np.linspace(0,0.99,64)
plt.plot(k_values,mall_lowk,label='k=0 to k=0.25')
plt.plot(k_values,mall_midk,label='k=0.25 to k=0.5')
plt.plot(k_values,mall_midhighk,label='k=0.5 to k=0.85')
plt.plot(k_values,mall_highk,color='black',label='k=0.85 to k=1')
plt.xlabel(r'$k_z$')
plt.ylabel(r'$P_{RSD}/P_{Real}$')
plt.title('The value of the divided spectrum at various sqrt(kx^2+ky^2)')
plt.grid()
plt.legend()
plt.savefig('fog_Mall_comparek_25032024.png')
#plt.show()
plt.close()

#Now get the Kaiser effect and mu values: 
f,ferr,mu,ktotal = compute_growthfactor(mall_div,0.25,'all masses')

mu_lowk = np.mean(mu[:,0:int(0.25*64)],axis=1)
mu_midk = np.mean(mu[:,int(0.25*64):int(0.5*64)],axis=1)
mu_midhighk = np.mean(mu[:,int(0.5*64):int(0.85*64)],axis=1)
mu_highk = np.mean(mu[:,int(0.85*64):],axis=1)

