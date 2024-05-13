import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors

def load_files(file):
    path = '/net/draco/data2/vanveenhuyzen/rsd_project/nbodykit_sourcecode/griddata/'
    m_pos = np.load(path+'grid_pos_{}.npy'.format(file))
    m_rsd = np.load(path+'grid_rsd_{}.npy'.format(file))
    return m_pos,m_rsd,(m_rsd/m_pos)

#Load in the data: 
mall_pos,mall_rsd,mall_div = load_files('mall')

#mall_rsd /= np.max(mall_rsd)

fig, ax = plt.subplots(figsize=(10,10))
cax1 = ax.imshow(mall_pos, origin='lower',extent=(0,1,0,1),cmap='nipy_spectral',norm=matplotlib.colors.LogNorm(vmin=50,vmax=1.1e5))
ax.set_xlabel(r'$k = \sqrt{k_x^2 + k_y^2}$')
ax.set_ylabel(r'$k_z$')
ax.set_title(r'$P_{RSD}/P_{real}$ for all galaxies')
ax.grid(visible=True)
fig.colorbar(cax1,label=r'$P_{RSD}(k)/P_{Pos}(k)$')
plt.savefig('mall_real.png')
#plt.show()
plt.close()

factor = 2
No = 64 
Nn = No//factor

avg = np.zeros((32,32))
weights = np.zeros((64,64))

for i in range(Nn):
    for j in range(Nn):
        #Step 1 
        tl,tr,bl,br = mall_rsd[2*i,2*j],mall_rsd[2*i,2*j+1],mall_rsd[2*i+1,2*j],mall_rsd[2*i+1,2*j+1]
        pixels = np.array([tl,tr,bl,br])
        mean = np.mean(pixels)
    
        #Step 2
        norm_pixels = pixels/mean

        #Step 3 
        norm_mean = np.mean(norm_pixels)

        #Step 4 
        variance = (norm_pixels - norm_mean)**2 
        rms = np.sqrt(variance)
        
        diff_sq = (pixels-mean)**2 
        var = np.mean(diff_sq)
        #print(var)
        var_ = np.sqrt(np.mean(variance))
        print(np.mean(variance))

        avg[i,j] = mean 
        weights[2*i,2*j],weights[2*i,2*j+1],weights[2*i+1,2*j],weights[2*i+1,2*j+1] = 1/rms[0],1/rms[1],1/rms[2],1/rms[3]

fig, ax = plt.subplots(figsize=(10,10))
cax1 = ax.imshow(avg, origin='lower',extent=(0,1,0,1),cmap='nipy_spectral')
ax.set_xlabel(r'$k = \sqrt{k_x^2 + k_y^2}$')
ax.set_ylabel(r'$k_z$')
ax.set_title(r'$P_{RSD}/P_{real}$ for all galaxies')
ax.grid(visible=True)
fig.colorbar(cax1,label=r'$P_{RSD}(k)/P_{Pos}(k)$')
plt.savefig('mall_rsd_32x.png')
#plt.show()
plt.close()

fig, ax = plt.subplots(figsize=(10,10))
cax1 = ax.imshow(weights, origin='lower',extent=(0,1,0,1),cmap='nipy_spectral',vmin=1,vmax=100)
ax.set_xlabel(r'$k = \sqrt{k_x^2 + k_y^2}$')
ax.set_ylabel(r'$k_z$')
ax.set_title(r'Weights as a function of their (k_transversal,k_z)')
ax.grid(visible=True)
fig.colorbar(cax1,label=r'$Value of the weights$')
plt.savefig('weights_normalized_individual_0507.png')
#plt.show()
plt.close()

np.save('fitting/weights_0506.npy',weights)

theoretical_kaiser = np.load('kaiser.npy')
print(theoretical_kaiser)
mall_div_noKaiser = mall_div/theoretical_kaiser

fig, ax = plt.subplots(figsize=(10,10))
cax1 = ax.imshow(mall_div_noKaiser, origin='lower',extent=(0,1,0,1),cmap='nipy_spectral',vmin=0.5,vmax=2.5)
ax.set_xlabel(r'$k = \sqrt{k_x^2 + k_y^2}$')
ax.set_ylabel(r'$k_z$')
ax.set_title(r'$P_{RSD}/P_{real}$ for all galaxies without the Kaiser effect')
ax.grid(visible=True)
fig.colorbar(cax1,label=r'$P_{RSD}(k)/P_{Pos}(k)$')
plt.savefig('mall_div_noKaiser.png')
#plt.show()
plt.close()

mall_rsd_nokaiser = mall_rsd/(theoretical_kaiser*mall_pos)

avg = np.zeros((32,32))
weights = np.zeros((64,64))

for i in range(Nn):
    for j in range(Nn):
        #Step 1 
        tl,tr,bl,br = mall_rsd_nokaiser[2*i,2*j],mall_rsd_nokaiser[2*i,2*j+1],\
                        mall_rsd_nokaiser[2*i+1,2*j],mall_rsd_nokaiser[2*i+1,2*j+1]
        pixels = np.array([tl,tr,bl,br])
        mean = np.mean(pixels)
    
        #Step 2
        norm_pixels = pixels/mean

        #Step 3 
        norm_mean = np.mean(norm_pixels)

        #Step 4 
        variance = (norm_pixels - norm_mean)**2 
        rms = np.sqrt(variance)

        diff_sq = (pixels-mean)**2 
        var = np.mean(diff_sq)
        #print(var)
        var_ = np.sqrt(np.mean(variance))
        print(np.mean(variance))

        avg[i,j] = mean 
        weights[2*i,2*j],weights[2*i,2*j+1],weights[2*i+1,2*j],weights[2*i+1,2*j+1] = 1/var_,1/var_,1/var_,1/var_

fig, ax = plt.subplots(figsize=(10,10))
cax1 = ax.imshow(weights, origin='lower',extent=(0,1,0,1),cmap='nipy_spectral',vmin=1,vmax=100)
ax.set_xlabel(r'$k = \sqrt{k_x^2 + k_y^2}$')
ax.set_ylabel(r'$k_z$')
ax.set_title(r'Weights as a function of their (k_transversal,k_z)')
ax.grid(visible=True)
fig.colorbar(cax1,label=r'$Value of the weights$')
plt.savefig('weights_normalized_nokaiser.png')
#plt.show()
plt.close()

np.save('weights_nokaiser.npy',weights)

#Grab the (k,Pk) coordinates along mu=0 since these this data should have the lowest amount of noise

#print(mall_pos[0,:])
pk_values = mall_pos[0,:]
k_values = np.linspace(0,1,64)

def interp_grid(grid):
    """
    Function to interpolate the (k,Pk) vector along mu=0 (lowest noise) in the whole grid. 
    """

    #Call a new, empty grid to fill 
    interp_grid = np.zeros_like(grid)

    #Get the (k,Pk) coordinates along the mu=0 axis
    pk = grid[0,:]
    k = np.linspace(0,0.99,64)
    k_scaled = np.floor(k*64).astype(int)
    #print(k_scaled)

    kx,ky = np.meshgrid(k,k)
    k_val = np.zeros_like(kx)

    for i in range(64):
        for j in range(64):
            #Compute magnitude
            kmag = np.sqrt(kx[i,j]**2 + ky[i,j]**2)
            kmag_scaled = np.floor(kmag*64).astype(int)
            #print(kmag_scaled)

            #Create a mask and find the corresponding P(k) values
            k_mask = np.where(kmag_scaled == k_scaled)
            #print(k_mask)
            pk_values = pk[k_mask]

            #Check if multiple pk_values have this 
            if len(pk_values) > 1:
                interp_grid[i,j] = np.mean(pk_values)
            elif len(pk_values) == 1:
                interp_grid[i,j] = pk_values
            else:
                interp_grid[i,j] = 0

            #Compare this k magnitude with k-values from (k,Pk)
            #Fill in the corresponding Pk value to the closest k value 

    #We should have a full grid based on one data value 
    return interp_grid

mall_pos_alt = interp_grid(mall_pos)

fig, ax = plt.subplots(figsize=(10,10))
cax1 = ax.imshow(mall_pos_alt, origin='lower',extent=(0,1,0,1),cmap='nipy_spectral',norm=matplotlib.colors.LogNorm(vmin=50,vmax=1.1e5))
ax.set_xlabel(r'$k = \sqrt{k_x^2 + k_y^2}$')
ax.set_ylabel(r'$k_z$')
ax.set_title(r'$P_{RSD}/P_{real}$ for all galaxies')
ax.grid(visible=True)
fig.colorbar(cax1,label=r'$P_{RSD}(k)/P_{Pos}(k)$')
plt.savefig('mall_real_noiseless.png')
#plt.show()
plt.close()

def compute_mu(k_trans,k_z):
    """
    Compute the value of mu = cos(theta) for each grid point, taking theta to be the angle with the k_z axis 
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

kmax = 1 
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

_,ktotal = mu_grid(mall_pos)

plt.imshow(ktotal,extent=(0,1,0,1),vmin=0,vmax=1.5)
plt.colorbar()
#plt.show()
plt.close()

def RSDgrid_1D(k,Pk,gridsize):

    #The gridsize determines how many pixels we will use to construct a grid
    N = gridsize

    #First, lets create a grid of NxN pixels within the range k_x,k_y = 0 to 1 
    x = np.linspace(0,0.99,N,dtype=np.float32)
    x_gridpoints,y_gridpoints = np.meshgrid(x,x)

    #We scale the k values by a factor N so that we can compare integers rather than floats
    k_scaled = np.floor(N*k).astype(int)

    grid = np.zeros((N,N))
    #Construct a nested loop in which we go through all the points
    for i in range(len(x_gridpoints[0])):
        for j in range(len(y_gridpoints[0])):

            #Find the current absolute k value of the grid point 
            k_current = np.floor(N*np.sqrt(x_gridpoints[i,j]**2 + y_gridpoints[i,j]**2)).astype(int)

            #The key step: find the points at which data and grid coincide
            mask = np.where(k_scaled == k_current)
            pk_values = Pk[mask]

            #If a grid values has values assigned to it, continue,
            #else fill with a very low number (empty point) for plotting purposes
            if len(pk_values) > 0:
                grid[i,j] = np.mean(pk_values)
            else:
                grid[i,j] = 1.

    return grid

mall_real_noiseless = RSDgrid_1D(ktotal,mall_pos,64)


fig, ax = plt.subplots(figsize=(10,10))
cax1 = ax.imshow(mall_real_noiseless, origin='lower',extent=(0,1,0,1),cmap='nipy_spectral',norm=matplotlib.colors.LogNorm(vmin=50,vmax=1.1e5))
ax.set_xlabel(r'$k = \sqrt{k_x^2 + k_y^2}$')
ax.set_ylabel(r'$k_z$')
ax.set_title(r'$P_{RSD}/P_{real}$ for all galaxies')
ax.grid(visible=True)
fig.colorbar(cax1,label=r'$P_{RSD}(k)/P_{Pos}(k)$')
plt.savefig('mall_real_noNoise.png')
#plt.show()
plt.close()

#Weights function that uses 3 by 3 pixels:
def variance_grid(data_grid):
    """Compute the variance per pixel by calculating the normalized difference squared
    between the pixel and the mean of the surrounding 3 by 3 grid of pixels (where possible). 

    Parameters:
    ----------  
    data_grid : 2-dimensional numpy array of shape (N,N)
        The grid points for which we want to find the variances relative to their nearest neighbours

    Returns:
    -------
    variance_grid: 2-dimensional numpy array of shape (N,N)
        Grid filled with variances of the data_grid array pixels
    """

    N = np.shape(data_grid)[0] #Get the length of the grid axis
    variance_grid = np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            #Create a section of 3 by 3 pixels around the current pixel m_ij, note this is limited e.g. to 2 by 2 at (0,0)
            rowidx_start = np.max([0,i-1])
            rowidx_stop = np.min([i+2,N])
            colidx_start = np.max([0,j-1])
            colidx_stop = np.min([j+2,N])

            pxl_section = data_grid[rowidx_start:rowidx_stop,colidx_start:colidx_stop]

            #Get the mean of the pixels and normalize them by it 
            pxl_mean = np.mean(pxl_section)
            pxl_section_norm = pxl_section/pxl_mean

            #Again get the now normalized mean of the pixels and compute the square difference 
            pxl_norm_mean = np.mean(pxl_section_norm)
            pxl_section_variance = (pxl_section_norm-pxl_norm_mean)**2

            #Get the variance of the current pixel m_ij only, the conditionals in the slice/brackets ensure
            #we select the correct element! 
            pxl_variance = pxl_section_variance[int(i>0),int(j>0)]
            variance_grid[i,j] = pxl_variance
            
    return variance_grid
#plt.savefig("fig"+("_noweights" if noweights else "")+".png")

grid_variance = variance_grid(mall_rsd)

fig, ax = plt.subplots(figsize=(10,10))
cax1 = ax.imshow(1/np.sqrt(grid_variance), origin='lower',extent=(0,1,0,1),cmap='nipy_spectral',vmin=1,vmax=100)
ax.set_xlabel(r'$k = \sqrt{k_x^2 + k_y^2}$')
ax.set_ylabel(r'$k_z$')
ax.set_title(r'Weights as a function of their (k_transversal,k_z)')
ax.grid(visible=True)
fig.colorbar(cax1,label=r'$Value of the weights$')
plt.savefig('weights_normalized_individual_3by3.png')
#plt.show()
plt.close()

fig, ax = plt.subplots(figsize=(10,10))
cax1 = ax.imshow(grid_variance, origin='lower',extent=(0,1,0,1),cmap='nipy_spectral',vmin=1e-5,vmax=0.1)
ax.set_xlabel(r'$k = \sqrt{k_x^2 + k_y^2}$')
ax.set_ylabel(r'$k_z$')
ax.set_title(r'Variances sigma^2 as a function of their (k_transversal,k_z)')
ax.grid(visible=True)
fig.colorbar(cax1,label=r'$Value of the weights$')
plt.savefig('variances.png')
#plt.show()
plt.close()

def weights_grid(variance,p):
    """Compute the weights per pixel by smoothing over the variances and taking the inverse square root 

    Parameters:
    ----------  
    variance : 2-dimensional numpy array of shape (N,N)
        The variances that we want to smooth using their nearest neighbours
    p: integer
        Size of the smoothing axis (e.g. p=5 means smoothing uses 5 by 5 variance grid)

    Returns:
    -------
    weights_grid: 2-dimensional numpy array of shape (N,N)
        Grid filled with the weights using the smoothed variances of the data_grid array pixels
    """

    N = np.shape(variance)[0] #Get the length of the grid axis

    smoothed_variance = np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            #Section of p by p pixels to smooth the variance over 
            rowidx_start = np.max([0,i-(p-2)])
            rowidx_stop = np.min([i+(p-1),64])
            colidx_start = np.max([0,j-(p-2)])
            colidx_stop = np.min([j+(p-1),64])

            pxl_section = variance[rowidx_start:rowidx_stop,colidx_start:colidx_stop]

            #We compute the mean for the sub-array of pixels to 'smooth' out the variance 
            pxl_mean = np.mean(pxl_section) 
            smoothed_variance[i,j] = pxl_mean #Note, this is still in units of sigma^2, the rms is the sqrt of this! 
    
    return smoothed_variance 

smoothed_variance = weights_grid(grid_variance,p=5)

var_div = variance_grid(mall_div)
weights_div = weights_grid(var_div,p=5)


fig, ax = plt.subplots(figsize=(10,10))
cax1 = ax.imshow(smoothed_variance, origin='lower',extent=(0,1,0,1),cmap='nipy_spectral',vmin=1e-5,vmax=0.1)
ax.set_xlabel(r'$k = \sqrt{k_x^2 + k_y^2}$')
ax.set_ylabel(r'$k_z$')
ax.set_title(r'Smoothed variances sigma^2 as a function of their (k_transversal,k_z)')
ax.grid(visible=True)
fig.colorbar(cax1,label=r'$Value of the weights$')
plt.savefig('smoothed_variances.png')
#plt.show()
plt.close()

weights_smoothed = 1/np.sqrt(smoothed_variance)

fig, ax = plt.subplots(figsize=(10,10))
cax1 = ax.imshow(1/np.sqrt(smoothed_variance), origin='lower',extent=(0,1,0,1),cmap='nipy_spectral',vmin=1,vmax=100)
ax.set_xlabel(r'$k = \sqrt{k_x^2 + k_y^2}$')
ax.set_ylabel(r'$k_z$')
ax.set_title(r'Smoothed weights w=1/sigma as a function of their (k_transversal,k_z)')
ax.grid(visible=True)
fig.colorbar(cax1,label=r'$Value of the weights$')
plt.savefig('smoothed_weights.png')
#plt.show()
plt.close()

np.save('weights_smooth.npy',weights_smoothed)
np.save('weights_malldiv.npy',1/np.sqrt(weights_div))

fig, ax = plt.subplots(figsize=(10,10))
cax1 = ax.imshow(1/np.sqrt(weights_div), origin='lower',extent=(0,1,0,1),cmap='nipy_spectral',vmin=1,vmax=100)
ax.set_xlabel(r'$k = \sqrt{k_x^2 + k_y^2}$')
ax.set_ylabel(r'$k_z$')
ax.set_title(r'Smoothed weights w=1/sigma as a function of their (k_transversal,k_z) using mall_div')
ax.grid(visible=True)
fig.colorbar(cax1,label=r'$Value of the weights$')
plt.savefig('smoothed_weights_malldiv.png')
#plt.show()
plt.close()

