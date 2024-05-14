import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors

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

fig, ax = plt.subplots(figsize=(10,10))
cax1 = ax.imshow(mall_pos, origin='lower',extent=(0,1,0,1),cmap='nipy_spectral',norm=matplotlib.colors.LogNorm(vmin=50,vmax=1.1e5))
ax.set_xlabel(r'$k = \sqrt{k_x^2 + k_y^2}$')
ax.set_ylabel(r'$k_z$')
ax.set_title(r'$P_{real}$ for all galaxies')
ax.grid(visible=True)
fig.colorbar(cax1,label=r'$P_{RSD}(k)/P_{Pos}(k)$')
plt.savefig('mall_real.png')
#plt.show()
plt.close()

#unused
theoretical_kaiser = np.load('kaiser.npy')
mall_rsd_nokaiser = mall_rsd/(theoretical_kaiser*mall_pos)

#Weights function that uses 3 by 3 pixels:
def compute_variance(data_grid):
    """Compute the variance per pixel by calculating the normalized difference squared
    between the pixel and the mean of the surrounding 3 by 3 grid of pixels (where possible). 

    Parameters:
    ----------  
    data_grid : 2-dimensional numpy array of shape (N,N)
        The grid points for which we want to find the variances relative to their nearest neighbours

    Returns:
    -------
    variance_grid : 2-dimensional numpy array of shape (N,N)
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

def smooth_variance(variance,p):
    """Compute the weights per pixel by smoothing over the variances and taking the inverse square root 

    Parameters:
    ----------  
    variance : 2-dimensional numpy array of shape (N,N)
        The variances that we want to smooth using their nearest neighbours
    p: integer
        Size of the smoothing axis (e.g. p=5 means smoothing uses 5 by 5 variance grid)

    Returns:
    -------
    weights_grid : 2-dimensional numpy array of shape (N,N)
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

#Use the functions defined above to compute the weights for both the RSD and divided grid, using the fact that weights = 1/sqrt(variance)
variance_rsd = compute_variance(mall_rsd)
variance_rsd_smooth = smooth_variance(variance_rsd,p=5)
weights_rsd = 1/np.sqrt(variance_rsd_smooth)

variance_div = compute_variance(mall_div)
variance_div_smooth = smooth_variance(variance_div,p=5)
weights_div = 1/np.sqrt(variance_div_smooth)

#Create horizontal subplots for both weight/variance versions
fig, ax = plt.subplots(1,3,figsize=(20,10))
cax1 = ax[0].imshow(variance_rsd, origin='lower',extent=(0,1,0,1),cmap='nipy_spectral',vmin=1e-5,vmax=0.1)
ax[0].set_xlabel(r'$k = \sqrt{k_x^2 + k_y^2}$')
ax[0].set_ylabel(r'$k_z$')
ax[0].set_title(r'Variance $\sigma^2$')
ax[0].grid(visible=True)
fig.colorbar(cax1,ax=ax[0])

cax2 = ax[1].imshow(variance_rsd_smooth,origin='lower',extent=(0,1,0,1),cmap='nipy_spectral',vmin=1e-5,vmax=0.1)
ax[1].set_xlabel(r'$k = \sqrt{k_x^2 + k_y^2}$')
ax[1].set_ylabel(r'$k_z$')
ax[1].set_title(r'Smoothed variance $\sigma^2$')
ax[1].grid(visible=True)
fig.colorbar(cax2,ax=ax[1])

cax3 = ax[2].imshow(weights_rsd, origin='lower',extent=(0,1,0,1),cmap='nipy_spectral',vmin=0,vmax=100)
ax[2].set_xlabel(r'$k = \sqrt{k_x^2 + k_y^2}$')
ax[2].set_ylabel(r'$k_z$')
ax[2].set_title(r'Weights $w = 1/\sigma$')
ax[2].grid(visible=True)
fig.colorbar(cax3,ax=ax[2])

fig.suptitle('The variance, smoothed variance and resulting weights using the RSD grid')
fig.savefig('weights_rsd.png')
#plt.show()
plt.close()

fig, ax = plt.subplots(1,3,figsize=(20,10))
cax1 = ax[0].imshow(variance_div, origin='lower',extent=(0,1,0,1),cmap='nipy_spectral',vmin=1e-5,vmax=0.1)
ax[0].set_xlabel(r'$k = \sqrt{k_x^2 + k_y^2}$')
ax[0].set_ylabel(r'$k_z$')
ax[0].set_title(r'Variance $\sigma^2$')
ax[0].grid(visible=True)
fig.colorbar(cax1,ax=ax[0])

cax2 = ax[1].imshow(variance_div_smooth,origin='lower',extent=(0,1,0,1),cmap='nipy_spectral',vmin=1e-5,vmax=0.1)
ax[1].set_xlabel(r'$k = \sqrt{k_x^2 + k_y^2}$')
ax[1].set_ylabel(r'$k_z$')
ax[1].set_title(r'Smoothed variance $\sigma^2$')
ax[1].grid(visible=True)
fig.colorbar(cax2,ax=ax[1])

cax3 = ax[2].imshow(weights_div, origin='lower',extent=(0,1,0,1),cmap='nipy_spectral',vmin=0,vmax=100)
ax[2].set_xlabel(r'$k = \sqrt{k_x^2 + k_y^2}$')
ax[2].set_ylabel(r'$k_z$')
ax[2].set_title(r'Weights $w = 1/\sigma$')
ax[2].grid(visible=True)
fig.colorbar(cax3,ax=ax[2])

fig.suptitle('The variance, smoothed variance and resulting weights using the divided grid')
fig.savefig('weights_div.png')
#plt.show()
plt.close()
