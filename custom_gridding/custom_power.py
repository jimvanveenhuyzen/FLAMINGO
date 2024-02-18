import numpy as np
import matplotlib.pyplot as plt
import dask.array as da

import sys
sys.path.append('/net/draco/data2/vanveenhuyzen/rsd_project/nbodykit_sourcecode')

import nbodykit_custom
print(nbodykit_custom.__file__)

from nbodykit_custom.source.catalog import ArrayCatalog
from nbodykit_custom.lab import *
from nbodykit_custom import setup_logging, style

def Hubble(a,h,omega_m,omega_lambda):
    Hubble_0 = h*100
    Hubble_a = Hubble_0*np.sqrt(omega_m * a**(-3) + omega_lambda)
    return np.round(Hubble_a,3)

h = 0.681
d3a_omegaM = 0.306
d3a_omegaLambda = 0.694
H_a0 = Hubble(1,h,d3a_omegaM,d3a_omegaLambda)
print(H_a0)

def add_RSD(catalogue,z,line_of_sight):
    a = 1/(1+z)
    rsd_pos = catalogue['Position'] + (catalogue['Velocity'] * line_of_sight)/(a*H_a0)

    #Before we return the RSD Positions, we add the periodicity: positions cannot exceed 1000 or drop below 0
    #First, determine the axis of which the RSD effect is added 
    los_idx = int(np.nonzero(line_of_sight)[0])

    #The catalogue entries are Dask arrays, so we need to convert to numpy to perform vector operations
    rsd_pos_numpy = rsd_pos.compute()

    #Second, we use two conditions to 'reverse' the positional distances that exceed 1000 or drop below 0
    mask_low = np.where(rsd_pos_numpy[:,los_idx] < 0)[0]
    mask_high = np.where(rsd_pos_numpy[:,los_idx] > 1000)[0]

    #Apply the masks:
    rsd_pos_numpy[mask_low,los_idx] = 1000. - (1000.+rsd_pos_numpy[mask_low,los_idx]) 
    rsd_pos_numpy[mask_high,los_idx] = 2*1000. - rsd_pos_numpy[mask_high,los_idx] #This works assuming the RSD Positions are not >2000, which they are never

    #Finally, convert back to a dask dataframe column 
    rsd_pos = da.from_array(rsd_pos_numpy,chunks=rsd_pos.chunks)

    return rsd_pos

def genPower_1D(position,mass,velocity,Nmesh,dk,kmin,show_data):
    """
    This function generates a 1-dimensional power spectrum from an input vector containing particle positions
    """

    #First, create a catalogue and fill the Position and Mass columns
    length = len(position[:,0])
    data = np.empty(length, dtype=[('Position', ('f8', 3)),('Velocity', ('f8', 3)),('Weight', ('f8'))])
    data['Position'] = position
    data['Weight'] = mass
    data['Velocity'] = velocity
    print('We use {} particle positions.'.format(length))

    #create an array catalog
    
    cat = ArrayCatalog(data)
    print("The columns are ", cat.columns)

    line_of_sight = [0,0,1]
    z = 0
    RSDPosition = add_RSD(cat,z,line_of_sight)
    cat['RSDPosition'] = RSDPosition

    #convert to a MeshSource, using TSC interpolation on Nmesh^3 
    mesh = cat.to_mesh(window='tsc', Nmesh=Nmesh, compensated=True, position='RSDPosition', BoxSize=[1000,1000,1000])
    
    #compute the fast-fourier transform using linear binning & obtain the power
    fft = FFTPower(mesh,mode='1d',dk=dk,kmin=kmin) 
    Pk = fft.power 

    if show_data == False: 
        # print out the meta-data
        for k in Pk.attrs:
            print("%s = %s" %(k, str(Pk.attrs[k])))

    return Pk['k'], Pk['power'].real, Pk.attrs['shotnoise']

#####################################################################
#From this part, we read in the data and save the FFT power results!
#####################################################################


galaxy_filterVel = np.load('/net/draco/data2/vanveenhuyzen/rsd_project/particle_data/galaxy_filterVel.npy')
galaxy_filterMass = np.load('/net/draco/data2/vanveenhuyzen/rsd_project/particle_data/galaxy_filterMass.npy')
galaxy_filterPos = np.load('/net/draco/data2/vanveenhuyzen/rsd_project/particle_data/galaxy_filterPos.npy')

k_1D, Pk_1D, shotnoise_1D = genPower_1D(galaxy_filterPos,galaxy_filterMass,galaxy_filterVel,512,0.005,0.01,False)
print(Pk_1D[0:5])

def basis_custom(k,p,Npixel):
    """
    In this function we define an alternate way of gridding the data directly to a 2-dimesional grid, 
    consisting of sqrt(kx^2+ky^2) and kz, with the grid values being the corresponding power values P(k)

    Arguments:
        -p: power values in the mesh
        -k: k values in the mesh
        -Npixel: defines the number of pixels we will use to project the data onto the grid. 

    Returns: 
        -grid: the final grid 
    """

    #Flatten to get 1D arrays
    kx = k[0].flatten()
    ky = k[1].flatten()
    kz = k[2].flatten()

    x_axis = p.shape[0]
    y_axis = p.shape[1]
    z_axis = p.shape[2]

    #Obtain the size of the coordinate meshgrid used to get the data: this data has shape (Nmesh,Nmesh,Nmesh/2 + 1)
    Nmesh = x_axis

    p_kxconst = p.real[510,256:,:-1]
    print(p_kxconst.shape)
    p_kyconst = p.real[:,500,:]
    #print(ky)
    print(ky[500])
    p_kzconst = p.real[:,:,250]

    p_kxconst_log = np.where(p_kxconst > 0,np.log10(p_kxconst),0)
    p_kyconst_log = np.where(p_kyconst > 0,np.log10(p_kyconst),0)
    p_kzconst_log = np.where(p_kzconst > 0,np.log10(p_kzconst),0)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xlabel(r'$k_x$')
    ax.set_ylabel(r'$k_y$')
    ax.set_title('k_y, k_z grid of P(k) values using some constant k_x')
    ax.grid(visible=True)
    cax = ax.imshow(p_kxconst_log,origin='lower',\
                            extent=(0,np.max(ky),0,np.max(kz)),cmap='nipy_spectral')#,norm=matplotlib.colors.LogNorm(vmin=5e1,vmax=1.5e5))
    fig.colorbar(cax,label='3D power')
    plt.savefig('kx_const_14022024.png')

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xlabel(r'$k_x$')
    ax.set_ylabel(r'$k_z$')
    ax.set_title('k_x, k_z grid of P(k) values using some constant k_y')
    ax.grid(visible=True)
    cax = ax.imshow(p_kyconst, origin='lower',extent=(np.min(kx),np.max(kx),np.min(kz),np.max(kz)),cmap='nipy_spectral')#,norm=matplotlib.colors.LogNorm(vmin=5e1,vmax=1.5e5))
    fig.colorbar(cax,label='3D power')
    plt.savefig('ky_const_14022024.png')

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xlabel(r'$k_x$')
    ax.set_ylabel(r'$k_z$')
    ax.set_title('k_x, k_y grid of P(k) values using some constant k_z')
    ax.grid(visible=True)
    cax = ax.imshow(p_kzconst, origin='lower',extent=(np.min(kx),np.max(kx),np.min(ky),np.max(ky)),cmap='nipy_spectral')#,norm=matplotlib.colors.LogNorm(vmin=5e1,vmax=1.5e5))
    fig.colorbar(cax,label='3D power')
    plt.savefig('kz_const_14022024.png')

    #Set up the grid arrays
    gridpoints = np.linspace(0,0.99,Npixel)

    #In the following nested loop we compute the absolute value of k = sqrt(kx^2 + ky^2)
    k_abs = np.zeros((Nmesh,Nmesh))
    for x in range(x_axis):
        for y in range(y_axis):
            
            k_ = kx[x]**2  + ky[y]**2
            k_abs[x,y] = np.sqrt(k_)

    #Next, we create a 3D cube that stacks all k_abs values along the x- and y axis upon each other [kz.shape] times 
    k_abs_cube = np.dstack([k_abs]*z_axis)
    print(k_abs_cube.shape)


    #Also create a 3D cube that stacks all 2D layers inside a cube, the 2D layers all having the same value, namely the kz of the current layer 
    kz_cube = np.ones((Nmesh,Nmesh,int(0.5*Nmesh+1)))
    for z in range(z_axis):
        kz_cube[:,:,z] *= kz[z]

    kmax = 1 #The maximum k value we want to display in the image

    pixel_points = np.linspace(0,kmax-0.01,Npixel)
    grid_fac = np.floor(Npixel*pixel_points/kmax).astype(int)

    k_abs_cube_fac = np.floor(Npixel*k_abs_cube/kmax).astype(int)
    kz_cube_fac = np.floor(Npixel*kz_cube/kmax).astype(int)

    grid = np.zeros((Npixel,Npixel))
    for kz_,zidx in enumerate(grid_fac):
        print('Current kz = {0}/{1}'.format(kz_,Npixel))

        for kabs_,kidx in enumerate(grid_fac):

            mask3d = np.where( (kz_ == kz_cube_fac) & (kabs_ == k_abs_cube_fac) )
            pkcurr = p.real[mask3d]

            if len(pkcurr) > 0:
                grid[zidx,kidx] = np.mean(pkcurr) #I had flipped zidx and kidx originally so some images are swapped (|k| and kz that is)
    
    print(grid)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xlabel(r'$k = \sqrt{k_x^2 + k_y^2}$')
    ax.set_ylabel(r'$k_z$')
    ax.set_title('Filled grid directly from using p and k, Npix=8')
    ax.grid(visible=True)
    cax = ax.imshow(grid, origin='lower',extent=(0,1,0,1),cmap='nipy_spectral')#,norm=matplotlib.colors.LogNorm(vmin=5e1,vmax=1.5e5))
    fig.colorbar(cax,label='3D power')
    plt.savefig('RSD_Npix32_17022024.png')

    np.save('rsdgrid_Npix32_17022024.npy',grid)

    #The result is an N by N grid of Pk values 
    return grid

x3d = np.load('x3d.npy',allow_pickle=True)
y3d = np.load('y3d.npy')

grid = basis_custom(x3d,y3d,Npixel=32)

fig, ax = plt.subplots(figsize=(8, 6))
ax.set_xlabel(r'$k = \sqrt{k_x^2 + k_y^2}$')
ax.set_ylabel(r'$k_z$')
ax.set_title('Filled grid directly from compute_3d_power (y3d and y3d.x), N=64')
ax.grid(visible=True)
cax = ax.imshow(grid, origin='lower',extent=(0,1,0,1),cmap='nipy_spectral')#,norm=matplotlib.colors.LogNorm(vmin=5e1,vmax=1.5e5))
fig.colorbar(cax,label='3D power')
#plt.savefig('rsd_11022024.png')


