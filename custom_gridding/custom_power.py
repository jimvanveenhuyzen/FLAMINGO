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

def basis_custom(y3d,x3d,gridsize):
    """
    In this function we define an alternate way of gridding the data directly to a 2-dimesional grid, 
    consisting of sqrt(kx^2+ky^2) and kz, with the grid values being the corresponding power values P(k)

    Arguments:
        -y3d: power values in the mesh
        -x3d: k values in the mesh
        -gridsize: defines the number of pixels we will use to project the data onto the grid. 

    Returns: 
        -grid: the final grid 
    """

    N = gridsize 

    #y3d = np.random.random((128,128,65))
    #y3d = np.array(y3d,dtype='complex')
    #print(y3d)

    print('-'*100)
    print('Starting custom function...')

    y3d = np.load('y3d.npy')
    print(np.shape(y3d))

    #Generate some mock data 
    x3d_0 = np.linspace(-1.6,1.6,512)
    x3d_1 = np.linspace(-1.6,1.6,512)
    x3d_2 = np.linspace(0.01,1.6,256)
    x3d_2 = np.append(x3d_2,-1.6)

    #Match the shape of the real case 
    x3d = [x3d_0.reshape(512,1,1),x3d_1.reshape(1,512,1),x3d_2.reshape(1,1,257)]

    test = np.load('x3d.npy',allow_pickle=True) #require allow_picke=True for some reason 
    for t in test:
        print(t.shape)

    #Flatten to get 1D arrays
    kx = test[0].flatten()
    ky = test[1].flatten()
    kz = test[2].flatten()

    #print(kz)

    x_axis = y3d.shape[0]
    y_axis = y3d.shape[1]
    z_axis = y3d.shape[2]

    p_kxconst = y3d.real[510,256:,:-1]
    print(p_kxconst.shape)
    p_kyconst = y3d.real[:,500,:]
    #print(ky)
    print(ky[500])
    p_kzconst = y3d.real[:,:,250]

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
    gridpoints = np.linspace(0,0.99,N)

    #In the following nested loop we compute the absolute value of k = sqrt(kx^2 + ky^2)
    k_abs = np.zeros((512,512))
    for x in range(x_axis):
        for y in range(y_axis):
            
            k_ = kx[x]**2  + ky[y]**2
            k_abs[x,y] = np.sqrt(k_)

    #Next, we create a 3D cube that stacks all k_abs values along the x- and y axis upon each other [kz.shape] times 
    k_abs_cube = np.dstack([k_abs]*z_axis)
    print(k_abs_cube.shape)


    #Also create a 3D cube that stacks all 2D layers inside a cube, the 2D layers all having the same value, namely the kz of the current layer 
    kz_cube = np.ones((512,512,257))
    for z in range(z_axis):
        kz_cube[:,:,z] *= kz[z]

    kmax = 1 #The maximum k value we want to display in the image

    pixel_points = np.linspace(0,kmax-0.01,N)
    grid_fac = np.floor(N*pixel_points/kmax).astype(int)

    k_abs_cube_fac = np.floor(N*k_abs_cube/kmax).astype(int)
    kz_cube_fac = np.floor(N*kz_cube/kmax).astype(int)

    grid_alt = np.zeros((N,N))
    for kz_,zidx in enumerate(grid_fac):
        print('Current kz:',kz_)

        for kabs_,kidx in enumerate(grid_fac):

            mask3d = np.where( (kz_ == kz_cube_fac) & (kabs_ == k_abs_cube_fac) )
            pkcurr = y3d.real[mask3d]

            if len(pkcurr) > 0:
                grid_alt[zidx,kidx] = np.mean(pkcurr) #I had flipped zidx and kidx originally so some images are swapped (|k| and kz that is)
    
    print(grid_alt)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xlabel(r'$k = \sqrt{k_x^2 + k_y^2}$')
    ax.set_ylabel(r'$k_z$')
    ax.set_title('Filled grid directly from using y3d and y3d.x, Npix=32')
    ax.grid(visible=True)
    cax = ax.imshow(grid_alt, origin='lower',extent=(0,1,0,1),cmap='nipy_spectral')#,norm=matplotlib.colors.LogNorm(vmin=5e1,vmax=1.5e5))
    fig.colorbar(cax,label='3D power')
    #plt.savefig('RSD__14022024.png')

    np.save('rsdgrid_Npix64_14022024.npy',grid_alt)
    
    #Multiply by num of grid points for more accurate comparison (floats versus ints)
    k_abs = np.floor(N*k_abs).astype(int)

    #As a result we now have a N by N grid array with the Pk values filled in, but we need to match k_z with the grid
    gridpoints_factor = np.floor(N*gridpoints).astype(int)
    kz_factor = np.floor(N*kz).astype(int)

    #First, we use a single loop to obtain the indices we need to use to slice the k_z coordinate for y3d 
    z_indices = np.zeros(1,dtype=int)
    for z in range(len(gridpoints_factor)):

        kz_current = np.floor(gridpoints_factor[z]).astype(int)
        mask = np.where(kz_factor == kz_current)

        if len(mask[0]) > 1:
            mask = mask[0][0]
        else:
            mask = mask[0]

        #append the index to the array
        z_indices = np.append(z_indices,mask)

    #As a result, we now have an array that lists the indices of y3d that correspond to the k_z values of the grid
    z_indices = z_indices[1:]
    #print(z_indices)

    #In this double loop, we fill the grid: first by looping over k_z, and then obtaining all values where |k|
    #is equal to the current sqrt(kx^2+ky^2) grid point. 
    grid = []
    for idx in z_indices: 
        grid_i = []
        for k_ in range(N):

            #k_current = np.floor(100*gridpoints_sq[k_]).astype(int)

            k_current = gridpoints_factor[k_]
            mask = np.where(k_current == k_abs)

            Pk_values = y3d.real[mask,idx]

            #Case 1: there are one or more Pk values for the grid point 
            if len(Pk_values) > 0:
                grid_i.append(np.mean(Pk_values))
            #Case 2: we have no Pk values for the grid point 
            else:
                grid_i.append(0.0001)
        
        #Append the k_z row to the total grid 
        grid.append(grid_i)

    #The result is an N by N grid of Pk values 
    return grid

grid = basis_custom(1,1,gridsize=64)

fig, ax = plt.subplots(figsize=(8, 6))
ax.set_xlabel(r'$k = \sqrt{k_x^2 + k_y^2}$')
ax.set_ylabel(r'$k_z$')
ax.set_title('Filled grid directly from compute_3d_power (y3d and y3d.x), N=64')
ax.grid(visible=True)
cax = ax.imshow(grid, origin='lower',extent=(0,1,0,1),cmap='nipy_spectral')#,norm=matplotlib.colors.LogNorm(vmin=5e1,vmax=1.5e5))
fig.colorbar(cax,label='3D power')
#plt.savefig('rsd_11022024.png')


