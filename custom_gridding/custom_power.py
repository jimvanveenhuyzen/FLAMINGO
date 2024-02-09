import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append('/net/draco/data2/vanveenhuyzen/rsd_project/nbodykit_sourcecode')

import nbodykit_custom
print(nbodykit_custom.__file__)

from nbodykit_custom.source.catalog import ArrayCatalog
from nbodykit_custom.lab import *
from nbodykit_custom import setup_logging, style

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
    #convert to a MeshSource, using TSC interpolation on Nmesh^3 
    mesh = cat.to_mesh(window='tsc', Nmesh=Nmesh, compensated=True, position='Position', BoxSize=[1000,1000,1000])
    
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

def basis_custom(y3d):
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

    test = np.load('x3d.npy',allow_pickle=True)
    for t in test:
        print(t.shape)

    #Flatten to get 1D arrays
    kx = x3d[0].flatten()
    ky = x3d[1].flatten()
    kz = x3d[2].flatten()

    x_axis = y3d.shape[0]
    y_axis = y3d.shape[1]
    z_axis = y3d.shape[2]

    #Set up the grid arrays
    gridpoints = np.linspace(0,0.99,100)
    gridpoints_sq = gridpoints**2 
    grid_x = np.copy(gridpoints_sq)
    grid_y = np.copy(gridpoints)

    #In the following nested loop we compute the absolute values of kx^2 + ky^2 
    k_abs = np.zeros((512,512))
    for x in range(x_axis):
        for y in range(y_axis):
            
            k_ = kx[x]**2  + ky[y]**2
            k_abs[x,y] = k_ 
        
    #Multiply by num of grid points for easier comparison
    k_abs = np.floor(100*k_abs).astype(int)

    #As a result we now have a N by N grid array with the Pk values filled in, but we need to match k_z with the grid
    gridpoints_factor = np.floor(100*gridpoints).astype(int)
    kz_factor = np.floor(100*kz).astype(int)

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

    z_indices = z_indices[1:]
    print(z_indices)

    #In this double loop, we fill the grid: first by looping over k_z, and then obtaining all values where |k|
    #is equal to the current sqrt(kx^2+ky^2) grid point.  
    grid_ = np.zeros((100,100))
    grid_total = []
    for idx in z_indices: 
        grid_i = []
        for k_ in range(len(grid_[0])):

            k_current = np.floor(100*gridpoints_sq[k_]).astype(int)
            mask = np.where(k_current == k_abs)

            Pk_values = y3d.real[mask,idx]
            if len(Pk_values) > 0:
                #grid_[idx,k_] = np.mean(Pk_values)
                grid_i.append(np.mean(Pk_values))
            else:
                #grid_[idx,k_] = 0.0001
                grid_i.append(0.0001)

    
        grid_total.append(grid_i)

    """
    grid = np.zeros((100,100))
    for i in range(100):
        for j in range(100):

            k_current = np.floor(100*gridpoints_sq[j]).astype(int)
            mask = np.where(k_current == k_abs)

            Pk_values = y3d.real[mask,i]
            if len(Pk_values) > 0:
                grid[i,j] = np.mean(Pk_values)
            else:
                grid[i,j] = 0.0001

    """

    return grid_total

grid = basis_custom(1)

fig, ax = plt.subplots(figsize=(8, 6))
ax.set_xlabel(r'$k = \sqrt{k_x^2 + k_y^2}$')
ax.set_ylabel(r'$k_z$')
ax.set_title('Filled grid directly from compute_3d_power (y3d and y3d.x)')
ax.grid(visible=True)
cax = ax.imshow(grid, origin='lower',extent=(0,1,0,1),cmap='nipy_spectral')#,norm=matplotlib.colors.LogNorm(vmin=5e1,vmax=1.5e5))
fig.colorbar(cax,label='3D power')
plt.savefig('rsd.png')


