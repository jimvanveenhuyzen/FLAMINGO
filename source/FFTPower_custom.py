import numpy as np
import matplotlib.pyplot as plt
import dask.array as da

#Add the filepath of the nbodykit sourcecode
#The permissions on the Sterrewacht computers can be messy, so this is just an ad hoc solution
import sys
sys.path.append('/net/draco/data2/vanveenhuyzen/rsd_project/nbodykit_sourcecode')
#The above sys allows us to import the custom version of the nbodykit sourcecode
import nbodykit_custom

from nbodykit_custom.source.catalog import ArrayCatalog
from nbodykit_custom.lab import *
from nbodykit_custom import setup_logging, style

def Hubble(a,h,omega_m,omega_lambda):
    """Compute the Hubble parameter H(a) as a function of scale factor a (redshift),
    using the set cosmology of this simulation, which sets h and the omegas

    Parameters:
    ----------
    a : float
        Scale factor, related to the redshift z of the simulation via a = 1/(1+z)
    h : float
        Reduced Planck's constant, cosmological parameter
    omega_m : float
        Matter density parameter
    omega_lambda : float
        Dark energy density parameter

    Returns:
    -------
    Hubble_a : float 
        Hubble parameter H(a), used to compute redshift-space positions

    """
    Hubble_0 = h*100
    Hubble_a = Hubble_0*np.sqrt(omega_m * a**(-3) + omega_lambda)
    return np.round(Hubble_a,3)

#Define the D3A cosmological parameters as globals since we only use this simulation
d3a_h = 0.681
d3a_omegaM = 0.306
d3a_omegaLambda = 0.694
#Change the a_ for various redshifts
a_ = 1
H_a0 = Hubble(a_,d3a_h,d3a_omegaM,d3a_omegaLambda)
print(H_a0)

def add_RSD(catalogue,z,line_of_sight):
    """Compute the catalogue column of redshift-space positions using H(a), the scale factor a
    and finally the positions and velocities of the galaxy catalogue

    Parameters:
    ----------
    catalogue : ArrayCatalogue object (nbodykit), uses Dask
        The galaxy catalogue containing various galaxy properties like positions, velocities and mass
    z : float
        Redshift of the simulation, used to compute the redshift-space positions
    line_of_sight : unit-vector list: [1,0,0], [0,1,0] or [0,0,1]
        Specify the axis along which we define the line-of-sight 

    Returns:
    -------
    rsd_pos : ArrayCatalogue object (nbodykit), uses Dask
        Catalogue column that contains the galaxy position in redshift space 
    """
    #Convert redshift to scale factor and find H(a)
    a = 1/(1+z)
    H_a = Hubble(a,h=d3a_h,omega_m=d3a_omegaM,omega_lambda=d3a_omegaLambda)

    #Compute the RSD positions
    rsd_pos = catalogue['Position'] + (catalogue['Velocity'] * line_of_sight)/(a*H_a)

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

def compute_power(position,mass,velocity,Nmesh,dk,kmin,z,line_of_sight,mesh_pos):
    """This function generates a 1-dimensional power spectrum from an input vector containing particle positions, mass and velocities
    Note! This function is mainly used to call the (adjusted) FFTPower function, so the return values of this function are mainly unused
    
    Parameters:
    ----------
    position : 2D numpy array of shape (N,3), with N number of halos
        The positions of the halos
    mass : 1D numpy array of shape (N,), with N number of halos
        The stellar masses of the halos 
    velocity : 2D numpy array of shape (N,3), with N number of halos
        The velocities of the halos 
    Nmesh : int (power of 2)
        The size of the meshgrid, results in (0.5*Nmesh*Nmesh*Nmesh) data points
    dk : float
        Linear spacing between interpolated k values
    kmin : float
        k value to start interpolation at
    z : float
        Redshift of the simulation, used to compute the redshift-space positions
    line_of_sight : unit-vector list: [1,0,0], [0,1,0] or [0,0,1]
        Specify the axis along which we define the line-of-sight  
    mesh_pos : string (either 'Position' or 'RSDPosition')
        Specify whether to compute the power using the real-space or redshift-space positions

    Returns: 
    -------
    'k' : 1D array 
        k values of the FFTPower
    'Pk' : 1D array
        Power P(k) values of the FFTPower
    'shotnoise' : float
        The shotnoise associated with the FFT, for 1D power spectra it is a constant value at every k
    """

    #First, create a catalogue and fill the Position and Mass columns
    length = len(position[:,0])
    data = np.empty(length, dtype=[('Position', ('f8', 3)),('Velocity', ('f8', 3)),('Weight', ('f8'))])
    data['Position'] = position
    data['Weight'] = mass
    data['Velocity'] = velocity
    print('We use {} particle positions.'.format(length))

    #Create an array catalog
    cat = ArrayCatalog(data)
    print("The columns are ", cat.columns)

    #Call the add_RSD function to compute the RSDPosition catalogue column 
    RSDPosition = add_RSD(cat,z,line_of_sight)
    cat['RSDPosition'] = RSDPosition

    #Convert to a MeshSource, using TSC interpolation on Nmesh^3 
    mesh = cat.to_mesh(window='tsc', Nmesh=Nmesh, compensated=True, position=mesh_pos, BoxSize=[1000,1000,1000])
    
    #Compute the fast-fourier transform using linear binning & obtain the power, calls upon the custom version of FFTPower! 
    fft = FFTPower(mesh,mode='1d',dk=dk,kmin=kmin) 
    Pk = fft.power 

    return Pk['k'], Pk['power'].real, Pk.attrs['shotnoise']

def basis_custom(k,p,Npixel):
    """In this function we define an alternate way of gridding the data directly to a 2-dimesional grid, 
    consisting of sqrt(kx^2+ky^2) and kz, with the grid values being the corresponding power values P(k).
    Note that the x-axis of the final grid is given by k_abs, while the y-axis is k_z

    The input to this function is data mid-way in the FFTPower routine from nbodykit, so we have to play
    around with array dimensions a bit in the beginning 

    Parameters:
    ----------
    k : 
        k values in the mesh
    p :  
        Power values in the mesh
    Npixel :
        Defines the number of pixels we will use to project the data onto the grid. 

    Returns: 
    -------
    grid:
        The final grid filled with power values from k=0 to k=1
    num_values: 
        Tracks the number of values we took the mean over per grid point, used in assigning model weights 
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

    #In the following nested loop we compute the absolute value of k = sqrt(kx^2 + ky^2)
    #I could use numpy broadcasting here, but for Nmesh = 64 speed doesn't matter here too much 
    k_abs = np.zeros((Nmesh,Nmesh))
    for x in range(x_axis):
        for y in range(y_axis):
            
            k_ = kx[x]**2  + ky[y]**2
            k_abs[x,y] = np.sqrt(k_)

    #Next, we create a 3D cube that stacks all k_abs values along the x- and y axis upon each other [kz.shape] times 
    k_abs_cube = np.dstack([k_abs]*z_axis)

    #Also create a 3D cube that stacks all 2D layers inside a cube, the 2D layers all having the same value, namely the kz of the current layer 
    kz_cube = np.ones((Nmesh,Nmesh,int(0.5*Nmesh+1)))
    for z in range(z_axis):
        kz_cube[:,:,z] *= kz[z]

    kmax = 1 #The maximum k value we want to display in the image, constant in this project, could be set as additional argument 

    """Here, we multiply k values by Npixel and set their datatype as integer. This is done so that we can compare integers in 
    the np.where mask inside the loop, rather than comparing floats. Comparing floats with == is a bad idea since this leads to
    numerical instability of the algorithm due to e.g. rounding errors inherent to floats 
    """
    pixel_points = np.linspace(0,kmax-0.01,Npixel)
    grid_fac = np.floor(Npixel*pixel_points/kmax).astype(int)

    k_abs_cube_fac = np.floor(Npixel*k_abs_cube/kmax).astype(int)
    kz_cube_fac = np.floor(Npixel*kz_cube/kmax).astype(int)

    grid = np.zeros((Npixel,Npixel))
    num_values = np.zeros((Npixel,Npixel))

    #We do a double for loop, starting with a loop over kz (y axis of the grid), and per y we loop through kabs (x axis of the grid)
    for kz_,zidx in enumerate(grid_fac):
        print('Current kz = {0}/{1}'.format(kz_,Npixel))

        for kabs_,kidx in enumerate(grid_fac):

            #We scan the cube for points that have the same coordinates as the current (kz,kabs), if True, we fill these into the grid
            mask3d = np.where( (kz_ == kz_cube_fac) & (kabs_ == k_abs_cube_fac) )
            pkcurr = p.real[mask3d]

            if len(pkcurr) > 0:
                grid[zidx,kidx] = np.mean(pkcurr) #Take the mean in case of multiple mesh-points being assigned per grid point
                num_values[zidx,kidx] = len(pkcurr) #Keep track of the amount of values assigned to each grid point 
    
    #The result is an Npixel by Npixel grid of power P(k) values 
    return grid, num_values

k_1D, Pk_1D, shotnoise_1D = compute_power(galaxy_filterPos,galaxy_filterMass,galaxy_filterVel,512,0.005,0.01,0,[0,0,1],'Position') 

#Now, using the values from FFTPower saved elsewhere we call the basis_custom function to get our final grid
k_ = np.load('massfiltered_files/real/x3d_pos_mall.npy',allow_pickle=True)
p_ = np.load('massfiltered_files/real/y3d_pos_mall.npy')
sn_ = np.load('massfiltered_files/real/shotnoise_pos_mall.npy')

p_nosn = p_-sn_ #Subtract the shotnoise

print('The shotnoise is {:.3f}'.format(sn_))

grid,num_val = basis_custom(k_,p_nosn,Npixel=64)
#np.save('griddata/grid_pos_mall64.npy',grid)
