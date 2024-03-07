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
#Change the a_ for various redshifts
a_ = 1/(1+1)
H_a0 = Hubble(a_,h,d3a_omegaM,d3a_omegaLambda)
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

def genPower_1D(position,mass,velocity,Nmesh,dk,kmin,mesh_pos):
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
    z = 1
    RSDPosition = add_RSD(cat,z,line_of_sight)
    cat['RSDPosition'] = RSDPosition

    #convert to a MeshSource, using TSC interpolation on Nmesh^3 
    mesh = cat.to_mesh(window='tsc', Nmesh=Nmesh, compensated=True, position=mesh_pos, BoxSize=[1000,1000,1000])
    
    #compute the fast-fourier transform using linear binning & obtain the power
    fft = FFTPower(mesh,mode='1d',dk=dk,kmin=kmin) 
    Pk = fft.power 

    return Pk['k'], Pk['power'].real, Pk.attrs['shotnoise']

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
    
    #The result is an N by N grid of Pk values 
    return grid

galaxy_filterVel = np.load('/net/draco/data2/vanveenhuyzen/rsd_project/particle_data/galaxy_filterVel.npy')
galaxy_filterMass = np.load('/net/draco/data2/vanveenhuyzen/rsd_project/particle_data/galaxy_filterMass.npy')
galaxy_filterPos = np.load('/net/draco/data2/vanveenhuyzen/rsd_project/particle_data/galaxy_filterPos.npy')

#Here, we want to perform various mass cuts

def galaxy_masscut(pos,vel,mass,m1,m2):
    """
    In this function we want to mask the data by applying a mass cut: we only consider 
    galaxies with masses between m1 and m2
    """

    Nbefore = len(mass)

    #Create a mass-based mask 
    mask = np.where( (mass > m1) & (mass < m2) )

    #Apply the mass over the data 
    pos_masked = pos[mask]
    vel_masked = vel[mask]
    mass_masked = mass[mask]

    Nafter = len(mass_masked)

    #print(mass_masked)

    ratio = Nafter/Nbefore*100
    print(r'The mass-based subset of the data contains {:.4f} % of the total galaxies'.format(ratio))

    return pos_masked,vel_masked,mass_masked 

#k_1D, Pk_1D, shotnoise_1D = genPower_1D(galaxy_filterPos,galaxy_filterMass,galaxy_filterVel,512,0.005,0.01)

x3d = np.load('x3d.npy',allow_pickle=True)
y3d = np.load('y3d.npy')
shotnoise = np.load('shotnoise.npy')
print(shotnoise)
y3d_noSN = y3d-shotnoise

Mmax = np.max(galaxy_filterMass)
Mmin = np.min(galaxy_filterMass)

print('The highest mass galaxy has a mass of {:.2e}'.format(Mmax))
print('The lowest mass galaxy has a mass of {:.2e}'.format(Mmin))

Mhigh = 1e11
Mmid = 5e10
Mveryhigh = 3e11

#Find bin edges on a log scale
bin_edges = np.logspace(np.log10(Mmin), np.log10(Mmax), num=6, base=10)
print('bin edges:',bin_edges)

M1log = bin_edges[1]
M2log = bin_edges[2]
M3log = bin_edges[3]
M4log = bin_edges[4]

t1,t2,t3 = galaxy_masscut(galaxy_filterPos,galaxy_filterVel,galaxy_filterMass,Mmin,M1log)
t1,t2,t3 = galaxy_masscut(galaxy_filterPos,galaxy_filterVel,galaxy_filterMass,M1log,M2log)
t1,t2,t3 = galaxy_masscut(galaxy_filterPos,galaxy_filterVel,galaxy_filterMass,M2log,M3log)
t1,t2,t3 = galaxy_masscut(galaxy_filterPos,galaxy_filterVel,galaxy_filterMass,M3log,M4log)
t1,t2,t3 = galaxy_masscut(galaxy_filterPos,galaxy_filterVel,galaxy_filterMass,M4log,Mmax)

M1 = 2.783e10
M2 = 4.016e10
M3 = 6.001e10
M4 = 9.658e10

#t1,t2,t3 = galaxy_masscut(galaxy_filterPos,galaxy_filterVel,galaxy_filterMass,Mmin,M1)
#t1,t2,t3 = galaxy_masscut(galaxy_filterPos,galaxy_filterVel,galaxy_filterMass,M1,M2)
#t1,t2,t3 = galaxy_masscut(galaxy_filterPos,galaxy_filterVel,galaxy_filterMass,M2,M3)
#t1,t2,t3 = galaxy_masscut(galaxy_filterPos,galaxy_filterVel,galaxy_filterMass,M3,M4)

#t1,t2,t3 = galaxy_masscut(galaxy_filterPos,galaxy_filterVel,galaxy_filterMass,M4,Mmax)

#k_1D, Pk_1D, shotnoise_1D = genPower_1D(galaxy_filterPos,galaxy_filterMass,galaxy_filterVel,512,0.005,0.01,'Position')
#k_1D, Pk_1D, shotnoise_1D = genPower_1D(t1,t3,t2,512,0.005,0.01,'RSDPosition')

mass_z0_5 = np.load('/net/draco/data2/vanveenhuyzen/rsd_project/data_pipeline/z1/z1_filterMass.npy')
pos_z0_5 = np.load('/net/draco/data2/vanveenhuyzen/rsd_project/data_pipeline/z1/z1_filterPos.npy')
vel_z0_5 = np.load('/net/draco/data2/vanveenhuyzen/rsd_project/data_pipeline/z1/z1_filterVel.npy')

k_1D, Pk_1D, shotnoise_1D = genPower_1D(pos_z0_5,mass_z0_5,vel_z0_5,512,0.005,0.01,'Position')

k_ = np.load('massfiltered_files/real/x3d_pos_z1.npy',allow_pickle=True)
p_ = np.load('massfiltered_files/real/y3d_pos_z1.npy')
sn_ = np.load('massfiltered_files/real/shotnoise_pos_z1.npy')

p_nosn = p_-sn_

grid = basis_custom(k_,p_nosn,Npixel=64)
np.save('griddata/grid_pos_z1.npy',grid)

