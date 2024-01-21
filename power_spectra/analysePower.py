import numpy as np
import matplotlib.pyplot as plt
import h5py
import matplotlib.colors

from nbodykit.source.catalog import ArrayCatalog
from nbodykit.lab import *

#What we want to do now is create a 2D powerspectrum P(k,mu) with non-NaN k-values for (very) small mu

galaxy_filterVel = np.load('/net/draco/data2/vanveenhuyzen/rsd_project/particle_data/galaxy_filterVel.npy')
galaxy_filterMass = np.load('/net/draco/data2/vanveenhuyzen/rsd_project/particle_data/galaxy_filterMass.npy')
galaxy_filterPos = np.load('/net/draco/data2/vanveenhuyzen/rsd_project/particle_data/galaxy_filterPos.npy')

position,velocity,mass = galaxy_filterPos,galaxy_filterVel,galaxy_filterMass
length = len(position[:,0])
data = np.empty(length, dtype=[('Position', ('f8', 3)),('Velocity', ('f8', 3)),('Weight', ('f8'))])
data['Position'] = position
data['Weight'] = mass
data['Velocity'] = velocity 
print('We use {} particle positions.'.format(length))

#Create an array catalog
cat = ArrayCatalog(data)
print("The columns are ", cat.columns)

line_of_sight = [0,0,1]
Nmesh = 512
kmin = 0.1
dk = 0.05
Nmu = 11

mesh = cat.to_mesh(window='tsc', Nmesh=Nmesh, compensated=True, position='Position', BoxSize=[1000,1000,1000])
fft = FFTPower(mesh,mode='2d',dk=dk,kmin=kmin,Nmu=Nmu,los=line_of_sight) 
Pkmu = fft.power 

np.set_printoptions(threshold=np.inf)

k_data = Pkmu['k'][:,int(Nmu/2):]
Pk_data = Pkmu['power'][:,int(Nmu/2):]
SN_data = Pkmu.attrs['shotnoise']
mu_data = Pkmu['mu'][:,int(Nmu/2):]

#print(mu_data[:100,:])

###################
#Now, lets rewrite the converting to kx and ky using the new data (shapes):
###################

k_val_2D = np.load('2Dpower_k.npy')
Pk_val_2D = np.load('2Dpower_pk.npy')
mu_val_2D = np.load('2Dpower_mu.npy')

print('The shape of Pk is')
print(Pk_val_2D.shape)
print('The shape of k and mu are')
print(k_val_2D.shape)
print(mu_val_2D.shape)

#calculate kx and ky 
kx = k_val_2D * mu_val_2D
ky = k_val_2D * np.sqrt(1 - (mu_val_2D)**2)

def fill2Dgrid(kx,ky,Pk):

    #First, lets create a grid of 100 by 100 points within the range k_x,k_y = 0-1
    x = np.linspace(0,0.99,64,dtype=np.float32)
    x_gridpoints,y_gridpoints = np.meshgrid(x,x)

    x,y,z = kx,ky,Pk

    x_factor100 = np.floor(64*x).astype(int)
    y_factor100 = np.floor(64*y).astype(int)

    grid = np.zeros((64,64))
    for i in range(len(x_gridpoints[0])):
        for j in range(len(y_gridpoints[0])):
            kx_current = np.floor(64*x_gridpoints[i,j]).astype(int)
            ky_current = np.floor(64*y_gridpoints[i,j]).astype(int)

            mask = np.where((kx_current == x_factor100)&(ky_current == y_factor100))
            pk_values = z[mask]

            if len(pk_values) > 0:
                grid[i,j] = np.mean(pk_values)
            else:
                grid[i,j] = 1e-5

            #Check for the missing values:
            #We do see a lot of zeroes...
            #if ky_current < 0.1:
            #    count += 1 
            #    mask = np.where((kx_current == x_factor100)&(ky_current == y_factor100))
            #    print(np.sum(mask))
                
    fig, ax = plt.subplots(figsize=(8, 6))
    cax = ax.imshow(grid,extent=(0, 1, 0, 1), origin='lower', cmap='nipy_spectral',norm=matplotlib.colors.LogNorm())
    fig.colorbar(cax,label='P(k) value')
    ax.set_xlabel(r'$k_x$')
    ax.set_ylabel(r'$k_y$')
    ax.set_title('RSD grid [0,1,0] without interp')
    ax.grid(visible=True)
    plt.savefig('withRSD_64grid_010_19012024.png')
    plt.close()

    return grid

grid2D = fill2Dgrid(kx,ky,Pk_val_2D)

def fullPlot(grid):
    mirrorX = np.flip(grid,axis=0)
    mirrorY = np.flip(grid,axis=1)
    flip = np.flip(grid)

    top = np.concatenate((flip,mirrorX),axis=1)
    bottom = np.concatenate((mirrorY,grid),axis=1)
    total = np.concatenate((top,bottom),axis=0)

    fig, ax = plt.subplots(figsize=(8, 6))
    cax = ax.imshow(total,extent=(-1, 1, -1, 1), origin='lower', cmap='nipy_spectral',norm=matplotlib.colors.LogNorm())
    fig.colorbar(cax,label='P(k) value')
    ax.set_xlabel(r'$k_x$')
    ax.set_ylabel(r'$k_y$')
    ax.set_title('RSD grid [0,1,0] without interp')
    ax.grid(visible=True)
    plt.savefig('withRSDfull_64grid_010_19012024.png')
    plt.close()

fullPlot(grid2D)

"""

k_val = np.load('1Dpower_k.npy')
Pk_val = np.load('1Dpower_pk.npy')
plt.plot(k_val,Pk_val)
plt.xlabel('k values')
plt.ylabel('Pk values')
plt.xscale('log')
plt.yscale('log')
plt.title('1D power spectrum with no RSD')
plt.savefig('1Dpower_plot15012024.png')
plt.close()

k_val_2D = np.load('2Dpower_k.npy')
Pk_val_2D = np.load('2Dpower_pk.npy')
k_val_mu80 = k_val_2D[80,:]
Pk_val_mu80 = Pk_val_2D[80,:]

plt.plot(k_val_mu80,Pk_val_mu80)
plt.xlabel('k values')
plt.ylabel('Pk values')
plt.xscale('log')
plt.yscale('log')
#plt.title('Plot of the Pk values as a function of mu at k={:.2f}'.format(k_val_mu80))
plt.savefig('2Dpower_15012024.png')
plt.close()
k_val = np.load('kVAL_51.npy')
Pk_val = np.load('pkVAL_51.npy')
mu_val = np.load('muVAL_51.npy')

k_val = np.load('k_val_avg.npy')
Pk_val = np.load('Pk_val_avg.npy')
mu_val = np.load('mu_val_avg.npy')


print(k_val.shape)
print(Pk_val.shape)

k_value_80 = np.nanmean(k_val[:,100])
#print(k_val[:,10])
Pk_val_k80 = Pk_val[:,100]

plt.plot(mu_val,Pk_val_k80)
plt.xlabel('mu values')
plt.ylabel('Pk values')
#plt.yscale('log')
plt.title('Plot of the Pk values as a function of mu at k={:.2f}'.format(k_value_80))
plt.savefig('Pk_mu_19122023.png')
plt.close()

#Plot the full power spectrum for various mu values:
plt.plot(k_val,Pk_val)
plt.xlabel('k values')
plt.ylabel('Pk values')
plt.yscale('log')
plt.title('Plot of the whole power spectrum')
plt.savefig('fullspectrum_19122023.png')
plt.close()
"""