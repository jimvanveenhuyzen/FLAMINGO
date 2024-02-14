import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.colors

grid = np.load('posgrid_Npix64_14022024.npy')
rsdgrid = np.load('rsdgrid_Npix64_14022024.npy')

def fullGrid(grid):
    mirrorX = np.flip(grid,axis=0)
    mirrorY = np.flip(grid,axis=1)
    flip = np.flip(grid)

    top = np.concatenate((flip,mirrorX),axis=1)
    bottom = np.concatenate((mirrorY,grid),axis=1)
    total = np.concatenate((top,bottom),axis=0)

    return total

fig, ax = plt.subplots(figsize=(8, 6))
ax.set_xlabel(r'$k = \sqrt{k_x^2 + k_y^2}$')
ax.set_ylabel(r'$k_z$')
ax.set_title('Filled grid directly from using y3d and y3d.x, Npix=64')
ax.grid(visible=True)
cax = ax.imshow(fullGrid(rsdgrid), origin='lower',extent=(0,1,0,1),cmap='nipy_spectral',norm=matplotlib.colors.LogNorm())
fig.colorbar(cax,label='3D power')
plt.savefig('rsdgrid_Npix64_log10_15022024.png')
plt.show()

fig, ax = plt.subplots(figsize=(8, 6))
ax.set_xlabel(r'$k = \sqrt{k_x^2 + k_y^2}$')
ax.set_ylabel(r'$k_z$')
ax.set_title('Filled grid directly from using y3d and y3d.x, Npix=64')
ax.grid(visible=True)
cax = ax.imshow(fullGrid(rsdgrid/grid), origin='lower',extent=(0,1,0,1),cmap='nipy_spectral')
fig.colorbar(cax,label='3D power')
plt.savefig('RSDdivided_Npix64_15022024.png')
plt.show()
