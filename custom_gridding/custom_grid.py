import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.colors

grid = np.load('posgrid_Npix64_14022024.npy')
rsdgrid = np.load('rsdgrid_Npix64_14022024.npy')

grid_nbodykit = np.load('posgrid_nbodykit_15022024.npy')
rsdgrid_nbodykit = np.load('rsdgrid_nbodykit_15022024.npy')

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
ax.set_title('Filled Pos grid directly from using y3d and y3d.x, Npix=64')
ax.grid(visible=True)
cax = ax.imshow(fullGrid(grid), origin='lower',extent=(-1,1,-1,1),cmap='nipy_spectral',norm=matplotlib.colors.LogNorm(vmin=50,vmax=1.1e5))
fig.colorbar(cax,label='3D power')
plt.savefig('posgrid_Npix64_log10_15022024.png')
plt.show()

fig, ax = plt.subplots(figsize=(8, 6))
ax.set_xlabel(r'$k = \sqrt{k_x^2 + k_y^2}$')
ax.set_ylabel(r'$k_z$')
ax.set_title('Filled divided grid directly from using y3d and y3d.x, Npix=64')
ax.grid(visible=True)
cax = ax.imshow(fullGrid(rsdgrid/grid), origin='lower',extent=(-1,1,-1,1),cmap='nipy_spectral')
fig.colorbar(cax,label='3D power')
plt.savefig('RSDdivided_Npix64_15022024.png')
plt.show()

fig, ax = plt.subplots(figsize=(8, 6))
ax.set_xlabel(r'$k_x$')
ax.set_ylabel(r'$k_y$')
ax.set_title('Pos grid from nbodykit output in k and mu, Npix=64')
ax.grid(visible=True)
cax = ax.imshow(fullGrid(grid_nbodykit), origin='lower',extent=(-1,1,-1,1),cmap='nipy_spectral',norm=matplotlib.colors.LogNorm(vmin=50,vmax=1.1e5))
fig.colorbar(cax,label='3D power')
plt.savefig('nbodykitposgrid_Npix64_15022024.png')
plt.show()

fig, ax = plt.subplots(figsize=(8, 6))
ax.set_xlabel(r'$k_x$')
ax.set_ylabel(r'$k_y$')
ax.set_title('Pos grids divided, Npix=64')
ax.grid(visible=True)
cax = ax.imshow(fullGrid(grid/grid_nbodykit), origin='lower',extent=(-1,1,-1,1),cmap='nipy_spectral',vmin=0.9,vmax=1.1)
fig.colorbar(cax,label='3D power')
plt.savefig('posgridsDivided_Npix64_15022024.png')
plt.show()

#Very intuitive variable name
divided_grids_divided = (rsdgrid/grid) / (rsdgrid_nbodykit/grid_nbodykit)

fig, ax = plt.subplots(figsize=(8, 6))
ax.set_xlabel(r'$k = \sqrt{k_x^2 + k_y^2}$')
ax.set_ylabel(r'$k_z$')
ax.set_title('Divided grids divided directly from using y3d and y3d.x, Npix=64')
ax.grid(visible=True)
cax = ax.imshow(fullGrid(divided_grids_divided), origin='lower',extent=(-1,1,-1,1),cmap='nipy_spectral',vmin=0.9,vmax=1.1)
fig.colorbar(cax,label='3D power')
plt.savefig('dividedRSDgrids_divided_Npix64_15022024.png')
plt.show()


