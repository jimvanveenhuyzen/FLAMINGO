import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.gridspec import GridSpec


# import sys
# sys.path.append('/net/draco/data2/vanveenhuyzen/rsd_project/nbodykit_sourcecode')

# import nbodykit_custom
# print(nbodykit_custom.__file__)

# from nbodykit_custom.source.catalog import ArrayCatalog
# from nbodykit_custom.lab import *
# from nbodykit_custom import setup_logging, style

k_rsd = np.load('/net/draco/data2/vanveenhuyzen/rsd_project/power_spectra/2Dpower_mu101_k.npy')
Pk_rsd = np.load('/net/draco/data2/vanveenhuyzen/rsd_project/power_spectra/2Dpower_mu101_pk.npy')
mu_rsd = np.load('/net/draco/data2/vanveenhuyzen/rsd_project/power_spectra/2Dpower_mu101_mu.npy')

k_real = np.load('/net/draco/data2/vanveenhuyzen/rsd_project/power_spectra/2Dpower_noRSD_mu101_k.npy')
Pk_real = np.load('/net/draco/data2/vanveenhuyzen/rsd_project/power_spectra/2Dpower_noRSD_mu101_pk.npy')
mu_real = np.load('/net/draco/data2/vanveenhuyzen/rsd_project/power_spectra/2Dpower_noRSD_mu101_mu.npy')

kperp_real = k_real * mu_real
kparallel_real = k_real * np.sqrt(1-mu_real**2)

kperp_rsd = k_rsd * mu_rsd
kparallel_rsd = k_rsd * np.sqrt(1-mu_rsd**2)

def basis_custom(kperp,kparallel,p,Npixel):
    """Re-plot the grids using (k,mu,Pk)
    """

    kmax = 1 #The maximum k value we want to display in the image, constant in this project 

    pixel_points = np.linspace(0,kmax-0.01,Npixel)
    grid_fac = np.floor(Npixel*pixel_points/kmax).astype(int)

    kparallel_fac = np.floor(Npixel*kparallel/kmax).astype(int)
    kperp_fac = np.floor(Npixel*kperp/kmax).astype(int)

    grid = np.zeros((Npixel,Npixel))

    for kz_,zidx in enumerate(grid_fac):
        print('Current kz = {0}/{1}'.format(kz_,Npixel))

        for kabs_,kidx in enumerate(grid_fac):

            mask3d = np.where( (kz_ == kperp_fac) & (kabs_ == kparallel_fac) )
            pkcurr = p.real[mask3d]

            if len(pkcurr) > 0:
                grid[zidx,kidx] = np.mean(pkcurr) #I had flipped zidx and kidx originally so some images are swapped (|k| and kz that is)
    
    #The result is an N by N grid of Pk values 
    return grid

real_grid = basis_custom(kperp_real,kparallel_real,Pk_real,64)
rsd_grid = basis_custom(kperp_rsd,kparallel_rsd,Pk_rsd,64)

fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(16, 8))

# Plot the first subplot
cax1 = ax0.imshow(real_grid, origin='lower', extent=(0, 1, 0, 1), cmap='nipy_spectral', norm=matplotlib.colors.LogNorm(vmin=50, vmax=1.1e5))
ax0.set_xlabel(r'$k_{\perp} = \sqrt{k_x^2 + k_y^2}$', fontsize=20)
ax0.set_ylabel(r'$k_{\parallel} = k_z$', fontsize=20)
ax0.tick_params(axis='both',which='major',labelsize=20)
ax0.set_title('Real space',fontsize=20)
ax0.grid(visible=True)

# Plot the second subplot
cax2 = ax1.imshow(rsd_grid, origin='lower', extent=(0, 1, 0, 1), cmap='nipy_spectral', norm=matplotlib.colors.LogNorm(vmin=50, vmax=1.1e5))
ax1.set_xlabel(r'$k_{\perp} = \sqrt{k_x^2 + k_y^2}$', fontsize=20)
#ax1.set_ylabel(r'$k_{\parallel} = k_z$', fontsize=14)
ax1.tick_params(axis='both',which='major',labelsize=20)
ax1.set_title('Redshift space',fontsize=20)
ax1.grid(visible=True)

# Remove y-axis labels for the right subplot
ax1.yaxis.set_major_formatter(plt.NullFormatter())

# Adjust positions to reduce horizontal space
fig.subplots_adjust(wspace=0.10, right=0.85)  # Reduce wspace further and adjust right

# Create a colorbar for both plots
cbar_ax = fig.add_axes([0.88, 0.1, 0.02, 0.8])  # Move colorbar more to the right
cbar = fig.colorbar(cax2, cax=cbar_ax)

cbar.set_label('P(k)', fontsize=20)
cbar.ax.tick_params(labelsize=20)
# Rotate the colorbar label to horizontal
cbar.ax.yaxis.label.set_rotation(0)
cbar.ax.yaxis.label.set_horizontalalignment('center')
cbar.ax.yaxis.set_label_coords(0.5, 1.06)

# Adjust layout
#fig.tight_layout()

plt.savefig('mposrsd_mu_05062024.png')
#plt.show()
plt.close()

