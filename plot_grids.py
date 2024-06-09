import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.gridspec import GridSpec

def fullGrid(grid):
    mirrorX = np.flip(grid,axis=0)
    mirrorY = np.flip(grid,axis=1)
    flip = np.flip(grid)
    top = np.concatenate((flip,mirrorX),axis=1)
    bottom = np.concatenate((mirrorY,grid),axis=1)
    total = np.concatenate((top,bottom),axis=0)
    return total

def load_files(file):
    path = '/net/draco/data2/vanveenhuyzen/rsd_project/nbodykit_sourcecode/griddata/'
    m_pos = np.load(path+'grid_pos_{}.npy'.format(file))
    m_rsd = np.load(path+'grid_rsd_{}.npy'.format(file))
    return m_pos,m_rsd,(m_rsd/m_pos)

#Load in the data: 
strongestlowest_pos,strongestlowest_rsd,strongestlowest_div = load_files('STRONGESTlowest')
strongestlow_pos,strongestlow_rsd,strongestlow_div = load_files('STRONGESTlow')
strongestupper_pos,strongestupper_rsd,strongestupper_div = load_files('STRONGESTupper')

strongest_pos,strongest_rsd,strongest_div = load_files('STRONGEST_AGN')


mlowest_pos,mlowest_rsd,mlowest_div = load_files('mlowestLog')
mlow_pos,mlow_rsd,mlow_div = load_files('mlowLog')
mupper_pos,mupper_rsd,mupper_div = load_files('mupperLog')

mall_pos,mall_rsd,mall_div = load_files('mall')

z0_5_pos,z0_5_rsd,z0_5_div = load_files('z0_5')
z1_pos,z1_rsd,z1_div = load_files('z1')
z2_pos,z2_rsd,z2_div = load_files('z2')

fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(16, 8))

# Plot the first subplot
cax1 = ax0.imshow(mall_pos, origin='lower', extent=(0, 1, 0, 1), cmap='nipy_spectral', norm=matplotlib.colors.LogNorm(vmin=50, vmax=1.1e5))
ax0.set_xlabel(r'$k_{\perp} = \sqrt{k_x^2 + k_y^2}$', fontsize=20)
ax0.set_ylabel(r'$k_{\parallel} = k_z$', fontsize=20)
ax0.tick_params(axis='both',which='major',labelsize=20)
ax0.set_title('Real space',fontsize=20)
ax0.grid(visible=True)

# Plot the second subplot
cax2 = ax1.imshow(mall_rsd, origin='lower', extent=(0, 1, 0, 1), cmap='nipy_spectral', norm=matplotlib.colors.LogNorm(vmin=50, vmax=1.1e5))
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

plt.savefig('mposmrsd_05062024.png')
#plt.show()
plt.close()


fig, ax = plt.subplots(figsize=(16,8))
ax.set_xlabel(r'$k_{\perp} = \sqrt{k_x^2 + k_y^2}$', fontsize=20)
ax.set_ylabel(r'$k_{\parallel} = k_z$', fontsize=20)
ax.set_title(r'$P_{RSD}/P_{Real}$',fontsize=20)
ax.tick_params(which='major',labelsize=12)
ax.grid(visible=True)
ax.set_aspect('equal')
cax = ax.imshow(fullGrid(mall_div), origin='lower',extent=(-1,1,-1,1),cmap='nipy_spectral',vmin=0.5,vmax=2.5)

cbar = fig.colorbar(cax)
cbar.ax.tick_params(labelsize=20)
plt.savefig('mall_06062024.png')
#plt.show()
plt.close()

##############2 by 2 massbins plot 

#fig, axs = plt.subplots(2, 2, figsize=(16, 8), constrained_layout=True)
#fig.subplots_adjust(right=0.85, wspace=0.1)  # Make room for the colorbar

fig = plt.figure(figsize=(16, 8))
#gs = GridSpec(2, 2, width_ratios=[1, 1], wspace=0.0, hspace=0.3) 

# Define the positions of the subplots
positions = [
    [0.2, 0.55, 0.38, 0.4],  # Top left
    [0.42, 0.55, 0.38, 0.4],  # Top right
    [0.2, 0.115, 0.38, 0.4],    # Bottom left
    [0.42, 0.115, 0.38, 0.4]    # Bottom right
]

# Titles and data for each subplot
titles = ['All galaxies', 'Low-mass galaxies', 'Mid-mass galaxies', 'High-mass galaxies']
data = [mall_div, mlowest_div, mlow_div, mupper_div]


# Iterate over subplots and add the image to each
for i in range(4):
    ax = fig.add_axes(positions[i])
    cax = ax.imshow(fullGrid(data[i]), origin='lower', extent=(-1, 1, -1, 1),
                    cmap='nipy_spectral', vmin=0.5, vmax=2.5)
    ax.set_title(titles[i], fontsize=14)
    ax.set_aspect('equal')
    ax.grid(visible=True)
    ax.tick_params(which='major', labelsize=12)

    if i % 2 == 0:
        ax.set_ylabel(r'$k_{\parallel} = k_z$', fontsize=20)
        # Set custom y-ticks and labels, skipping some ticks
        y_ticks = np.linspace(-1, 1, 9)  # Example: 5 ticks
        y_tick_labels = ['{:.1f}'.format(t) for j, t in enumerate(y_ticks) if j % 2 == 0]
        y_ticks = [t for j, t in enumerate(y_ticks) if j % 2 == 0]
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_tick_labels)
    
    if i == 1 or i ==3:
        # Set custom y-ticks and labels, skipping some ticks
        y_ticks = np.linspace(-1, 1, 9)  # Example: 5 ticks
        y_tick_labels = ['' for j, t in enumerate(y_ticks) if j % 2 == 0]
        y_ticks = [t for j, t in enumerate(y_ticks) if j % 2 == 0]
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_tick_labels)


    if i >= 2:
        ax.set_xlabel(r'$k_{\perp} = \sqrt{k_x^2 + k_y^2}$', fontsize=20)

    # # Set x and y labels only for the appropriate subplots
    # if i % 2 == 0:
    #     ax.set_ylabel(r'$k_{\parallel} = k_z$', fontsize=20)
    # if i >= 2:
    #     ax.set_xlabel(r'$k_{\perp} = \sqrt{k_x^2 + k_y^2}$', fontsize=20)

    # Disable specific tick numbers
    if i == 1:
        ax.set_yticklabels([])
        ax.set_xticklabels([])
    if i == 0:
        ax.set_xticklabels([])
        #ax.set_yticklabels([-1,-0.5,0,0.5,1])
    #if i == 2: 
        #ax.set_yticklabels([-1,-0.5,0,0.5,1])
    if i == 3:
        ax.set_yticklabels([])

# Add a single colorbar to the right of the plots
cbar_ax = fig.add_axes([0.75, 0.12, 0.03, 0.83])
cbar = fig.colorbar(cax, cax=cbar_ax)
cbar.ax.tick_params(labelsize=20)

#fig.suptitle(r'$P_{RSD}/P_{Real}$', fontsize=20, y=0.98)  # Adjust y as needed

plt.savefig('massbins.png')
plt.close()

##############2 by 2 redshift plot 

fig = plt.figure(figsize=(16, 8))

# Define the positions of the subplots
positions = [
    [0.2, 0.55, 0.38, 0.4],  # Top left
    [0.42, 0.55, 0.38, 0.4],  # Top right
    [0.2, 0.115, 0.38, 0.4],    # Bottom left
    [0.42, 0.115, 0.38, 0.4]    # Bottom right
]

# Titles and data for each subplot
titles_z = ['z=0', 'z=0.5', 'z=1', 'z=2']
data_z = [mall_div, z0_5_div, z1_div, z2_div]


# Iterate over subplots and add the image to each
for i in range(4):
    ax = fig.add_axes(positions[i])
    cax = ax.imshow(fullGrid(data_z[i]), origin='lower', extent=(-1, 1, -1, 1),
                    cmap='nipy_spectral', vmin=0.5, vmax=2.5)
    ax.set_title(titles_z[i], fontsize=14)
    ax.set_aspect('equal')
    ax.grid(visible=True)
    ax.tick_params(which='major', labelsize=12)

    if i % 2 == 0:
        ax.set_ylabel(r'$k_{\parallel} = k_z$', fontsize=20)
        # Set custom y-ticks and labels, skipping some ticks
        y_ticks = np.linspace(-1, 1, 9)  # Example: 5 ticks
        y_tick_labels = ['{:.1f}'.format(t) for j, t in enumerate(y_ticks) if j % 2 == 0]
        y_ticks = [t for j, t in enumerate(y_ticks) if j % 2 == 0]
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_tick_labels)
    
    if i == 1 or i ==3:
        # Set custom y-ticks and labels, skipping some ticks
        y_ticks = np.linspace(-1, 1, 9)  # Example: 5 ticks
        y_tick_labels = ['' for j, t in enumerate(y_ticks) if j % 2 == 0]
        y_ticks = [t for j, t in enumerate(y_ticks) if j % 2 == 0]
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_tick_labels)

    if i >= 2:
        ax.set_xlabel(r'$k_{\perp} = \sqrt{k_x^2 + k_y^2}$', fontsize=20)
    # Disable specific tick numbers
    if i == 1:
        ax.set_yticklabels([])
        ax.set_xticklabels([])
    if i == 0:
        ax.set_xticklabels([])
    if i == 3:
        ax.set_yticklabels([])

# Add a single colorbar to the right of the plots
cbar_ax = fig.add_axes([0.75, 0.12, 0.03, 0.83])
cbar = fig.colorbar(cax, cax=cbar_ax)
cbar.ax.tick_params(labelsize=20)

plt.savefig('redshiftbins.png')
plt.close()

##############2 by 2 massbins plot 

#fig, axs = plt.subplots(2, 2, figsize=(16, 8), constrained_layout=True)
#fig.subplots_adjust(right=0.85, wspace=0.1)  # Make room for the colorbar

fig = plt.figure(figsize=(16, 8))
#gs = GridSpec(2, 2, width_ratios=[1, 1], wspace=0.0, hspace=0.3) 

# Define the positions of the subplots
positions = [
    [0.2, 0.55, 0.38, 0.4],  # Top left
    [0.42, 0.55, 0.38, 0.4],  # Top right
    [0.2, 0.115, 0.38, 0.4],    # Bottom left
    [0.42, 0.115, 0.38, 0.4]    # Bottom right
]

# Titles and data for each subplot
titles_strongest = ['All galaxies', 'Low-mass galaxies', 'Mid-mass galaxies', 'High-mass galaxies']
data_strongest = [strongest_div, strongestlowest_div, strongestlow_div, strongestupper_div]


# Iterate over subplots and add the image to each
for i in range(4):
    ax = fig.add_axes(positions[i])
    cax = ax.imshow(fullGrid(data_strongest[i]), origin='lower', extent=(-1, 1, -1, 1),
                    cmap='nipy_spectral', vmin=0.5, vmax=2.5)
    ax.set_title(titles_strongest[i], fontsize=14)
    ax.set_aspect('equal')
    ax.grid(visible=True)
    ax.tick_params(which='major', labelsize=12)

    if i % 2 == 0:
        ax.set_ylabel(r'$k_{\parallel} = k_z$', fontsize=20)
        # Set custom y-ticks and labels, skipping some ticks
        y_ticks = np.linspace(-1, 1, 9)  # Example: 5 ticks
        y_tick_labels = ['{:.1f}'.format(t) for j, t in enumerate(y_ticks) if j % 2 == 0]
        y_ticks = [t for j, t in enumerate(y_ticks) if j % 2 == 0]
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_tick_labels)

    if i == 1 or i ==3:
        # Set custom y-ticks and labels, skipping some ticks
        y_ticks = np.linspace(-1, 1, 9)  # Example: 5 ticks
        y_tick_labels = ['' for j, t in enumerate(y_ticks) if j % 2 == 0]
        y_ticks = [t for j, t in enumerate(y_ticks) if j % 2 == 0]
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_tick_labels)

    if i >= 2:
        ax.set_xlabel(r'$k_{\perp} = \sqrt{k_x^2 + k_y^2}$', fontsize=20)

    # # Set x and y labels only for the appropriate subplots
    # if i % 2 == 0:
    #     ax.set_ylabel(r'$k_{\parallel} = k_z$', fontsize=20)
    # if i >= 2:
    #     ax.set_xlabel(r'$k_{\perp} = \sqrt{k_x^2 + k_y^2}$', fontsize=20)

    # Disable specific tick numbers
    if i == 1:
        ax.set_yticklabels([])
        ax.set_xticklabels([])
    if i == 0:
        ax.set_xticklabels([])
        #ax.set_yticklabels([-1,-0.5,0,0.5,1])
    #if i == 2: 
        #ax.set_yticklabels([-1,-0.5,0,0.5,1])
    if i == 3:
        ax.set_yticklabels([])

# Add a single colorbar to the right of the plots
cbar_ax = fig.add_axes([0.75, 0.12, 0.03, 0.83])
cbar = fig.colorbar(cax, cax=cbar_ax)
cbar.ax.tick_params(labelsize=20)

#fig.suptitle(r'$P_{RSD}/P_{Real}$', fontsize=20, y=0.98)  # Adjust y as needed

plt.savefig('massbins_STRONGEST_AGN.png')
plt.close()