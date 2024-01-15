import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
import h5py
from scipy.interpolate import griddata

def convert_to_kxky(k,mu):
    #We use k = sqrt(kx^2 + ky^2) and mu = cos(theta)
    kx = k*mu
    ky = k*np.sqrt(1-mu**2)
    return kx,ky

def filter_coordinates(xy,z):
    #Simple function to remove all the NaN P(k) values from the sample and all the values that somehow have P(k) < 0
    xy_filter = xy[~np.isnan(xy).any(axis=1)]
    z_filter = z[~np.isnan(z)]

    xy_filter = xy_filter[z_filter > 0.]
    z_filter = z_filter[z_filter > 0.]
    return xy_filter,z_filter

def obtain_coordinates(k,Pk,mu):
    #This function computes (x,y,z) = (kx,ky,Pk) coordinates to plot as an RSD signal
    xy_coordinates = np.zeros((1,2))
    z_coordinates = np.zeros(1)
    for i in range(len(mu)):
        mu_current = mu[i]
        index = len(mu)-1
        x,y = convert_to_kxky(k[index-i],mu_current)
        xy = np.column_stack((x,y))
        xy_coordinates = np.concatenate((xy_coordinates,xy))
        z_coordinates = np.concatenate((z_coordinates,Pk[index-i]))

    #Remove the NaN values by default...
    xy_coordinates,z_coordinates = filter_coordinates(xy_coordinates[1:],z_coordinates[1:])
    return xy_coordinates,z_coordinates

k_val_2D = np.load('2Dpower_k.npy')
Pk_val_2D = np.load('2Dpower_pk.npy')
mu_val_2D = np.load('2Dpower_mu.npy')

def create_grid():
    num_pixels = 1000

    x_values = np.linspace(0,1,num_pixels)
    y_values = np.linspace(0,1,num_pixels)
    #The required mu values range from mu = 1 (theta = 0) to mu = 0 (theta = 1)

    xv,yv = np.meshgrid(x_values,y_values)
    print(xv.shape)
    k_values = np.zeros(xv.shape)
    print(k_values.shape)
    for i in range(len(xv[0])):
        for j in range(len(xv[:,0])):
            k_magnitude = np.sqrt(xv[i,j]**2 + yv[i,j]**2)
            k_values[i,j] = k_magnitude
    
    plt.plot(xv, yv, marker='o', color='k', linestyle='none',markersize=2)
    plt.gca().set_aspect('equal')
    plt.close()

    coordinates,z_values = obtain_coordinates(k_val_2D,Pk_val_2D,mu_val_2D)
    grid_z = griddata(coordinates,z_values,(xv, yv), method='linear')
    print(grid_z.shape)

    fig, ax = plt.subplots(figsize=(8, 6))
    cax = ax.imshow(grid_z, extent=(0, 1, 0, 1), origin='lower', cmap='hsv',norm=matplotlib.colors.LogNorm())
    ax.set_title(r'Redshift-space Distortion, z=0, $k_{max}$ ~ 1.63, $N_{\mu}$ = 101')
    fig.colorbar(cax,label='P(k) value')
    ax.set_xlabel(r'$k_x$')
    ax.set_ylabel(r'$k_y$')
    ax.grid(visible=True)
    fig.savefig('2Drsd_linear_15012024.png')
    plt.close()

    return grid_z

grid_z_2D = create_grid()

########################################################
#Now, for the power spectra 1D data: 
########################################################

def create_arc(r,N_interp):
    angle_values = np.linspace(0,0.5*np.pi,N_interp)
    lin_xvalues = r * np.cos(angle_values)
    lin_yvalues = r * np.sin(angle_values)
    return lin_xvalues,lin_yvalues

def create_1Dgrid(r,z,N_interp):
    N_k = len(r)
    grid_coord = np.zeros((N_k,N_interp,2)) #Create an array to fill with k_x,k_y,Pk coordinates
    print(grid_coord)
    for k in range(N_k):
        x_val,y_val = create_arc(r[k],N_interp)
        grid_coord[k,:,0] = x_val
        grid_coord[k,:,1] = y_val
    return grid_coord

#Load in the 1D data
k_val = np.load('1Dpower_k.npy')
Pk_val = np.load('1Dpower_pk.npy')

#Filter out the NaN values from the data 
k_val_nonan = k_val[~np.isnan(k_val)]
Pk_val_nonan = Pk_val[~np.isnan(k_val)]

#Plot the 1D values over a 2D grid using arcs 
kxky_1D = create_1Dgrid(k_val_nonan,Pk_val_nonan,100)
print(kxky_1D.shape)

plt.scatter(kxky_1D[:,:,0],kxky_1D[:,:,1],s=3)
plt.xlim([0,1])
plt.ylim([0,1])
plt.gca().set_aspect('equal')
plt.savefig('1Dgrid_15012024.png')
plt.close()

xv,yv = np.meshgrid(np.linspace(0,1,1000),np.linspace(0,1,1000))

xy_coords1D_reshape = kxky_1D.reshape(-1,2)
pkVAL_1Dnonan_repeat = np.repeat(Pk_val_nonan,100)
print(xy_coords1D_reshape.shape)
print(pkVAL_1Dnonan_repeat.shape)
grid_z = griddata(xy_coords1D_reshape,pkVAL_1Dnonan_repeat,(xv, yv), method='linear')
print(grid_z.shape)

fig, ax = plt.subplots(figsize=(8, 6))
cax = ax.imshow(grid_z_2D/grid_z, extent=(0, 1, 0, 1), origin='lower', cmap='hsv')
ax.set_title(r'Redshift-space Distortion, z=0, $k_{max}$ ~ 1.63, 1D power spectrum')
fig.colorbar(cax,label='P(k) value')
ax.set_xlabel(r'$k_x$')
ax.set_ylabel(r'$k_y$')
ax.grid(visible=True)
fig.savefig('RSDdivided_15012024.png')
plt.close()