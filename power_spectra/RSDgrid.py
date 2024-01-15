import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
import h5py
from scipy.interpolate import griddata

#k_val = np.load('k_val_avg.npy')
#Pk_val = np.load('Pk_val_avg.npy')
#mu_val = np.load('mu_val_avg.npy')
#print(Pk_val[0])

k_val = np.load('kVAL_101.npy')
Pk_val = np.load('pkVAL_101.npy')
mu_val = np.load('muVAL_101.npy')

def convert_to_kxky1(k,mu):
    #We use k = sqrt(kx^2 + ky^2) and mu = cos(theta)
    theta = np.arccos(mu)
    kx = np.cos(theta) * k
    ky = np.sin(theta) * k
    return kx,ky

def convert_to_kxky(k,mu):
    #We use k = sqrt(kx^2 + ky^2) and mu = cos(theta)
    kx = k*mu
    ky = k*np.sqrt(1-mu**2)
    return kx,ky

def filter_coordinates(xy,z):
    #Simple function to remove all the NaN P(k) values from the sample
    xy_filter = xy[~np.isnan(xy).any(axis=1)]
    z_filter = z[~np.isnan(z)]

    xy_filter = xy_filter[z_filter > 0.]
    z_filter = z_filter[z_filter > 0.]
    return xy_filter,z_filter

def obtain_coordinates(k,Pk,mu,filter=True):
    #This function computes (x,y,z) = (kx,ky,Pk) coordinates to plot as an RSD signal
    xy_coordinates = np.zeros((1,2))
    z_coordinates = np.zeros(1)
    for i in range(len(mu)):
        mu_current = mu[i]
        index = len(mu)-1
        x,y = convert_to_kxky(k[index-i],mu_current)
        #print('print k_y as example\n',y)
        xy = np.column_stack((x,y))
        xy_coordinates = np.concatenate((xy_coordinates,xy))
        z_coordinates = np.concatenate((z_coordinates,Pk[index-i]))

    #Remove the NaN values by default...
    if filter == True:
        xy_coordinates,z_coordinates = filter_coordinates(xy_coordinates[1:],z_coordinates[1:])
        return xy_coordinates,z_coordinates
    else:
        return xy_coordinates[1:],z_coordinates[1:]

test1_nonan,test2_nonan = obtain_coordinates(k_val,Pk_val,mu_val)
print(test1_nonan.shape)
print(test2_nonan.shape)
print(np.max(test1_nonan))

def create_grid():
    num_pixels = 640

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

    coordinates = test1_nonan
    z_values = test2_nonan
    #print('shape of coordinates',coordinates.shape)
    #print('shape of the Pk values',z_values.shape)
    coordinates,z_values = obtain_coordinates(k_val,Pk_val,mu_val)
    grid_z = griddata(coordinates,z_values,(xv, yv), method='nearest')
    print(grid_z.shape)

    fig, ax = plt.subplots(figsize=(8, 6))
    cax = ax.imshow(grid_z, extent=(0, 1, 0, 1), origin='lower', cmap='hsv',norm=matplotlib.colors.LogNorm())
    #cax = ax.imshow([coordinates,z_values], extent=(0, 1, 0, 1), origin='lower', cmap='viridis',norm=matplotlib.colors.LogNorm())
    ax.set_title(r'Redshift-space Distortion, z=0, $k_{max}$ ~ 1.63, $N_{\mu}$ = 6400')
    fig.colorbar(cax,label='P(k) value')
    ax.set_xlabel(r'$k_x$')
    ax.set_ylabel(r'$k_y$')
    ax.grid(visible=True)
    fig.savefig('rsd_16122023_linear.png')
    plt.close()

    return grid_z

grid_z_2D = create_grid()

print(np.min(test1_nonan[:,0]))
print(np.min(test1_nonan[:,1]))

#Now plot the 1D RSD grid:

kVAL_1D = np.load('kVAL1D_51.npy')
pkVAL_1D = np.load('pkVAL1D_51.npy')

print(kVAL_1D.shape)

def create_arc(r,N_interp):
    #print('the value of k is',r)
    angle_values = np.linspace(0,0.5*np.pi,N_interp)
    lin_xvalues = r * np.cos(angle_values)
    lin_yvalues = r * np.sin(angle_values)

    #plt.scatter(lin_xvalues,lin_yvalues)
    #plt.savefig('1dPS_testing18122023.png')
    #plt.close()
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

kVAL_1Dnonan = kVAL_1D[~np.isnan(kVAL_1D)]
pkVAL_1Dnonan = pkVAL_1D[~np.isnan(kVAL_1D)]

xy_coords1D = create_1Dgrid(kVAL_1Dnonan,pkVAL_1Dnonan,100)
print(xy_coords1D.shape)

plt.scatter(xy_coords1D[:,:,0],xy_coords1D[:,:,1],s=3)
plt.xlim([0,1])
plt.ylim([0,1])
plt.gca().set_aspect('equal')
plt.savefig('1dGRID_testing18122023.png')
plt.close()

xv,yv = np.meshgrid(np.linspace(0,1,640),np.linspace(0,1,640))

xy_coords1D_reshape = xy_coords1D.reshape(-1,2)
pkVAL_1Dnonan_repeat = np.repeat(pkVAL_1Dnonan,100)
print(xy_coords1D_reshape.shape)
print(pkVAL_1Dnonan_repeat.shape)

grid_z = griddata(xy_coords1D_reshape,pkVAL_1Dnonan_repeat,(xv, yv), method='nearest')
print(grid_z.shape)

fig, ax = plt.subplots(figsize=(8, 6))
#cax = ax.imshow(grid_z_2D/grid_z, extent=(0, 1, 0, 1), origin='lower', cmap='hsv',norm=matplotlib.colors.LogNorm())
cax = ax.imshow(grid_z_2D/grid_z, extent=(0, 1, 0, 1), origin='lower', cmap='hsv')
ax.set_title(r'Redshift-space Distortion, z=0, $k_{max}$ ~ 1.63, 1D power spectrum')
fig.colorbar(cax,label='P(k) value')
ax.set_xlabel(r'$k_x$')
ax.set_ylabel(r'$k_y$')
ax.grid(visible=True)
fig.savefig('rsd_15012024_divided.png')
plt.close()

print(grid_z_2D.shape)

print(grid_z)
print(grid_z_2D/grid_z)

from scipy.spatial.distance import cdist
print('trying...')

available_x = xy_coords1D_reshape[:,0]
available_y = xy_coords1D_reshape[:,1]
available_z = pkVAL_1Dnonan_repeat

num_val = 100#len(available_z)
x_grid,y_grid = np.meshgrid(np.linspace(0,1,num_val),np.linspace(0,1,num_val))

# Flatten the grid
flat_x = x_grid.flatten()
flat_y = y_grid.flatten()

# Calculate distances between available points and grid points
distances = cdist(np.column_stack((flat_x, flat_y)), np.column_stack((available_x, available_y)))

# Find the nearest available point indices for each grid point
nearest_indices = np.argmin(distances, axis=1)

# Initialize grid with zeros
filled_z = np.zeros_like(x_grid)

# Assign z-values from nearest available points to the grid
for grid_idx, avail_idx in enumerate(nearest_indices):
    filled_z.flat[grid_idx] = available_z[avail_idx]

print(filled_z)
print(filled_z.shape)


fig, ax = plt.subplots(figsize=(8, 6))
cax = ax.imshow(filled_z,extent=(0, 1, 0, 1), origin='lower', cmap='hsv',norm=matplotlib.colors.LogNorm())
fig.colorbar(cax,label='P(k) value')
ax.set_xlabel(r'$k_x$')
ax.set_ylabel(r'$k_y$')
ax.grid(visible=True)
plt.savefig('noInterp1D_18122023.png')
plt.close()

x2D = test1_nonan[:,0]
y2D = test1_nonan[:,1]
z2D = test2_nonan

# Calculate distances between available points and grid points
distances = cdist(np.column_stack((flat_x, flat_y)), np.column_stack((x2D, y2D)))

# Find the nearest available point indices for each grid point
nearest_indices = np.argmin(distances, axis=1)

# Initialize grid with zeros
filled_z2 = np.zeros_like(x_grid)

# Assign z-values from nearest available points to the grid
for grid_idx, avail_idx in enumerate(nearest_indices):
    filled_z2.flat[grid_idx] = z2D[avail_idx]

fig, ax = plt.subplots(figsize=(8, 6))
cax = ax.imshow(filled_z2,extent=(0, 1, 0, 1), origin='lower', cmap='hsv',norm=matplotlib.colors.LogNorm())
fig.colorbar(cax,label='P(k) value')
ax.set_xlabel(r'$k_x$')
ax.set_ylabel(r'$k_y$')
ax.grid(visible=True)
plt.savefig('noInterp2D_15012024.png')
plt.close()

fig, ax = plt.subplots(figsize=(8, 6))
cax = ax.imshow(filled_z2/filled_z,extent=(0, 1, 0, 1), origin='lower', cmap='hsv',norm=matplotlib.colors.LogNorm())
fig.colorbar(cax,label='P(k) value')
ax.set_xlabel(r'$k_x$')
ax.set_ylabel(r'$k_y$')
ax.grid(visible=True)
plt.savefig('rsd_divided_15012024.png')
plt.close()

#indices_x = np.round(available_x * (num_val-1)).astype(int)
#indices_y = np.round(available_y * (num_val-1)).astype(int)

#fill_z = np.zeros_like(xv)
#fill_z[indices_y, indices_x] = available_z
#print(fill_z)
#print(np.shape(fill_z))

print('Trying no interp...')
def fill_grid(k,mu,Pk):

    #First, lets create a grid of 100 by 100 points within the range k_x,k_y = 0-1
    x = np.linspace(0,0.99,100)
    x_gridpoints,y_gridpoints = np.meshgrid(x,x)

    #Try getting a grid of 1D values to divide by... (TESTING)
    grid_z_test = griddata(xy_coords1D_reshape,pkVAL_1Dnonan_repeat,(x_gridpoints,y_gridpoints), method='linear')
    grid_z_test[np.isnan(grid_z_test)] = 1
    print('The values for the 1D PS are:')
    print(np.shape(grid_z_test))
    print(grid_z_test)

    #Create an empty dictionary to fill with various keys that correspond to the grid points
    Pk_values = {}
    for i in range(len(x_gridpoints[0])):
        for j in range(len(x_gridpoints[0])):
            x_coord = np.round(x_gridpoints[i,j],2)
            y_coord = np.round(y_gridpoints[i,j],2)
            Pk_values[(x_coord,y_coord)] = []
    
    #print(Pk_values.keys())
    xy_data,z_data = obtain_coordinates(k_val,Pk_val,mu_val)
    #Filter out the values k_x,k_y > 1, since we do not need them for this grid anyway
    xy_data_filter = xy_data[(xy_data[:,0] < 1.)|(xy_data[:,1] < 1.)]
    z_data_filter = z_data[(xy_data[:,0] < 1.)|(xy_data[:,1] < 1.)]

    k_tuple = []
    for k in range(len(xy_data_filter[:,0])):
        k_coordinates = np.round(xy_data_filter[k,:],2)
        k_tuple.append((k_coordinates[0],k_coordinates[1]))

    count = 0
    for tuple_ in k_tuple:
        count += 1 
        #print(tuple_)
        if tuple_ in Pk_values.keys():
            Pk_values[tuple_].append(z_data_filter[count])
            
    for key,values in Pk_values.items():
        if len(values) > 0:
            Pk_values[key] = np.mean(values)
        else:
            Pk_values[key] = 1e-5

    grid_values = np.array(list(Pk_values.values()))
    grid_values = grid_values.reshape(100,100)

    fig, ax = plt.subplots(figsize=(8, 6))
    cax = ax.imshow(grid_values,extent=(0, 1, 0, 1), origin='lower', cmap='nipy_spectral',norm=matplotlib.colors.LogNorm())
    fig.colorbar(cax,label='P(k) value')
    ax.set_xlabel(r'$k_x$')
    ax.set_ylabel(r'$k_y$')
    ax.grid(visible=True)
    plt.savefig('noInterp2D_15012024.png')
    plt.close()

    print(np.shape(xy_data_filter))
    print(np.shape(z_data_filter))
    
    return 1 

fill_grid(1,1,1)