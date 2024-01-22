import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors

def convert_to_kxky(k,mu):
    #We use k = sqrt(kx^2 + ky^2) and mu = cos(theta)
    kx = k*mu
    ky = k*np.sqrt(1-mu**2)
    return kx,ky

def filter_coordinates(xy,z):
    #Simple function to remove all the NaN P(k) values from the sample and all the values that somehow have P(k) < 0
    xy_filter = xy[~np.isnan(xy).any(axis=1)]
    z_filter = z[~np.isnan(xy).any(axis=1)]

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
    ax.set_title('RSD Grid with LoS [0,0,1] without interp (factor 3)')
    ax.grid(visible=True)
    #plt.savefig('RSDfull_15012024_factor3.png')
    plt.close()


k_val_2D = np.load('2Dpower_zProj_k.npy')
Pk_val_2D = np.load('2Dpower_zProj_pk.npy')
mu_val_2D = np.load('2Dpower_zProj_mu.npy')

k_val_2D = np.load('2Dpower_k.npy')
Pk_val_2D = np.load('2Dpower_pk.npy')
mu_val_2D = np.load('2Dpower_mu.npy')

def fill_grid(k,mu,Pk):

    #First, lets create a grid of 100 by 100 points within the range k_x,k_y = 0-1
    x = np.linspace(0,99,100,dtype=np.int32)
    x_gridpoints,y_gridpoints = np.meshgrid(x,x)

    #Create an empty dictionary to fill with various keys that correspond to the grid points
    #In our case, we get (0,0), (0.0,0.01), ... , (1.0,0.99), (1.0,1.0)
    Pk_values = {}
    for i in range(len(x_gridpoints[0])):
        for j in range(len(x_gridpoints[0])):
            #x_coord = np.round(x_gridpoints[i,j],2)
            #y_coord = np.round(y_gridpoints[i,j],2)
            #Pk_values[(x_coord,y_coord)] = []
            Pk_values[(x_gridpoints[i,j],y_gridpoints[i,j])] = []
    
    #print(Pk_values.keys()) #These are all the keys which are the coordinates 
            
    xy_data,z_data = obtain_coordinates(k_val_2D,Pk_val_2D,mu_val_2D)
    #Filter out the values k_x,k_y > 1, since we do not need them for this grid anyway
    xy_data_filter = xy_data[(xy_data[:,0] < 1.)&(xy_data[:,1] < 1.)]
    z_data_filter = z_data[(xy_data[:,0] < 1.)&(xy_data[:,1] < 1.)]
    print('Shape of kxky and Pk for 1D')
    print(np.shape(xy_data_filter))
    print(np.shape(z_data_filter))

    #Now, we construct a list of tuples that correspond to coordinates in the data 
    #E.g. if we have a point with (x,y) = (0.1,0.45), we construct a tuple of its coordinates
    k_tuple = []
    for k in range(len(xy_data_filter[:,0])):
        k_coordinates = np.floor(xy_data_filter[k,:] * 100)
        k_tuple.append((k_coordinates[0],k_coordinates[1]))

    #Next is the main step of this process. We add the Pk values at the correct (kx,ky) coordinates by adding it to the corresponding
    #tuple in the dictionary. 
    count = 0
    for tuple_ in k_tuple:
        Pk_values[tuple_].append(z_data_filter[count])
        count += 1 
            
    #Here we check two things: if a key (coordinate) has multiple Pk values associated, we take the mean of these values
    #We also add a 'filler' value to the empty keys (so the grid coordinates without Pk values) so that we can plot it effectively 
    for key,values in Pk_values.items():
        if len(values) > 0:
            Pk_values[key] = np.mean(values)
        else:
            Pk_values[key] = 1e-5

    #Finally, we have a dictionary of length 10000 now, so we convert it back to 100 by 100.
    #This results in a grid of 100 by 100 Pk values that correspond to k_x,k_y values. 
    grid_values = np.array(list(Pk_values.values()))
    grid_values = grid_values.reshape(100,100)

    fig, ax = plt.subplots(figsize=(8, 6))
    cax = ax.imshow(grid_values,extent=(0, 1, 0, 1), origin='lower', cmap='nipy_spectral',norm=matplotlib.colors.LogNorm())
    fig.colorbar(cax,label='P(k) value')
    ax.set_xlabel(r'$k_x$')
    ax.set_ylabel(r'$k_y$')
    ax.set_title('RSD Grid with LoS [0,0,1] without interp')
    ax.grid(visible=True)
    plt.savefig('noInterp2D_17012024.png')
    plt.close()

    print(np.shape(xy_data_filter))
    print(np.shape(z_data_filter))
    
    return grid_values

grid2D = fill_grid(1,1,1)
fullPlot(grid2D)

#############################
#Now, start the 1D part: 
#############################

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
kxky_1D_reshape = kxky_1D.reshape(-1,2)
Pk_val_nonan_reshape = np.repeat(Pk_val_nonan,100)
print('Shape of kxky and Pk for 1D')
print(kxky_1D_reshape.shape)
print(Pk_val_nonan_reshape.shape)

def fill_grid_1D(k,mu,Pk):

    #First, lets create a grid of 100 by 100 points within the range k_x,k_y = 0-1
    x = np.linspace(0,99,100,dtype=np.int32)
    x_gridpoints,y_gridpoints = np.meshgrid(x,x)
    print(x_gridpoints)

    #Create an empty dictionary to fill with various keys that correspond to the grid points
    #In our case, we get (0,0), (0.0,0.01), ... , (1.0,0.99), (1.0,1.0)
    Pk_values = {}
    for i in range(len(x_gridpoints[0])):
        for j in range(len(x_gridpoints[0])):
            #x_coord = np.round(x_gridpoints[i,j],2)
            #y_coord = np.round(y_gridpoints[i,j],2)
            #Pk_values[(x_coord,y_coord)] = []
            Pk_values[(x_gridpoints[i,j],y_gridpoints[i,j])] = []
            
    xy_data,z_data = kxky_1D_reshape, Pk_val_nonan_reshape
    #Filter out the values k_x,k_y > 1, since we do not need them for this grid anyway
    xy_data_filter = xy_data[(xy_data[:,0] < 1.)&(xy_data[:,1] < 1.)]
    z_data_filter = z_data[(xy_data[:,0] < 1.)&(xy_data[:,1] < 1.)]

    #Now, we construct a list of tuples that correspond to coordinates in the data 
    #E.g. if we have a point with (x,y) = (0.1,0.45), we construct a tuple of its coordinates
    k_tuple = []
    for k in range(len(xy_data_filter[:,0])):
        #k_coordinates = np.floor(xy_data_filter[k,:] * 100)/100
        k_coordinates = np.floor(xy_data_filter[k,:] * 100)
        k_tuple.append((k_coordinates[0],k_coordinates[1]))

    #Next is the main step of this process. We add the Pk values at the correct (kx,ky) coordinates by adding it to the corresponding
    #tuple in the dictionary. 
    count = 0
    for tuple_ in k_tuple:
        Pk_values[tuple_].append(z_data_filter[count])
        count += 1 
            
    #Here we check two things: if a key (coordinate) has multiple Pk values associated, we take the mean of these values
    #We also add a 'filler' value to the empty keys (so the grid coordinates without Pk values) so that we can plot it effectively 
    for key,values in Pk_values.items():
        if len(values) > 0:
            Pk_values[key] = np.mean(values)
        else:
            Pk_values[key] = 1e-5

    #Finally, we have a dictionary of length 10000 now, so we convert it back to 100 by 100.
    #This results in a grid of 100 by 100 Pk values that correspond to k_x,k_y values. 
    grid_values = np.array(list(Pk_values.values()))
    grid_values = grid_values.reshape(100,100)

    fig, ax = plt.subplots(figsize=(8, 6))
    cax = ax.imshow(grid_values,extent=(0, 1, 0, 1), origin='lower', cmap='nipy_spectral',norm=matplotlib.colors.LogNorm())
    fig.colorbar(cax,label='P(k) value')
    ax.set_xlabel(r'$k_x$')
    ax.set_ylabel(r'$k_y$')
    ax.grid(visible=True)
    plt.savefig('noInterp1D_17012024.png')
    plt.close()

    print(np.shape(xy_data_filter))
    print(np.shape(z_data_filter))
    
    return grid_values

grid1D = fill_grid_1D(1,1,1)

#fig, ax = plt.subplots(figsize=(8, 6))
#cax = ax.imshow(grid1D/grid2D,extent=(0, 1, 0, 1), origin='lower', cmap='nipy_spectral',norm=matplotlib.colors.LogNorm())
#fig.colorbar(cax,label='P(k) value')
#ax.set_xlabel(r'$k_x$')
#ax.set_ylabel(r'$k_y$')
#ax.grid(visible=True)
#ax.set_title('Divide the 1D and 2D non-interpolated RSD grids over each other')
#plt.savefig('noInterp_Divided_15012024.png')
#plt.close()

def fill1Dgrid(kxky,Pk):

    #First, lets create a grid of 100 by 100 points within the range k_x,k_y = 0-1
    x = np.linspace(0,0.99,100,dtype=np.float32)
    x_gridpoints,y_gridpoints = np.meshgrid(x,x)

    xy_data,z_data = kxky,Pk
    #Filter out the values k_x,k_y > 1, since we do not need them for this grid anyway

    xy_data_filter = xy_data[(xy_data[:,0] < 1.)&(xy_data[:,1] < 1.)]
    z_data_filter = z_data[(xy_data[:,0] < 1.)&(xy_data[:,1] < 1.)]

    k_data_factor100 = np.floor(100*np.sqrt(xy_data_filter[:,0]**2 + xy_data_filter[:,1]**2)).astype(int)

    grid = np.zeros((100,100))
    for i in range(len(x_gridpoints[0])):
        for j in range(len(y_gridpoints[0])):
            k_current = np.floor(100*np.sqrt(x_gridpoints[i,j]**2 + y_gridpoints[i,j]**2)).astype(int)
            mask = np.where(k_current == k_data_factor100)
            Pk_values = z_data_filter[mask]

            if len(Pk_values) > 0: 
                grid[i,j] = Pk_values[0]
            else: 
                grid[i,j] = 1e-5

    #Replace the mising pixel with an adjacant pixel 
    grid[-1,-1] = grid[-1,-2]
    fig, ax = plt.subplots(figsize=(8, 6))
    cax = ax.imshow(grid,extent=(0, 1, 0, 1), origin='lower', cmap='nipy_spectral',norm=matplotlib.colors.LogNorm())
    fig.colorbar(cax,label='P(k) value')
    ax.set_xlabel(r'$k_x$')
    ax.set_ylabel(r'$k_y$')
    ax.grid(visible=True)
    plt.savefig('noInterp1D_alternate_16012024.png')
    plt.close()

fill1Dgrid(kxky_1D_reshape, Pk_val_nonan_reshape)

#This method (obviously) doesnt work for 2D since we have an additional angle of mu 
#kxky2d,pk2d = obtain_coordinates(k_val_2D,Pk_val_2D,mu_val_2D)
#fill1Dgrid(kxky2d,pk2d)




