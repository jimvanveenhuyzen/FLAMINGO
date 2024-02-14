import numpy as np
import matplotlib.pyplot as plt
from nbodykit.source.catalog import ArrayCatalog
from nbodykit.lab import *
from nbodykit import setup_logging, style
import dask.array as da
import h5py
import sys

def plot_positions(positions,rsd_positions):

    #Create a filter so that we dont have to plot all halos (very expensive)
    condition_z = (positions[:, 2] > 510) & (positions[:, 2] < 610)
    condition_y = (positions[:, 1] > 300) & (positions[:, 1] < 470)
    condition_x = positions[:, 0] < 20

    pos = positions[condition_x & condition_y & condition_z]
    rsdpos = rsd_positions[condition_x & condition_y & condition_z]

    plt.scatter(pos[:,1],pos[:,2],s=3,color='black',label='Positions',zorder=5)
    plt.scatter(rsdpos[:,1],rsdpos[:,2],s=3,color='firebrick',label='RSDPositions',zorder=10)
    count1 = 0
    count2 = 0
    for i in range(len(pos[:,1])):
        if pos[i,2] < rsdpos[i,2]: 
            if count1 == 0:
                plt.plot([rsdpos[i,1],rsdpos[i,1]],[pos[i,2],rsdpos[i,2]],color='darkblue',linewidth=1.,zorder=1,label='Shift upward')
                count1 += 1 
            else:
                plt.plot([rsdpos[i,1],rsdpos[i,1]],[pos[i,2],rsdpos[i,2]],color='darkblue',linewidth=1.,zorder=1)
        else:
            if count2 == 0:
                plt.plot([rsdpos[i,1],rsdpos[i,1]],[pos[i,2],rsdpos[i,2]],color='darkgreen',linewidth=1.,zorder=1,label='Shift downward')
                count2 += 1 
            else: 
                plt.plot([rsdpos[i,1],rsdpos[i,1]],[pos[i,2],rsdpos[i,2]],color='darkgreen',linewidth=1.,zorder=1)
    plt.grid()
    #plt.xlim([100,110])
    plt.ylim([500,640])
    plt.xlabel('y [Mpc]')
    plt.ylabel('z [Mpc]')
    plt.tick_params(axis='both', which='both', direction='in', bottom=True, top=True, left=True, right=True)
    plt.legend(loc='upper right',ncol=2)
    plt.title('Clustering of galaxies due to the RSD effect along z-axis')
    plt.savefig('RSDeffect_cluster.png')
    plt.close()

    #Play around with these masks to find a proper cluster to plot: 
    condition_x_v2 = positions[:,0] < 10
    condition_y_v2 = (positions[:,1] < 1000) & (positions[:,1] > 0)
    condition_z_v2 = (positions[:,2] < 1000) & (positions[:,2] > 0)
    pos_v2 = positions[condition_x_v2 & condition_y_v2 & condition_z_v2]
    rsd_v2 = rsd_positions[condition_x_v2 & condition_y_v2 & condition_z_v2]

    plt.scatter(pos_v2[:,1],pos_v2[:,2],s=1,color='black',label='Positions',zorder=20)
    plt.grid()
    #plt.xlim([100,130])
    #plt.ylim([110,155])
    plt.xlabel('y [Mpc]')
    plt.ylabel('z [Mpc]')
    plt.legend(loc='upper right')
    plt.title('(y,z) coordinates of the galaxies')
    plt.savefig('full_posMap.png')
    plt.close()

    plt.scatter(rsd_v2[:,1],rsd_v2[:,2],s=1,color='black',label='RSDPositions',zorder=10)
    plt.grid()
    #plt.xlim([100,130])
    #plt.ylim([110,155])
    plt.xlabel('y [Mpc]')
    plt.ylabel('z [Mpc]')
    plt.legend(loc='upper right')
    plt.title('(y,z) coordinates of the galaxies')
    plt.savefig('full_rsdMap.png')
    plt.close()

def Hubble(a,h,omega_m,omega_lambda):
    Hubble_0 = h*100
    Hubble_a = Hubble_0*np.sqrt(omega_m * a**(-3) + omega_lambda)
    return np.round(Hubble_a,3)

h = 0.681
d3a_omegaM = 0.306
d3a_omegaLambda = 0.694
H_a0 = Hubble(1,h,d3a_omegaM,d3a_omegaLambda)
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

def genPower_1D(position,mass,velocity,Nmesh,dk,kmin,show_data):
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

    #Next, we want to add the Redshift Space Distortion effect
    #ADJUST THIS
    line_of_sight = [0,0,1]
    z = 0
    RSDPosition = add_RSD(cat,z,line_of_sight)
    cat['RSDPosition'] = RSDPosition

    #print(cat['Position'].compute())
    #print(cat['RSDPosition'].compute())

    #plot_positions(cat['Position'].compute(),cat['RSDPosition'].compute())

    #convert to a MeshSource, using TSC interpolation on Nmesh^3 
    mesh = cat.to_mesh(window='tsc', Nmesh=Nmesh, compensated=True, position='RSDPosition', BoxSize=[1000,1000,1000])
    
    #compute the fast-fourier transform using linear binning & obtain the power
    fft = FFTPower(mesh,mode='1d',dk=dk,kmin=kmin) 
    Pk = fft.power 

    if show_data == True: 
        # print out the meta-data
        for k in Pk.attrs:
            print("%s = %s" %(k, str(Pk.attrs[k])))

    return Pk['k'], Pk['power'].real, Pk.attrs['shotnoise']

#####################################################################
#From this part, we read in the data and save the FFT power results!
#####################################################################


galaxy_filterVel = np.load('/net/draco/data2/vanveenhuyzen/rsd_project/particle_data/galaxy_filterVel.npy')
galaxy_filterMass = np.load('/net/draco/data2/vanveenhuyzen/rsd_project/particle_data/galaxy_filterMass.npy')
galaxy_filterPos = np.load('/net/draco/data2/vanveenhuyzen/rsd_project/particle_data/galaxy_filterPos.npy')

k_1D, Pk_1D, shotnoise_1D = genPower_1D(galaxy_filterPos,galaxy_filterMass,galaxy_filterVel,512,0.001,0.001,False)

Pk_noSN_1D = Pk_1D-shotnoise_1D
#print(np.shape(Pk_noSN_1D))
#print(k_1D)
#print(Pk_noSN_1D)

print('Saving the 1D data...')
np.save('1Dpower_k.npy',k_1D)
np.save('1Dpower_pk.npy',Pk_noSN_1D)
print('Done saving!')
