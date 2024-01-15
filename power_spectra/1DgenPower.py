import numpy as np
import matplotlib.pyplot as plt
from nbodykit.source.catalog import ArrayCatalog
from nbodykit.lab import *
from nbodykit import setup_logging, style
import h5py
import sys

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

    #convert to a MeshSource, using TSC interpolation on Nmesh^3 
    mesh = cat.to_mesh(window='tsc', Nmesh=Nmesh, compensated=True, position='Position', BoxSize=[1000,1000,1000])
    
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

k_1D, Pk_1D, shotnoise_1D = genPower_1D(galaxy_filterPos,galaxy_filterMass,galaxy_filterVel,512,0.01,0.001,False)
Pk_noSN_1D = Pk_1D-shotnoise_1D
print(np.shape(Pk_noSN_1D))
print(k_1D)
print(Pk_noSN_1D)

print('Saving the 1D data...')
np.save('1Dpower_k.npy',k_1D)
np.save('1Dpower_pk.npy',Pk_noSN_1D)
print('Done saving!')