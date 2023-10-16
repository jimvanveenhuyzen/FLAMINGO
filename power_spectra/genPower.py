import numpy as np
import matplotlib.pyplot as plt
from nbodykit.source.catalog import ArrayCatalog
from nbodykit.lab import *
from nbodykit import setup_logging, style

def genPower_1D(position_file,Nmesh,dk,kmin,show_data):
    """
    This function generates a 1-dimensional power spectrum from an input vector containing particle positions
    """

    positions = np.genfromtxt('/net/draco/data2/vanveenhuyzen/rsd_project/particle_data/' + position_file)

    length = len(positions[:,0])
    data = np.empty(length, dtype=[('Position', ('f8', 3))])
    data['Position'] = positions
    print('We use {} particle positions.'.format(length))

    #create an array catalog
    
    cat = ArrayCatalog(data)
    print("The columns are ", cat.columns)
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

#calculate k, P(k) and the noise for all positional data
k1,power1,shotnoise1 = genPower_1D('haloPos_0.txt',1024,0.01,0.001,False)
k2,power2,shotnoise2 = genPower_1D('haloPos_0_25.txt',1024,0.01,0.001,False)
k3,power3,shotnoise3 = genPower_1D('haloPos_0_500.txt',1024,0.01,0.001,False)
k4,power4,shotnoise4 = genPower_1D('haloPos_0_1000.txt',1024,0.01,0.001,False)

#plot the power spectra of the data
plt.plot(k1, power1-shotnoise1,label='Halo 0')
plt.plot(k2, power2-shotnoise2,label='Halo 0 to 24')
plt.plot(k3, power3-shotnoise3,label='Halo 0 to 499')
plt.plot(k4, power4-shotnoise4,label='Halo 0 to 999, 218 galaxies')
# format the axes
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r"$k$ [$h \ \mathrm{Mpc}^{-1}$]")
plt.ylabel(r"$P(k)$ [$h^{-3}\mathrm{Mpc}^3$]")
plt.title('Power spectrum P(k) for various stellar positions')
plt.legend(loc='upper right')
plt.savefig('comparePower_halo0to999.png')
plt.show()
