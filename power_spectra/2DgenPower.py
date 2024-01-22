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

def FFT_average(catalogue,Nmesh,dk,kmin,Nmu,z):

    unit_vec = [[1,0,0],[0,1,0],[0,0,1]]
    for j in range(len(unit_vec)):

        #Add the RSD position to the catalogue for the line of sight vector
        RSDPosition = add_RSD(catalogue,z,unit_vec[j])
        catalogue['RSDPosition'] = RSDPosition

        #convert to a MeshSource, using TSC interpolation on Nmesh^3 
        mesh = catalogue.to_mesh(window='tsc', Nmesh=Nmesh, compensated=True, position='RSDPosition', BoxSize=[1000,1000,1000])

        fft = FFTPower(mesh,mode='2d',dk=dk,kmin=kmin,Nmu=Nmu,los=unit_vec[j]) 
        Pkmu = fft.power 
        #Get the number of k values:
        num_k_values = np.shape(Pkmu)[0]
        #Initialize initial matrices
        k_data = np.zeros((Nmu,num_k_values))
        mu_data = np.zeros(Nmu)
        Pk_data = np.copy(k_data)
        SN_data = np.copy(k_data)

        #Initialize the combined arrays for the first case
        if j == 0:
            Pk_data_total = np.zeros((len(unit_vec), int(np.ceil(Nmu/2)), num_k_values))
            SN_data_total = np.zeros((len(unit_vec), int(np.ceil(Nmu/2)), num_k_values))

        #Loop over all the positive (and 0) values of mu
        for i in range(Pkmu.shape[1]):
            #print(i)
            if Pkmu.coords['mu'][i] < 0:
                continue
            else:
                # plot each mu bin
                Pk = Pkmu[:,i] # select the ith mu bin
                k_data[i] = Pk['k']
                Pk_data[i] = Pk['power'].real
                SN_data[i] = Pk.attrs['shotnoise'] 
                mu_data[i] = Pkmu.coords['mu'][i]

        #The mu values used range from -1 to 1, but we want those only from 0 to 1, so we select only the latter half
        k_data = k_data[int(Nmu/2):,:]
        Pk_data = Pk_data[int(Nmu/2):,:]
        SN_data = SN_data[int(Nmu/2):,:]
        mu_data = mu_data[int(Nmu/2):]

        Pk_data_total[j] = Pk_data
        SN_data_total[j] = SN_data

    mean_Pk_data = np.mean(Pk_data_total,axis=0)
    mean_SN_data = np.mean(SN_data_total,axis=0)
    return k_data,mu_data,mean_Pk_data,mean_SN_data

def compute_FFT(catalogue,Nmesh,dk,kmin,Nmu,z,line_of_sight):

    #First we add the RSDPosition to the catalogue
    RSDPosition = add_RSD(catalogue,z,line_of_sight)
    catalogue['RSDPosition'] = RSDPosition

    #convert to a MeshSource, using TSC interpolation on Nmesh^3 
    mesh = catalogue.to_mesh(window='tsc', Nmesh=Nmesh, compensated=True, position='Position', BoxSize=[1000,1000,1000])

    fft = FFTPower(mesh,mode='2d',dk=dk,kmin=kmin,Nmu=Nmu,los=line_of_sight) 
    Pkmu = fft.power 

    k_data = Pkmu['k'][:,int(Nmu/2):]
    Pk_data = Pkmu['power'].real[:,int(Nmu/2):]
    SN_data = Pkmu.attrs['shotnoise']
    mu_data = Pkmu['mu'][:,int(Nmu/2):]
    return k_data,mu_data,Pk_data,SN_data

    #Get the number of k values:
    num_k_values = np.shape(Pkmu)[0]
    #Initialize initial matrices
    k_data = np.zeros((Nmu,num_k_values))
    mu_data = np.zeros(Nmu)
    Pk_data = np.copy(k_data)
    SN_data = np.copy(k_data)

    #Loop over all the positive (and 0) values of mu
    for i in range(Pkmu.shape[1]):
        #print(i)
        if Pkmu.coords['mu'][i] < 0:
            continue
        else:
            # plot each mu bin
            Pk = Pkmu[:,i] # select the ith mu bin
            k_data[i] = Pk['k']
            Pk_data[i] = Pk['power'].real
            SN_data[i] = Pk.attrs['shotnoise'] 
            mu_data[i] = Pkmu.coords['mu'][i]

    #The mu values used range from -1 to 1, but we want those only from 0 to 1, so we select only the latter half
    k_data = k_data[int(Nmu/2):,:]
    Pk_data = Pk_data[int(Nmu/2):,:]
    SN_data = SN_data[int(Nmu/2):,:]
    mu_data = mu_data[int(Nmu/2):]
    return k_data,mu_data,Pk_data,SN_data

def genPower_2D(position,mass,velocity,Nmesh,dk,kmin,Nmu,line_of_sight=[0,1,0]):
    """
    This function generates a 2-dimensional power spectrum from an input vector containing
    particle positions, masses and velocities 
    """
    length = len(position[:,0])
    data = np.empty(length, dtype=[('Position', ('f8', 3)),('Velocity', ('f8', 3)),('Weight', ('f8'))])
    data['Position'] = position
    data['Weight'] = mass
    data['Velocity'] = velocity 
    print('We use {} particle positions.'.format(length))

    #Create an array catalog
    cat = ArrayCatalog(data)
    print("The columns are ", cat.columns)

    #Next, either use the averaged L.o.S. direction or pick a unit vector
    if line_of_sight == None: 
        print('Apply the FFT_average function..')
        k,mu,Pk,shotnoise = FFT_average(cat,Nmesh,dk,kmin,Nmu,0)
    else:
        print('Apply the FFT function along {}'.format(line_of_sight))
        k,mu,Pk,shotnoise = compute_FFT(cat,Nmesh,dk,kmin,Nmu,0,line_of_sight)
    print('The sizes of the data are...')
    print(np.shape(k))
    print(np.shape(mu))
    print(np.shape(Pk))
    print(np.shape(shotnoise))

    print(k[:-20,:])
    print(mu[:-20,:])

    #for i in range(np.shape(mu)[1]):
    #    label = r'$\mu$=%.3f' % (np.nanmean(mu[:,i]))
    #    plt.loglog(k[:,i], Pk[:,i] - shotnoise, label=label)
    #plt.legend(loc='upper right', ncol=1)
    #plt.xlabel(r"$k$ [$h \ \mathrm{Mpc}^{-1}$]")
    #plt.ylabel(r"$P(k, \mu)$ [$h^{-3}\mathrm{Mpc}^3$]")
    #plt.xlim(0.01, 0.8)
    #plt.savefig('ps2D_1701024.png')

    #for i in range(len(mu)):
    #    label = r'$\mu$=%.3f' % (mu[i])
    #    plt.loglog(k[i,:], Pk[i,:] - shotnoise[i,:], label=label)
    #plt.legend(loc='upper right', ncol=1)
    #plt.xlabel(r"$k$ [$h \ \mathrm{Mpc}^{-1}$]")
    #plt.ylabel(r"$P(k, \mu)$ [$h^{-3}\mathrm{Mpc}^3$]")
    #plt.xlim(0.01, 0.8)
    #plt.savefig('ps2D_1501024.png')
    
    return k,mu,Pk,shotnoise

galaxy_filterVel = np.load('/net/draco/data2/vanveenhuyzen/rsd_project/particle_data/galaxy_filterVel.npy')
galaxy_filterMass = np.load('/net/draco/data2/vanveenhuyzen/rsd_project/particle_data/galaxy_filterMass.npy')
galaxy_filterPos = np.load('/net/draco/data2/vanveenhuyzen/rsd_project/particle_data/galaxy_filterPos.npy')

k_array,mu_array,Pk_array,SN_array = genPower_2D(galaxy_filterPos,galaxy_filterMass,galaxy_filterVel,Nmesh=512,dk=0.0005,kmin=0.0001,Nmu=2001)
print(np.shape(k_array.T))
print(np.shape(mu_array))
print(np.shape(Pk_array.T))

Pk_noSN = Pk_array - SN_array

print(np.shape(k_array))
print(np.shape(mu_array))
print(np.shape(Pk_noSN))

#print(mu_array)
#print(k_array)

print('Saving the 2D data...')
np.save('2Dpower_k.npy',k_array)
np.save('2Dpower_pk.npy',Pk_noSN)
np.save('2Dpower_mu.npy',mu_array)
print('Done saving!')