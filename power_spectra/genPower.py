import numpy as np
import matplotlib.pyplot as plt
from nbodykit.source.catalog import ArrayCatalog
from nbodykit.lab import *
from nbodykit import setup_logging, style
import h5py
import sys

#The simulations (generally) use a D3A cosmology, so define the Hubble constant using this value
h = 0.681 #D3A cosmology 
hubble0 = h*100 #km/s/Mpc

def Hubble(a,h,omega_m,omega_lambda):
    Hubble_0 = h*100
    Hubble_a = Hubble_0*np.sqrt(omega_m * a**(-3) + omega_lambda)
    return Hubble_a

def add_RSD(catalogue,z,line_of_sight):
    scale_factor = 1/(1+z)
    rsd_pos = catalogue['Position'] + (catalogue['Velocity'] * line_of_sight)/(scale_factor*hubble0)
    return rsd_pos

def genPower_1D(position,mass,Nmesh,dk,kmin,show_data):
    """
    This function generates a 1-dimensional power spectrum from an input vector containing particle positions
    """
    length = len(position[:,0])
    data = np.empty(length, dtype=[('Position', ('f8', 3)),('Weight', ('f8',1))])
    data['Position'] = position
    data['Weight'] = mass
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

def FFT_average(catalogue,Nmesh,dk,kmin,Nmu,z):

    unit_vec = [[1,0,0],[0,1,0],[0,0,1]]
    for j in range(len(unit_vec)):

        #Add the RSD position to the catalogue for the line of sight vector
        RSDPosition = add_RSD(catalogue,z,unit_vec[j])
        catalogue['RSDPosition'] = RSDPosition

        #convert to a MeshSource, using TSC interpolation on Nmesh^3 
        mesh = catalogue.to_mesh(window='tsc', Nmesh=Nmesh, compensated=True, position='Position', BoxSize=[1000,1000,1000])

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

def genPower_2D(position,mass,velocity,Nmesh,dk,kmin,Nmu,line_of_sight=None):
    """
    This function generates a 2-dimensional power spectrum from an input vector containing
    particle positions, masses and velocities 
    """
    length = len(position[:,0])
    data = np.empty(length, dtype=[('Position', ('f8', 3)),('Velocity', ('f8', 3)),('Weight', ('f8',1))])
    data['Position'] = position
    data['Weight'] = mass
    data['Velocity'] = velocity 
    print('We use {} particle positions.'.format(length))

    #create an array catalog
    
    cat = ArrayCatalog(data)
    print("The columns are ", cat.columns)
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

    for i in range(len(mu)):
        label = r'$\mu$=%.3f' % (mu[i])
        plt.loglog(k[i,:], Pk[i,:] - shotnoise[i,:], label=label)
    plt.legend(loc='upper right', ncol=1)
    plt.xlabel(r"$k$ [$h \ \mathrm{Mpc}^{-1}$]")
    plt.ylabel(r"$P(k, \mu)$ [$h^{-3}\mathrm{Mpc}^3$]")
    plt.xlim(0.01, 0.8)
    plt.savefig('ps2D_19122023.png')
    
    return k,mu,Pk,shotnoise

    """
    if average_los == True:
        #Next, we add the average RSD effect to the catalogue.
        #Average in the sense that we calculate the RSDPosition for each L.o.S. direction and take the average.
        unit_vectors = [[1,0,0],[0,1,0],[0,0,1]]
        RSDPosition_list = []
        for los in unit_vectors:
            RSDPos = add_RSD(cat,0,los)
            RSDPosition_list.append(RSDPos)
        cat['RSDPosition'] = np.mean(RSDPosition_list,axis=0)
    else:
        #Add the RSD effect along the z-axis (default): 
        line_of_sight = [0,0,1]
        cat['RSDPosition'] = add_RSD(cat,H0,0,line_of_sight)
    #cat['RSDPosition'] = cat['Position'] + cat['Velocity'] * line_of_sight
    #compute the fast-fourier transform using linear binning & obtain the power
    fft = FFTPower(mesh,mode='2d',dk=dk,kmin=kmin,Nmu=101,los=[0,0,1]) 
    Pkmu = fft.power 
    #print('-'*50,'z_axis:\n')
    #print(Pkmu[:,7]['power'].real)
    #print(np.shape(Pkmu))
    print('We use the following mu values:',t2)
    #print(Pkmu)
    # plot each mu bin
    num_k_values = np.shape(Pkmu)[0]
    num_mu_values = np.shape(Pkmu)[1]
    #print('-'*50,'y_axis:\n')
    #print(Pkmu[:,7]['power'].real)

    k_data = np.zeros((num_mu_values,num_k_values))
    Pk_data = np.copy(k_data)
    SN_data = np.copy(k_data)
    mu_data = np.zeros(num_mu_values)
    for i in range(Pkmu.shape[1]):
        print(i)
        if Pkmu.coords['mu'][i] < 0:
            continue
        else:
            # plot each mu bin
            Pk = Pkmu[:,i] # select the ith mu bin
            #print(Pkmu.coords['mu'][i])
            #print(Pk['k'])
            k_data[i] = Pk['k']
            Pk_data[i] = Pk['power'].real
            SN_data[i] = Pk.attrs['shotnoise'] 
            mu_data[i] = Pkmu.coords['mu'][i]
            label = r'$\mu$=%.3f' % (Pkmu.coords['mu'][i])
            plt.loglog(Pk['k'], Pk['power'].real - Pk.attrs['shotnoise'], label=label)

    # format the axes
    #plt.legend(loc=0, ncol=5)
    plt.xlabel(r"$k$ [$h \ \mathrm{Mpc}^{-1}$]")
    plt.ylabel(r"$P(k, \mu)$ [$h^{-3}\mathrm{Mpc}^3$]")
    plt.xlim(0.01, 0.8)
    plt.savefig('ps2D_11122023.png')
    #print(np.shape(k_data))
    #print('-'*100)
    #print(mu_data)
    return k_data[int(num_mu_values/2):,:],Pk_data[int(num_mu_values/2):,:],SN_data[int(num_mu_values/2):,:],mu_data[int(num_mu_values/2):]
    """

#calculate k, P(k) and the noise for all positional data
#k1,power1,shotnoise1 = genPower_1D('txt_files/haloPos_0.txt',1024,0.01,0.001,False)
#k2,power2,shotnoise2 = genPower_1D('txt_files/haloPos_0_25.txt',1024,0.01,0.001,False)
#k3,power3,shotnoise3 = genPower_1D('txt_files/haloPos_0_500.txt',1024,0.01,0.001,False)
#k4,power4,shotnoise4 = genPower_1D('txt_files/haloPos_0_1000.txt',1024,0.01,0.001,False)
#k5,power5,shotnoise5 = genPower_1D('gal_pos_200000.txt',1024,0.01,0.001,False)

SOAP_file = '/net/hypernova/data2/FLAMINGO/L1000N1800/HYDRO_FIDUCIAL/SOAP/halo_properties_0077.hdf5'
with h5py.File(SOAP_file,'r') as soap:
    CentreOfPot = soap['VR/CentreOfPotential'][...] #use this for the average position of the stellar particles per halo! 
soap.close()
galaxy_stellarMass = np.load('/net/draco/data2/vanveenhuyzen/rsd_project/particle_data/galaxy_stellarMass.npy')
galaxy_stellarVel = np.load('/net/draco/data2/vanveenhuyzen/rsd_project/particle_data/galaxy_meanVel.npy')
galaxy_filterVel = np.load('/net/draco/data2/vanveenhuyzen/rsd_project/particle_data/galaxy_filterVel.npy')
galaxy_filterMass = np.load('/net/draco/data2/vanveenhuyzen/rsd_project/particle_data/galaxy_filterMass.npy')
galaxy_filterPos = np.load('/net/draco/data2/vanveenhuyzen/rsd_project/particle_data/galaxy_filterPos.npy')

#print(galaxy_stellarVel[0:100])
#print(galaxy_stellarVel[0:100]/H0)
#print(galaxy_filterPos[0:100])

#The top line is using the unfiltered catalogue 
#k_array,mu_array,Pk_array,SN_array = genPower_2D(CentreOfPot[:-1],galaxy_stellarMass,galaxy_stellarVel,Nmesh=512,dk=0.001,kmin=0.0001,Nmu=201)
k_array,mu_array,Pk_array,SN_array = genPower_2D(galaxy_filterPos,galaxy_filterMass,galaxy_filterVel,Nmesh=512,dk=0.001,kmin=0.0001,Nmu=201)
Pk_noSN = Pk_array-SN_array
#k_1D, Pk_1D, SN_1D = genPower_1D(CentreOfPot[:-1],galaxy_stellarMass,512,0.001,0.0001,False)
k_1D, Pk_1D, SN_1D = genPower_1D(galaxy_filterPos,galaxy_filterMass,512,0.001,0.0001,False)
print(SN_1D)
Pk_noSN_1D = Pk_1D-SN_1D
print(np.shape(k_array))
print(np.shape(k_1D))

print('Saving the 1D data...')
np.save('kVAL1D_51.npy',k_1D)
np.save('pkVAL1D_51.npy',Pk_noSN_1D)
print('Saving the 2D data...')
np.save('kVAL_101.npy',k_array)
np.save('pkVAL_101.npy',Pk_noSN)
np.save('muVAL_101.npy',mu_array)
print('Done saving!')

sys.exit()
#calculate k, P(k) and the noise for all positional data
#k1,power1,shotnoise1 = genPower_1D('txt_files/haloPos_0.txt',1024,0.01,0.001,False)
#k2,power2,shotnoise2 = genPower_1D('txt_files/haloPos_0_25.txt',1024,0.01,0.001,False)
#k3,power3,shotnoise3 = genPower_1D('txt_files/haloPos_0_500.txt',1024,0.01,0.001,False)
#k4,power4,shotnoise4 = genPower_1D('txt_files/haloPos_0_1000.txt',1024,0.01,0.001,False)
#k5,power5,shotnoise5 = genPower_1D('gal_pos_200000.txt',1024,0.01,0.001,False)

k_galAll,power_galAll,shotnoise_galAll = genPower_1D(CentreOfPot[:-1],galaxy_stellarMass,1024,0.01,0.001,False)
k_galFilter,power_galFilter,shotnoise_galFilter = genPower_1D(galaxy_filterPos,galaxy_filterMass,1024,0.01,0.001,False)

def compute_shotnoise(L_x,L_y,L_z,mass):
    """
    Computes the shotnoise of the 1D power spectrum manually. 
    """
    volume = L_x*L_y*L_z
    return volume * np.sum(mass**2)/(np.sum(mass))**2

print('shotnoise of all data:',shotnoise_galAll)
print('shotnoise of filtered data:',shotnoise_galFilter)
print('manually calculated shotnoise of all data:',compute_shotnoise(1000,1000,1000,galaxy_stellarMass))
print('manually calculated shotnoise of filtered data:',compute_shotnoise(1000,1000,1000,galaxy_filterMass))

#plot the power spectra of the data
#plt.plot(k1, power1-shotnoise1,label='Halo 0')
#plt.plot(k2, power2-shotnoise2,label='Halo 0 to 24')
#plt.plot(k3, power3-shotnoise3,label='Halo 0 to 499')
#plt.plot(k4, power4-shotnoise4,label='Halo 0 to 999, 218 galaxies')
#plt.plot(k5, power5-shotnoise5,label='Halo 0 to 199.999, ~4000 galaxies')
plt.plot(k_galAll,power_galAll)#-shotnoise_galAll)
plt.plot(k_galFilter,power_galFilter)#-shotnoise_galFilter)

# format the axes
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r"$k$ [$h \ \mathrm{Mpc}^{-1}$]")
plt.ylabel(r"$P(k)$ [$h^{-3}\mathrm{Mpc}^3$]")
plt.title('Power spectrum P(k) for various stellar positions')
plt.legend(loc='upper right')
plt.savefig('ps_30122023.png')
plt.show()
