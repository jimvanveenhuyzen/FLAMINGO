import h5py
import swiftsimio as sw
import numpy as np
import matplotlib.pyplot as plt

#First we import the relevant data

path_hydro = "/net/hypernova/data2/FLAMINGO/L1000N1800/HYDRO_FIDUCIAL/SOAP/halo_properties_0077.hdf5"
with h5py.File(path_hydro, "r") as handle:
    struct_type = handle["VR/StructureType"][:] 
    #
    M200m = handle["SO/200_mean/TotalMass"][:]
    GM200m = handle["SO/200_mean/GasMass"][:] 
    SM200m = handle["SO/200_mean/StellarMass"][:]
    #
    M500c = handle["SO/500_crit/TotalMass"][:]
    GM500c = handle["SO/500_crit/GasMass"][:]
    SM500c = handle["SO/500_crit/StellarMass"][:]

#Example 1

M200m_main = M200m[(struct_type == 10)] #mask of only the main haloes
M200m_main_max = "{:e}".format(np.max(M200m_main))

print('The largest halo mass in this simulation is {} solar masses'.format(M200m_main_max))

hist_bins = np.linspace(10,16,100) #select bins in appropiate range
M200m_hist, M200m_bins  = np.histogram(np.log10(M200m_main),bins=hist_bins) #compute the hist of log10(M200m)

fig,ax = plt.subplots()
ax.plot(10**(M200m_bins[:-1]),M200m_hist/len(M200m_main)) 
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel(r'$M_{200m}$ [$M_{\odot}$]')
ax.set_ylabel(r'dn/d$log_{10} (M_{200m}) [Mpc^{-3}]$')
ax.set_title(r'Halo mass function at $\rho_{200m}$')
fig.savefig('hmf_200m.png')
plt.close()

#Example 2

GM200m_main = GM200m[(struct_type == 10)] #also apply main halo-mask over gass- and stellar mass
SM200m_main = SM200m[(struct_type == 10)]

bf_200m = (GM200m_main+SM200m_main)/M200m_main #compute the baryon fraction of 200 mean 

#import the snapshot file to get the cosmic baryon fraction
snapshot_file = "/net/hypernova/data2/FLAMINGO/L1000N1800/HYDRO_FIDUCIAL/snapshots/flamingo_0077/flamingo_0077.0.hdf5"
with h5py.File(snapshot_file,"r") as snapshot:
    Ob=snapshot["Cosmology"].attrs["Omega_b"][0] #otherwise we get a 1-element array instead of a scalar
    Om=snapshot["Cosmology"].attrs["Omega_m"][0]
cf=Ob/Om
print("The cosmic baryon fraction is {}".format(cf))

fig,ax = plt.subplots()
ax.scatter(M200m_main,bf_200m,s=1)
ax.hlines(cf,13,max(M200m_main),linestyle='dashed',label='$\Omega_{b}/ \Omega_{m}$',color='black')
ax.set_xscale('log')
ax.set_xlim([1e13,max(M200m_main)])
ax.set_ylim([0,0.18])
ax.set_xlabel(r'$M_{200m} [M_{\odot}]$')
ax.set_ylabel(r'$f_{baryon,200m}$')
ax.set_title('Baryon fraction against halo mass (200 mean)')
ax.legend(loc='upper left')
fig.savefig('bf_200m.png')
plt.close()

M500c_main = M500c[(struct_type == 10)] #repeat the previous steps for 500 crit
GM500c_main = GM500c[(struct_type == 10)]
SM500c_main = SM500c[(struct_type == 10)]

bf_500c = (GM500c_main+SM500c_main)/M500c_main

fig,ax = plt.subplots()
ax.scatter(M500c_main,bf_500c,s=1)
ax.hlines(cf,13,max(M200m_main),linestyle='dashed',label='$\Omega_{b}/ \Omega_{m}$',color='black')
ax.set_xscale('log')
ax.set_xlim([1e13,max(M500c_main)])
ax.set_ylim([0,0.18])
ax.set_xlabel(r'$M_{500c} [M_{\odot}]$')
ax.set_ylabel(r'$f_{baryon,500c}$')
ax.set_title('Baryon fraction against halo mass (500 crit)')
ax.legend(loc='upper left')
fig.savefig('bf_500c.png')
plt.close()
