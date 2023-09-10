import h5py
import swiftsimio as sw
import numpy as np
import matplotlib.pyplot as plt

path_hydro = "/net/hypernova/data2/FLAMINGO/L1000N1800/HYDRO_FIDUCIAL/SOAP/halo_properties_0077.hdf5"
with h5py.File(path_hydro, "r") as handle:
    M200m = handle["SO/200_mean/TotalMass"][:]
    R200m = handle["SO/200_mean/SORadius"][:]
    center = handle["VR/CentreOfPotential"][:]
    struct_type = handle["VR/StructureType"][:]
    GM200m = handle["SO/200_mean/GasMass"][:] 
    SM200m = handle["SO/200_mean/StellarMass"][:]
    M500c = handle["SO/500_crit/TotalMass"][:]
    GM500c = handle["SO/500_crit/GasMass"][:]
    SM500c = handle["SO/500_crit/StellarMass"][:]

print(np.shape(M200m))
 
M200m_main = M200m[(struct_type == 10)]

print(M200m_main)
print(np.shape(M200m_main))

M200m_main_max = "{:e}".format(np.max(M200m_main))
print('The largest halo mass in this simulation is', M200m_main_max)

plt.hist(np.log10(M200m_main),density=True)
plt.xlabel('log10 of the M200m halo mass')
plt.ylabel('Number density')
plt.title('Histogram of the M200main halo mass')
plt.savefig('hist_alt.png')
plt.close()

hist_bins = np.linspace(10,16,100)
M200m_hist, M200m_bins  = np.histogram(np.log10(M200m_main),bins=hist_bins)

fig,ax = plt.subplots()
ax.plot(M200m_bins[1:],M200m_hist/len(M200m_main))
ax.set_yscale('log')
ax.set_xscale('log')
ax.set_xlabel('log10 of the M200m halo mass')
ax.set_ylabel('Number density of halos')
ax.set_title('Histogram')
fig.savefig('hist1.png')
plt.show()
plt.close()

GM200m_main = GM200m[(struct_type == 10)]
SM200m_main = SM200m[(struct_type == 10)]

baryon_frac = (GM200m_main+SM200m_main)/M200m_main

fig,ax = plt.subplots()
ax.scatter(M200m_main,baryon_frac,s=1)
ax.set_xscale('log')
ax.set_xlim([1e13,max(M200m_main)])
ax.set_ylim([0,0.17])
ax.set_title('Baryon fraction against halo mass (200 mean)')
plt.show()
plt.close()

M500c_main = M500c[(struct_type == 10)]
GM500c_main = GM500c[(struct_type == 10)]
SM500c_main = SM500c[(struct_type == 10)]

fbar_c = (GM500c_main+SM500c_main)/M500c_main

#curiously, at some of the low halo mass values, the baryon fractions rises above 1 (impossible)

fig,ax = plt.subplots()
ax.scatter(M500c_main,fbar_c,s=1)
ax.set_xscale('log')
ax.set_xlim([1e13,max(M500c_main)])
ax.set_ylim([0,0.17])
ax.set_title('Baryon fraction against halo mass (500 crit)')
plt.show()
plt.close()
