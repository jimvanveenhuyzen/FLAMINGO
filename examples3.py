import h5py
import swiftsimio as sw
import numpy as np
import matplotlib.pyplot as plt

path_hydro = "/net/hypernova/data2/FLAMINGO/L1000N1800/HYDRO_FIDUCIAL/SOAP/halo_properties_0077.hdf5"
with h5py.File(path_hydro, "r") as handle:
    center = handle["VR/CentreOfPotential"][:]
    SM50kpc = handle["ExclusiveSphere/50kpc/StellarMass"][:]

#remove all the 0 elements, they give errors with the histogram

SM50kpc_filter = SM50kpc[np.nonzero(SM50kpc)]
SM50kpc_hist,SM50kpc_bins = np.histogram(np.log10(SM50kpc_filter),bins=np.linspace(8,13,25))
#print(min(SM50kpc))
#print(np.shape(SM50kpc))

fig,ax = plt.subplots()
ax.plot(SM50kpc_bins[1:],SM50kpc_hist/len(SM50kpc_filter))
ax.set_yscale('log')
ax.set_ylabel('dn/dlog10(M*) [Mpc^-3]')
ax.set_xlabel('Stellar mass M* [Msun]')
fig.savefig('smf.png')
plt.show()

gal_pos = center[(SM50kpc > 5e9) & (center[:,2] < 3)] #z < 3 Mpc

print(gal_pos.shape)
print(center.shape)

print("We are using", len(gal_pos)/len(center)*100,"% of the particles")

plt.scatter(gal_pos[:,0],gal_pos[:,1],s=1,color='black')
plt.xlabel("x [Mpc]")
plt.ylabel("y [Mpc]")
plt.title("x-y positions of the galaxies")
plt.show()
