import h5py
import swiftsimio as sw
import numpy as np
import matplotlib.pyplot as plt

group = "/net/hypernova/data2/FLAMINGO/L1000N1800/HYDRO_FIDUCIAL/VR/catalogue_0077/vr_catalogue_0077.catalog_groups.0"
particles = "/net/hypernova/data2/FLAMINGO/L1000N1800/HYDRO_FIDUCIAL/VR/catalogue_0077/vr_catalogue_0077.catalog_particles.0"
snapshot = "/net/hypernova/data2/FLAMINGO/L1000N1800/HYDRO_FIDUCIAL/snapshots/flamingo_0077/flamingo_0077.0.hdf5"

with h5py.File(group,"r") as group_file:
    halo_start_position = group_file["Offset"][0]
    halo_end_position = group_file["Offset"][1]

group_file.close()
print("Amount of halo IDs between 0 and 1",halo_end_position)

with h5py.File(particles,"r") as particles_file:
    particle_ids_in_halo = particles_file["Particle_IDs"][halo_start_position:halo_end_position]

particles_file.close()

with h5py.File(snapshot,"r") as snapshot_file:
    particle_ids_from_snapshot = snapshot_file["PartType4/ParticleIDs"][...]

print("The particle ids from snapshot file 0 are",particle_ids_from_snapshot,particle_ids_from_snapshot.shape)

partIDs_total = np.zeros(1800**3) #create empty array to fill out all particle IDs
partIDs_total_stars = np.zeros(541631305)
index_count = 0 #we will count which index we are on

Ncurr_list = []

partPos_total = np.zeros((1800**3,2))
partPos_total_stars = np.zeros((541631305,2))

for i in range(64): #change this to 32 or 64 
    snapshot_file = "/net/hypernova/data2/FLAMINGO/L1000N1800/HYDRO_FIDUCIAL/snapshots/flamingo_0077/flamingo_0077.{}.hdf5".format(i)
    with h5py.File(snapshot_file,"r") as snapshot:
        Ncurr=snapshot["Header"].attrs["NumPart_ThisFile"][4]
        print(Ncurr)
        Ncurr_list.append(Ncurr)
        partIDs_total_stars[index_count:index_count+Ncurr] = snapshot["PartType4/ParticleIDs"][...]
        #print(partIDs_total[index_count:index_count+Ncurr])
        partPos_total_stars[index_count:index_count+Ncurr] = snapshot["PartType4/Coordinates"][:][:,0:2] #fill with the x-y coordinates
    print('loop number',i)
    index_count += Ncurr
print('check')

print('amount of particles',np.sum(Ncurr_list))

_, indices_v, indices_p = np.intersect1d(particle_ids_in_halo,partIDs_total_stars,assume_unique=True,return_indices=True)

particle_positions_in_halo = partPos_total_stars[indices_p]

print('saving on a .txt file')

print('the array is',particle_positions_in_halo,particle_positions_in_halo.shape)

np.savetxt('positions_halo0_hydro_stars.txt',particle_positions_in_halo)

print('done')
