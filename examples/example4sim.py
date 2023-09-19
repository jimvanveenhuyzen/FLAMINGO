import h5py
import swiftsimio as sw
import numpy as np
import matplotlib.pyplot as plt

group = "/net/hypernova/data2/FLAMINGO/L1000N1800/DMO_FIDUCIAL/VR/catalogue_0077/vr_catalogue_0077.catalog_groups.0"
particles = "/net/hypernova/data2/FLAMINGO/L1000N1800/DMO_FIDUCIAL/VR/catalogue_0077/vr_catalogue_0077.catalog_particles.0"
snapshot = "/net/hypernova/data2/FLAMINGO/L1000N1800/DMO_FIDUCIAL/snapshots/flamingo_0077/flamingo_0077.0.hdf5"

with h5py.File(group,"r") as group_file:
    halo_start_position = group_file["Offset"][0]
    halo_end_position = group_file["Offset"][1]

group_file.close()
print("Amount of halo IDs between 0 and 1",halo_end_position)

with h5py.File(particles,"r") as particles_file:
    particle_ids_in_halo = particles_file["Particle_IDs"][halo_start_position:halo_end_position]

particles_file.close()

with h5py.File(snapshot,"r") as snapshot_file:
    particle_ids_from_snapshot = snapshot_file["PartType1/ParticleIDs"][...]

print("The particle ids from snapshot file 0 are",particle_ids_from_snapshot,particle_ids_from_snapshot.shape)

partIDs_total = np.zeros(1800**3) #create empty array to fill out all particle IDs
index_count = 0 #we will count which index we are on

partPos_total = np.zeros((1800**3,2))

for i in range(32): #change this to 32
    snapshot_file = "/net/hypernova/data2/FLAMINGO/L1000N1800/DMO_FIDUCIAL/snapshots/flamingo_0077/flamingo_0077.{}.hdf5".format(i)
    with h5py.File(snapshot_file,"r") as snapshot:
        Ncurr=snapshot["Header"].attrs["NumPart_ThisFile"][1]
        partIDs_total[index_count:index_count+Ncurr] = snapshot["PartType1/ParticleIDs"][...]
        partPos_total[index_count:index_count+Ncurr] = snapshot["PartType1/Coordinates"][:][:,0:2] #fill with the x-y coordinates
    print('loop number',i)
    index_count += Ncurr

_, indices_v, indices_p = np.intersect1d(particle_ids_in_halo,partIDs_total,assume_unique=True,return_indices=True)

particle_positions_in_halo = partPos_total[indices_p]

print('how many coordinates',particle_positions_in_halo.shape)

np.savetxt('positions_halo0.txt',particle_positions_in_halo)
