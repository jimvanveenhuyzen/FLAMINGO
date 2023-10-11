"""
This .py file is meant to only be ran once in order to read in the stellar positions, velocities, masses and ofcourse, IDs. 
These are then saved to a h5 file such that we can use these parameters later to match to individual halos. 
"""
import numpy as np
import h5py
import os 

def read_snapshots(snapshot_path):
    #snapshot_path should start and end with a / symbol! 

    num_files = 0
    snapshot_files = "/net/hypernova/data2/FLAMINGO" + snapshot_path
    for path in os.scandir(snapshot_files):
        if path.is_file():
            num_files += 1
    #subtract the 'numberless' snapshot file as we will not loop through it 
    num_files -= 1
    print('The number of snapshot files is {}'.format(num_files))

    #read in the 'numberless' snapshot file, which lists the total amount of stellar particles inside the simulations
    snapshot_main = snapshot_files + "flamingo_0077.hdf5"

    with h5py.File(snapshot_main,"r") as snapshot:
        Ntot = snapshot["Header"].attrs["NumPart_ThisFile"][4]
    snapshot.close()

    #create arrays to fill with positions, velocities, masses and of course IDs of the particles 
    stellar_pos_total = np.zeros((Ntot,3))
    stellar_vel_total = np.zeros((Ntot,3))
    stellar_mass_total = np.zeros(Ntot)
    partIDs_total_stars = np.zeros(Ntot)

    index_count = 0 
    #loop through all snapshot files
    for i in range(num_files):

        snapshot_file = snapshot_files + "flamingo_0077.{}.hdf5".format(i)
        with h5py.File(snapshot_file,"r") as snapshot:

            Ncurr=snapshot["Header"].attrs["NumPart_ThisFile"][4] #the number of particles per snapshot file 

            partIDs_total_stars[index_count:index_count+Ncurr] = snapshot["PartType4/ParticleIDs"][...] #fill with the particle IDs
            stellar_pos_total[index_count:index_count+Ncurr] = snapshot["PartType4/Coordinates"][...] #fill with the coordinates
            stellar_vel_total[index_count:index_count+Ncurr] = snapshot["PartType4/Velocities"][...] #fill with the velocities
            stellar_mass_total[index_count:index_count+Ncurr] = snapshot["PartType4/Masses"][...] #fill with the masses

        print('Reading in snapshot file {} out of 64...'.format(i))
        index_count += Ncurr

    #next, we save the data to a h5 file in binary format 
    print('Save the data to a h5 file...')
    stellar_properties = h5py.File('/disks/cosmodm/vanveenhuyzen/stellar_properties.h5', 'w')
    stellar_properties.create_dataset('ParticleIDs',data=partIDs_total_stars)
    stellar_properties.create_dataset('Coordinates',data=stellar_pos_total)
    stellar_properties.create_dataset('Velocities',data=stellar_vel_total)
    stellar_properties.create_dataset('Masses',data=stellar_mass_total)
    stellar_properties.close()

read_snapshots("/L1000N1800/HYDRO_FIDUCIAL/snapshots/flamingo_0077/")

