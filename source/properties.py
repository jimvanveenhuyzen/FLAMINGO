"""This .py file is meant to only be ran once in order to read in the stellar positions, velocities, masses and ofcourse, IDs. 
These are then saved to a h5 file such that we can use these parameters later to match to individual halos. 
"""
import numpy as np
import h5py
import os 
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

def read_snapshots(snapshot_path,membership_path,output_path='/HYDRO_STRONGEST_AGN/sp_STRONGEST_AGN.h5'):
    """Import the snapshot files which contain the particle data, and saves them 
    in a binary format using the h5py library. Expected total file size is ~40GB. 

    Parameters:
    ----------
    snapshot_path : string
        Specifies the directory where the snapshot files are located.
        Should start and end with a '/' character! 
    membership_path : string
        Specifies the membership directory.
    output_path : string
        Specifies the directory to which the .h5 datafile is saved. 

    """

    #Use the os library to scan the directory to find the number of snapshot files, generally this is 64 files 
    num_files = 0
    snapshot_files = "/net/hypernova/data2/FLAMINGO" + snapshot_path
    for path in os.scandir(snapshot_files):
        if path.is_file():
            num_files += 1
    #Subtract the 'numberless' snapshot file as we will not loop through it 
    num_files -= 1
    print('The number of snapshot files is {}'.format(num_files))

    #Read in the 'numberless' snapshot file, which lists the total amount of stellar particles (Ntot) inside the simulations
    snapshot_main = snapshot_files + "flamingo_0077.hdf5"

    with h5py.File(snapshot_main,"r") as snapshot:
        Ntot = snapshot["Header"].attrs["NumPart_ThisFile"][4]
    snapshot.close()

    print("The total number of particles is {:.3e}".format(Ntot))

    #Read in the membership files which relates halo ID to the VR halo finder 
    membership_files = "/net/hypernova/data2/FLAMINGO" + membership_path

    #Create arrays to fill with positions, velocities, masses and of course IDs of the particles 
    stellar_pos_total = np.zeros((Ntot,3))
    stellar_vel_total = np.zeros((Ntot,3))
    stellar_mass_total = np.zeros(Ntot)
    partIDs_total_stars = np.zeros(Ntot)
    stellar_FOFGroupIDs_total = np.zeros(Ntot)
    stellar_GroupNr = np.zeros(Ntot)

    index_count = 0 
    #Loop through all snapshot files
    for i in range(num_files):

        snapshot_file = snapshot_files + "flamingo_0077.{}.hdf5".format(i)
        with h5py.File(snapshot_file,"r") as snapshot:

            Ncurr=snapshot["Header"].attrs["NumPart_ThisFile"][4] #the number of particles per snapshot file 

            partIDs_total_stars[index_count:index_count+Ncurr] = snapshot["PartType4/ParticleIDs"][...] #fill with the particle IDs
            stellar_pos_total[index_count:index_count+Ncurr] = snapshot["PartType4/Coordinates"][...] #fill with the coordinates
            stellar_vel_total[index_count:index_count+Ncurr] = snapshot["PartType4/Velocities"][...] #fill with the velocities
            stellar_mass_total[index_count:index_count+Ncurr] = snapshot["PartType4/Masses"][...] #fill with the masses
            #stellar_FOFGroupIDs_total[index_count:index_count+Ncurr] = snapshot["PartType4/FOFGroupIDs"][...] #fill with the FOF Group IDs
        snapshot.close()

        membership_file = membership_files + "membership_0077.{}.hdf5".format(i)
        with h5py.File(membership_file,"r") as member:
            stellar_GroupNr[index_count:index_count+Ncurr]  = member["PartType4/GroupNr_all"][...] #fill with the stellar GroupNr
        member.close()

        print('Reading in snapshot file {} out of {}...'.format(i,num_files))
        index_count += Ncurr

    #Next, we save the data to a h5 file (binary format to save space) 
    print('Saving the data to a h5 file...')
    stellar_properties = h5py.File('/disks/cosmodm/vanveenhuyzen' + output_path, 'w')
    stellar_properties.create_dataset('ParticleIDs',data=partIDs_total_stars)
    stellar_properties.create_dataset('Coordinates',data=stellar_pos_total)
    stellar_properties.create_dataset('Velocities',data=stellar_vel_total)
    stellar_properties.create_dataset('Masses',data=stellar_mass_total)
    #stellar_properties.create_dataset('FOFGroupIDs',data=stellar_FOFGroupIDs_total)
    stellar_properties.create_dataset('GroupNr_all',data=stellar_GroupNr)
    stellar_properties.close()

read_snapshots("/L1000N1800/HYDRO_STRONGEST_AGN/snapshots/flamingo_0077/",\
                "/L1000N1800/HYDRO_STRONGEST_AGN/SOAP/membership_0077/")

