import numpy as np
import h5py
import os 

def read_membership(snapshot_path,membership_path):
    #snapshot_path should start and end with a / symbol! 

    num_files = 0
    membership_files = "/net/hypernova/data2/FLAMINGO" + membership_path
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

    print(Ntot)

    #create arrays to fill with positions, velocities, masses and of course IDs of the particles 
    stellar_GroupNr = np.zeros(Ntot)

    index_count = 0 
    #loop through all snapshot files
    for i in range(num_files):

        snapshot_file = snapshot_files + "flamingo_0077.{}.hdf5".format(i)
        membership_file = membership_files + "membership_0077.{}.hdf5".format(i)

        with h5py.File(snapshot_file,"r") as snapshot:
            Ncurr=snapshot["Header"].attrs["NumPart_ThisFile"][4] #the number of particles per snapshot file 
        #snapshot.close()

        with h5py.File(membership_file,"r") as member:
            stellar_GroupNr[index_count:index_count+Ncurr]  = member["PartType4/GroupNr_all"][...] #fill with the stellar GroupNr
        #member.close()

        print('Reading in snapshot file {} out of 64...'.format(i))
        index_count += Ncurr

    print(stellar_GroupNr)
    #next, we save the data to a h5 file in binary format 
    print('Save the data to a h5 file...')
    stellar_properties = h5py.File('/disks/cosmodm/vanveenhuyzen/stellar_properties.h5', 'w')
    stellar_properties.create_dataset('GroupNr_all',data=stellar_GroupNr)
    stellar_properties.close()

    return stellar_GroupNr

member_path = "/L1000N1800/HYDRO_FIDUCIAL/SOAP/membership_0077/"

GroupNr_all = read_membership("/L1000N1800/HYDRO_FIDUCIAL/snapshots/flamingo_0077/",member_path)

#open the HostHaloID and IDs: 
id_path = "/net/hypernova/data2/FLAMINGO/L1000N1800/HYDRO_FIDUCIAL/SOAP/halo_properties_0077.hdf5"
with h5py.File(id_path,"r") as vr_id:
    id = vr_id["VR/ID"][...]
    hosthalo_id = vr_id["VR/HostHaloID"][...]
    index = vr_id["VR/Index"][...]
vr_id.close()

print(id.shape)
print(GroupNr_all.shape)
print(hosthalo_id[0:30])
print(GroupNr_all[0:30])
print('index \n',index[0:30])

test = np.arange(1,len(id)+1,1) #note: this is the same array as id! 

print('maximum value of hosthaloID is:',max(hosthalo_id))
#print('maximum value of the groupnr_all',max(GroupNr_all))

#Save the sorted indices as a h5 file: 
"""
GroupNr_all_sortedIDs = np.argsort(GroupNr_all)

sorted = h5py.File('/disks/cosmodm/vanveenhuyzen/indices_sorted.h5', 'w')
sorted.create_dataset('GroupNr_all_sortedIDs',data=GroupNr_all_sortedIDs)
sorted.close()

selection = GroupNr_all[GroupNr_all == 200]
print(selection.shape)

test_indices = np.where(GroupNr_all == 10)
print(test_indices[0:100])
test_indices = np.where(GroupNr_all == 70)
print(test_indices)
test_indices = np.where(GroupNr_all == 98345)
print(test_indices)
print(np.shape(test_indices))
"""