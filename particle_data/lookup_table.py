import numpy as np
import h5py
import timeit
import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE" #prevents an uncommon error while reading in h5py files 

begin = timeit.default_timer()

with h5py.File('/disks/cosmodm/vanveenhuyzen/HYDRO_FIDUCIAL_z1/sp_z1.h5', 'r') as data:
    data.keys()
    GroupNr_all = data.get('GroupNr_all')
    GroupNr_all = np.array(GroupNr_all,dtype=np.int64)
data.close()

#First, sort the GroupNr data by the VR group number
GroupNr_sortedIDs = np.argsort(GroupNr_all)
GroupNr_all_sorted = GroupNr_all[GroupNr_sortedIDs]

#Initialize an empty list and some variables to keep track of indices 
lookup = []
index_start = 0
index_prev = 0
for i in range(len(GroupNr_all_sorted)):
    #Keep track of the current index
    index_current = i
    #If the element using the current index is different to the element using the previous index, we move on to next VR ID and append to the table 
    if GroupNr_all_sorted[index_current] != GroupNr_all_sorted[index_prev]:
        GroupNr_difference = GroupNr_all_sorted[index_current]-GroupNr_all_sorted[index_prev]

        #Check if the difference between the current and previous VR group is 1, if it is not
        #we need to fill in the lookup table with filler elements [-1,-1] for the empty VR groups that contain no stellar particles
        if GroupNr_difference != 1 and GroupNr_all_sorted[index_prev] != -1:
            for j in range(GroupNr_difference-1):
                lookup.append([-1,-1])
        #Append the corresponding indices of the snapshot data for this specific VR group 
        lookup.append([index_start,index_prev])
        #Update the 'starting' index for this specific lookup table entry 
        index_start = i
    #Keep track of the previous index
    index_prev = i 

lookup = np.array(lookup,dtype=np.int64)
print('The size of the lookup table is:',lookup.shape)
#Save as a binary file to save space 
np.save('z1/GroupNr_table_z1.npy',lookup)