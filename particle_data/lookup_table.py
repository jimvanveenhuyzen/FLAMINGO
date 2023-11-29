import numpy as np
import h5py
import timeit
import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE" #prevents an uncommon error while reading in h5py files 

begin = timeit.default_timer()

with h5py.File('/disks/cosmodm/vanveenhuyzen/stellar_properties.h5', 'r') as data:
    data.keys()
    #positions = data.get('Coordinates')
    #velocities = data.get('Velocities')
    #masses = data.get('Masses')
    GroupNr_all = data.get('GroupNr_all')
    GroupNr_all = np.array(GroupNr_all,dtype=np.int64)
    #positions = np.array(positions)
    #velocities = np.array(velocities)
    #masses = np.array(masses)

    #FOFGroupIDs = data.get('FOFGroupIDs')
    #FOFGroupIDs = np.array(FOFGroupIDs,dtype=np.int32)

data.close()

GroupNr_sortedIDs = np.argsort(GroupNr_all)
GroupNr_all_sorted = GroupNr_all[GroupNr_sortedIDs]
GroupNr_unique = len(np.unique(GroupNr_all)) #Number of unique GroupNumbers
print(GroupNr_all_sorted[2301980:2302000])
print(GroupNr_all[2301980:2302000])
lookup = []
index_start = 0
index_prev = 0

count = 0
#for i in range(10000000):
for i in range(len(GroupNr_all_sorted)):
    index_current = i
    #If the element using the current index is different to the element using the previous index, we move on to next ID and append to the table 
    if GroupNr_all_sorted[index_current] != GroupNr_all_sorted[index_prev]:
        print('The VR Group of the PREV and CURRENT index i is:')
        print(GroupNr_all_sorted[index_prev], GroupNr_all_sorted[index_current])
        GroupNr_difference = GroupNr_all_sorted[index_current]-GroupNr_all_sorted[index_prev]
        if GroupNr_difference != 1 and GroupNr_all_sorted[index_prev] != -1:
            for j in range(GroupNr_difference-1):
                lookup.append([-1,-1])
        #idx_diff = index_prev-index_start
        #if idx_diff == 1 or GroupNr_all_sorted[index_prev] == -1:

        #if GroupNr_difference == 1 or GroupNr_all_sorted[index_prev] == -1: #Second condition only happens once 
        lookup.append([index_start,index_prev])
        index_start = i
        #else:
        #    count += 1
        #    if count == 10:
        #        print('last 50 elements')
        #        print(index_current)
        #        print(np.array(lookup)[index_current-5:index_current+5])
        #        break 
        """
        else:
            #lookup.append([index_start,index_start+GroupNr_difference])
            idx1,idx2 = index_start-GroupNr_difference+1, index_start-1 
            lookup.append([idx1,idx2])
            lookup.append([index_start,index_prev])
            #index_start = i+GroupNr_difference-1
            #print(index_current,index_prev,index_start)
            if i % 100000 == 0:
                print('difference:',GroupNr_difference)
                print('current, prev, start\n',index_current,index_prev,index_start)
                print(lookup[-3:])
            index_start = i
        """
        #add if: are we skipping any? append -1 (or something else) diff between GroupNr_all_sorted(index_current) and GroupNr_all_sorted(index_prev) 
        #lookup.append([index_start,index_prev])
        #index_start = i
    index_prev = i 

lookup = np.array(lookup,dtype=np.int64)
#print(GroupNr_all_sorted[2301988:2301998])
print(lookup[0:10])
#print(lookup[-5:])
print(lookup[-500:])
print('The last VR group in NrGroup_all_sorted is',GroupNr_all_sorted[-1])
print(lookup.shape)
np.save('GroupNr_table.npy',lookup)
#np.savetxt('GroupNr_lookuptable.txt',lookup)
a = np.load('GroupNr_table.npy')
print(a[1000000:1000500])
print(a[0:10])
"""
sortedIDs = np.argsort(FOFGroupIDs)
print(sortedIDs[-100:])

FOFGroupIDs_sorted = FOFGroupIDs[sortedIDs]
#print(FOFGroupIDs_sorted[52690:52720])
print(np.count_nonzero(FOFGroupIDs_sorted == 1))


lookup_table = []
index_prev = 0
index_start = 0
for i in range(len(FOFGroupIDs)):

    index_current = i
    #If the element using the current index is different to the element using the previous index, we move on to next ID and append to the table 
    if FOFGroupIDs[index_current] != FOFGroupIDs[index_prev]:
        lookup_table.append([FOFGroupIDs[index_prev],index_start,index_prev])
        index_start = i

    index_prev = i 

np.savetxt('lookup_table.txt',lookup_table)

a = [1,2,3]
print(len(a))
lookup_table = np.genfromtxt('lookup_table.txt')
lookup_table = lookup_table.astype(int)
print(lookup_table.shape)
print(lookup_table[-1])


print(FOFGroupIDs[541631300:541631305])
print(FOFGroupIDs[541631304])
print('-'*200)

#lookup_table.append([10241977,541631300,541631304])
print(lookup_table[-3:])
#lookup_table = np.append(lookup_table, np.array([[10241977,541631300,541631304]]), axis=0)
print(lookup_table[-3:])

print(lookup_table.astype(int)[-10:])

#np.savetxt('lookup_table.txt',lookup_table)

print('-'*200)
print(lookup_table.astype(int)[0:10])

print('e.g. I want to get the indices of FOF group ID 63680, then I get...')
index_test = np.argwhere(lookup_table.astype(int)[:,0] == 63680)
corr_row = np.squeeze(lookup_table.astype(int)[index_test])
print(corr_row)
print('another example, ID number 1:')
index_test = np.argwhere(lookup_table.astype(int)[:,0] == 1)
corr_row = np.squeeze(lookup_table.astype(int)[index_test])
print(corr_row)

unique_ids = len(lookup_table[:,0]) #The number of unique FOFGroup IDs
indices_perID = np.zeros((unique_ids,2))

for i in range(unique_ids):
    indices_perID[i] = np.array([lookup_table[i,1],lookup_table[i,2]])

print('The final lookup table is:')
print(indices_perID)
np.savetxt('groupID_table.txt',indices_perID)

end = timeit.default_timer() - begin
print('The execution time was {:.1f} seconds!'.format(end))
"""