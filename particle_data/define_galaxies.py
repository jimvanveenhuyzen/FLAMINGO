import numpy as np
import matplotlib.pyplot as plt 
import h5py
import timeit
import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE" #prevents an uncommon error while reading in h5py files 

def distance(centre_of_pot,pos):
    #Distance between star and centre of potential squared, should be smaller than r**2=50**2 
    return (centre_of_pot[0]-pos[:,0])**2 + (centre_of_pot[1]-pos[:,1])**2 + (centre_of_pot[2]-pos[:,2])**2

#First, read in all the relevant stellar data, including Position, Velocity, Mass, ParticleID, FOFGroupID and GroupNr_all
with h5py.File('/disks/cosmodm/vanveenhuyzen/stellar_properties.h5', 'r') as data:
    data.keys()
    positions = data.get('Coordinates')
    velocities = data.get('Velocities')
    masses = data.get('Masses')
    GroupNr_all = data.get('GroupNr_all')
    positions = np.array(positions,dtype=np.float32)
    velocities = np.array(velocities,dtype=np.float32)
    masses = np.array(masses,dtype=np.float64)
    GroupNr_all = np.array(GroupNr_all,dtype=np.int64)
data.close()

#Load in the look-up table of the GroupNr indices, note that we must start from index 1 (for group 0), since the first index represents GroupNr -1 
GroupNr_lookuptable = np.load('GroupNr_table.npy') 
num_groups = len(GroupNr_lookuptable)

#First, sort the positions, velocities and masses using the sorted GroupNr_all indices
GroupNr_sortedIDs = np.argsort(GroupNr_all)
positions_sorted = positions[GroupNr_sortedIDs]
velocities_sorted = velocities[GroupNr_sortedIDs]
masses_sorted = masses[GroupNr_sortedIDs]

#Next, from the SOAP/VR files we read in the ID, HostHaloID, CentreOfPot and finally StellarMass to check our calculation 
SOAP_file = '/net/hypernova/data2/FLAMINGO/L1000N1800/HYDRO_FIDUCIAL/SOAP/halo_properties_0077.hdf5'
with h5py.File(SOAP_file,'r') as soap:
    HostHaloID = soap['VR/HostHaloID'][...]
    HaloID = soap['VR/ID'][...]
    CentreOfPot = soap['VR/CentreOfPotential'][...] #use this for the average position of the stellar particles per halo! 
    StellarMass = soap['InclusiveSphere/50kpc/StellarMass'][...]

num_halos = len(StellarMass)

"""
START
Do this code (until SubhaloID) somewhere else since we dont filter for mass anyway 
"""
#Create a mask to filter out halos without stars
filter_noStars = np.where(StellarMass != 0)
StellarMass_filtered = StellarMass[filter_noStars]
print('ratio between Stellar mass filter & stellar mass:',len(StellarMass_filtered)/len(StellarMass))

#Next, create a historgram to plot the stellar 
SM_hist,SM_bins = np.histogram(StellarMass_filtered,bins=np.logspace(8,13,100))


smf_max = np.argmax(SM_hist)
SM_max = SM_bins[smf_max]
print('The mass at which the halo density is maximal is {:.2e} M_sun'.format(SM_max))
print('The cut-off mass, under which the resolution is too poor: {:.2e} M_sun'.format(1.5*SM_max))

cutoff_mask = np.where(StellarMass > 1.5*SM_max)
halos_aboveCutoff = StellarMass[cutoff_mask]
print('Ratio between halos above 1.5 times cutoff mass and total halos:',len(halos_aboveCutoff)/len(StellarMass))

fig,ax = plt.subplots()
ax.plot(SM_bins[:-1],SM_hist/len(StellarMass_filtered))
ax.vlines(1.5*SM_max,0,1,color='black',linestyle='dashed')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_ylabel('dn/dlog10(M*) [Mpc^-3]')
ax.set_xlabel('Stellar mass log10(M*) [Msun]')
ax.set_title('Galaxy stellar mass function (SMF)')
fig.savefig('halo_SMF.png')
plt.close()
"""
END
"""

#Use this condition to check whether we have to use ID (for field halos) or the HostHaloID (for non-field halos)
#Note that we define field halos as sub-halos as well! They are defined as subhalos of themselves
SubHaloID = np.where(HostHaloID == -1, HaloID, HostHaloID)
SubHaloID = np.array(SubHaloID,dtype=np.int32)

sh_unique = np.unique(SubHaloID)
print('Number of SubHalo groups (including doubles)',len(SubHaloID))
print('Unique values in NrGroup_all',np.shape(GroupNr_lookuptable))

#Now we have the ID of the current SubHalo. 
#GroupNr_All contains the index of the VR Group the particle belongs to

print('Starting the timer of the main loop...')
begin = timeit.default_timer()

#Use GroupNr_all+1 because of how GroupNr is defined: 0 is an index, so represents group 1 
#GroupNr_all += 1
def SH_properties(halo):
    #First, obtain the group that the subhalo is in 
    halo_group = SubHaloID[halo] 
    halo_stellar_mass = StellarMass[halo]
    halo_pos = CentreOfPot[halo] #The VR Centre of Potential! 

    """
    Now, obtain all the positions of the particles inside this group (first mask).
    Be careful with the indexing here: in reality we are using (halo_group-1)+1, the +1 since the first element
    of our lookup table is for the ungrouped particles (-1), but -1 because of how GroupNr is defined: 0 is an index, so represents group 1 
    """
    #Get the current indices of the particles in this halo group 
    mask_ = GroupNr_lookuptable[halo_group] 
    if mask_[0] == -1:
        print('No stellar particles in this group')
        return [0.,0.,0.],[0.,0.,0.],0.
    
    positions_in_group = positions_sorted[mask_[0]:mask_[1]]
    velocities_in_group = velocities_sorted[mask_[0]:mask_[1]]
    masses_in_group = masses_sorted[mask_[0]:mask_[1]]

    #Next, find out which particles are within 50kpc of the CentreOfPot
    mask2 = np.where(distance(halo_pos,positions_in_group) < 0.05*0.05)

    #Apply this mask over the particles in the group 
    positions_in_halo = positions_in_group[mask2]
    velocities_in_halo = velocities_in_group[mask2]
    masses_in_halo = masses_in_group[mask2]

    #Number of particles in this subhalo 
    print("The number of stellar particles in subhalo {} is".format(halo),positions_in_halo.shape)

    #Calculate the mass of the galaxy 
    particle_mass = 1e10 #mass per stellar particle 
    galaxy_mass = particle_mass * np.sum(masses_in_halo)
    print('The total mass of the galaxy is {:.2e} M_sun.'.format(galaxy_mass))
    #print('The inclusive-sphere stellar mass according to the data is {:.2e} M_sun.'.format(halo_stellar_mass))

    #Compute the mean stellar velocity and the velocity dispersion 
    if len(velocities_in_halo) == 0:
        print('No stellar particles in the halo')
        return [0.,0.,0.],[0.,0.,0.],0.
    else:
        vel_disp = np.var(velocities_in_halo,axis=0)
        mean_vel = np.mean(velocities_in_halo,axis=0)

    return mean_vel,vel_disp,galaxy_mass
    
SH_meanVel = np.zeros((num_halos-1,3))
SH_velDisp = np.zeros((num_halos-1,3))
SH_stellarMass = np.zeros((num_halos-1))
for i in range(num_groups-1):
    mean_vel,vel_disp,gal_mass = SH_properties(i)
    print('The quantities are:\n')
    print(mean_vel,vel_disp,gal_mass)
    SH_meanVel[i] = mean_vel
    SH_velDisp[i] = vel_disp
    SH_stellarMass[i] = gal_mass
end = timeit.default_timer()-begin
print('end timer:',end)

print('CoP for subhalo 1000000',CentreOfPot[1000000])
#article_mass = 1e10 #mass per stellar particle 
#print(SH_meanVel)
#print(SH_velDisp)
#test__ = SH_stellarMass*1e10
#print(test__[-50:])
#print(StellarMass[-50:])
#np.savetxt('galaxy_meanVel.txt',SH_meanVel)
#np.savetxt('galaxy_velDisp.txt',SH_velDisp)
#np.savetxt('galaxy_stellarMass.txt',SH_stellarMass)
np.save('galaxy_meanVel.npy',SH_meanVel)
np.save('galaxy_velDisp.npy',SH_velDisp)
np.save('galaxy_stellarMass.npy',SH_stellarMass)

t = np.load('galaxy_stellarMass.npy')
print(t[0:100])







