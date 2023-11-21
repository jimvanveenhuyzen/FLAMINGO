import numpy as np
import matplotlib.pyplot as plt 
import h5py
import timeit
import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE" #prevents an uncommon error while reading in h5py files 

def distance(centre_of_pot,pos):
    #Distance between star and centre of potential squared, should be smaller than r**2=50**2 
    return (centre_of_pot[0]-pos[:,0])**2 + (centre_of_pot[1]-pos[:,1])**2 + (centre_of_pot[2]-pos[:,2])**2

"""
First, read in all the relevant stellar data, including Position, Velocity, Mass, ParticleID, FOFGroupID and GroupNr_all
"""
with h5py.File('/disks/cosmodm/vanveenhuyzen/stellar_properties.h5', 'r') as data:
    data.keys()
    positions = data.get('Coordinates')
    velocities = data.get('Velocities')
    masses = data.get('Masses')
    GroupNr_all = data.get('GroupNr_all')
    positions = np.array(positions,dtype=np.float32)
    velocities = np.array(velocities,dtype=np.float32)
    masses = np.array(masses,dtype=np.float32)
    GroupNr_all = np.array(GroupNr_all,dtype=np.int32)
data.close()

"""
Next, from the SOAP/VR files we read in the ID, HostHaloID and CentreOfPot
"""

SOAP_file = '/net/hypernova/data2/FLAMINGO/L1000N1800/HYDRO_FIDUCIAL/SOAP/halo_properties_0077.hdf5'
with h5py.File(SOAP_file,'r') as soap:
    HostHaloID = soap['VR/HostHaloID'][...]
    HaloID = soap['VR/ID'][...]
    CentreOfPot = soap['VR/CentreOfPotential'][...] #this might be a problem: might take all particles instead of only stellar 
    StellarMass = soap['InclusiveSphere/50kpc/StellarMass'][...]
    ##
    ES_sm = soap['ExclusiveSphere/50kpc/StellarMass'][...]
    FOFSH_sm = soap['FOFSubhaloProperties/StellarMass'][...]
    IS_COM = soap['InclusiveSphere/50kpc/CentreOfMass'][...]

print(StellarMass.shape)
print(CentreOfPot.dtype)

#Create a mask to filter out halos without stars
filter_noStars = np.where(StellarMass != 0)
StellarMass_filtered = StellarMass[filter_noStars]

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

#use this condition to check whether we have to use ID (for field halos) or the HostHaloID (for non-field halos)
#note that we define field halos as sub-halos as well! They are defined as subhalos of themselves
SubHaloID = np.where(HostHaloID == -1, HaloID, HostHaloID)
SubHaloID = np.array(SubHaloID,dtype=np.uint32)

#Remove all ungrouped particles 
#UNUSED: LOOK INTO THIS??
GroupNr_all_ungrouped = GroupNr_all[GroupNr_all != -1]
print(len(GroupNr_all_ungrouped)/len(GroupNr_all))


#Now we have the ID of the current SubHalo. 
#GroupNr_All contains the index of the VR Group the particle belongs too 

print('amount of halos\n',HostHaloID.shape)

def centre_of_mass(pos,mass):
    """
    Calculates the centre of mass position using input positions and masses
    """
    return np.matmul(mass,pos)/np.sum(mass)


print('start timer')
begin = timeit.default_timer()

#Use GroupNr_all+1 because of how GroupNr is defined: 0 is an index, so represents group 1 
GroupNr_all += 1
def SH_properties(halo):
    #First, obtain the group that the subhalo is in 
    halo_group = SubHaloID[halo] 
    halo_stellar_mass = StellarMass[halo]

    es_sm_halo = ES_sm[halo]
    fofsh_sm_halo = FOFSH_sm[halo]

    #Now, obtain all the positions of the particles inside this group (first mask)

    mask1 = np.where(GroupNr_all == halo_group) #IMPORTANT: np.where seems to be significantly faster 
    print(np.shape(mask1))

    positions_in_group = positions[mask1]
    velocities_in_group = velocities[mask1]
    masses_in_group = masses[mask1]

    halo_pos = CentreOfPot[halo] #The VR Centre of Potential! 
    #halo_pos = centre_of_mass(positions_in_group,masses_in_group) doesnt work better 
    #halo_pos = IS_COM[halo]

    #Next, find out which particles are within 50kpc of the CentreOfPot
    mask2 = np.where(distance(halo_pos,positions_in_group) < 0.05*0.05)

    positions_in_halo = positions_in_group[mask2]
    velocities_in_halo = velocities_in_group[mask2]
    masses_in_halo = masses_in_group[mask2]

    if len(positions_in_halo[:,0]) == 0:
        print('There are no particles within 50 kpc!')
        return None,None,None,None,None

    #Number of particles
    print("The number of stellar particles in subhalo {} is".format(halo),positions_in_halo.shape)

    #Calculate the mass of the galaxy 

    particle_mass = 1.07e9 #solar mass
    galaxy_mass = particle_mass * np.sum(masses_in_halo)
    group_mass = particle_mass * np.sum(masses_in_group)
    cutoff_mass = 1.5*SM_max
    print('The total mass of the galaxy is {:.2e} M_sun.'.format(galaxy_mass))
    print('The total mass of the group is {:.2e} M_sun.'.format(group_mass))
    print('The inclusive-sphere stellar mass according to the data is {:.2e} M_sun.'.format(halo_stellar_mass))
    print('The exclusive-sphere stellar mass according to the data is {:.2e} M_sun.'.format(es_sm_halo))
    print('The FOFSubhaloProperties stellar mass according to the data is {:.2e} M_sun.'.format(fofsh_sm_halo))

    #Check whether we can define the subhalo as a galaxy, if not return 0 
    if galaxy_mass < cutoff_mass:
        print('The mass of the galaxy is below the cut-off point of {:.1e} M_sun.'.format(cutoff_mass))
        return None,None,None,None,None

    #Compute some important properties we will use later 
    vel_disp = np.std(velocities_in_halo)
    mean_pos = np.mean(positions_in_halo)
    mean_vel = np.mean(velocities_in_halo)

    return positions_in_halo, mean_pos, mean_vel, vel_disp, galaxy_mass
    
pos_in_galaxies = np.zeros((1,3))
gal_count = 0
for i in range(100):
    pos,mean_pos,mean_vel,vel_disp,gal_mass = SH_properties(i)
    #Add the positions to the total amount of galaxy positions if the subhalo is defined as a galaxy
    if pos is not None:
        gal_count += 1
        pos_in_galaxies = np.concatenate((pos_in_galaxies,pos),axis=0) 
end = timeit.default_timer()-begin
print('end timer:',end)

pos_in_galaxies = np.delete(pos_in_galaxies,0,axis=0)
print('The total amount of particles is',pos_in_galaxies.shape)
print('The amount of Subhalos we identify as galaxies is',gal_count)
#np.savetxt('gal_pos_100.txt',pos_in_galaxies)






