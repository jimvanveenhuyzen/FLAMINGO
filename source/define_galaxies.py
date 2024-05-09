import numpy as np
import matplotlib.pyplot as plt 
import h5py
import timeit
import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE" #prevents an uncommon error while reading in h5py files 

def distance(centre_of_pot,pos):
    """Find the distance between a stellar particle and the centre of potential of the halo.

    Parameters:
    ----------
    centre_of_pot : 1-dimensional array of floats with shape (1,3)
        (x,y,z) coordinates of the centre of potential of the halo
    pos : 2-dimensional array of shape (N,3), given N stellar particles inside the halo 
        (x,y,z) coordinates of the individual stellar particles inside the halo
        
    Returns:
    -------
    dist_from_cen : 1-dimensional array of floats with shape (N,1)
        Square distance between the stellar particle and the centre of potential 
     
    """
    #Using the formula d^2 = (x-x0)^2 + (y-y0)^2 + (z-z0)^2
    dist_from_cen = (centre_of_pot[0]-pos[:,0])**2 + (centre_of_pot[1]-pos[:,1])**2 + (centre_of_pot[2]-pos[:,2])**2
    return dist_from_cen

#First, read in all the relevant stellar data, including Position, Velocity, Mass, ParticleID, FOFGroupID and GroupNr_all
with h5py.File('/disks/cosmodm/vanveenhuyzen/HYDRO_STRONGEST_AGN/sp_STRONGEST_AGN.h5', 'r') as data:
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
GroupNr_lookuptable = np.load('STRONGEST_AGN/GroupNr_table_STRONGEST_AGN.npy') 
num_groups = len(GroupNr_lookuptable)

#First, sort the positions, velocities and masses using the sorted GroupNr_all indices
GroupNr_sortedIDs = np.argsort(GroupNr_all)
positions_sorted = positions[GroupNr_sortedIDs]
velocities_sorted = velocities[GroupNr_sortedIDs]
masses_sorted = masses[GroupNr_sortedIDs]

#Next, from the SOAP/VR files we read in the ID, HostHaloID, CentreOfPot and finally StellarMass to check our calculation 
SOAP_file = '/net/hypernova/data2/FLAMINGO/L1000N1800/HYDRO_STRONGEST_AGN/SOAP/halo_properties_0077.hdf5'
with h5py.File(SOAP_file,'r') as soap:
    HostHaloID = soap['VR/HostHaloID'][...]
    HaloID = soap['VR/ID'][...]
    CentreOfPot = soap['VR/CentreOfPotential'][...] #use this for the average position of the stellar particles per halo! 
    StellarMass = soap['InclusiveSphere/50kpc/StellarMass'][...]

num_halos = len(StellarMass) #Number of subhalos 

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

def SH_properties(halo,radius):
    """For each subhalo, we mask out all the particles that are NOT within {radius} kpc of the
    centre of potential of that halo. For the particles that do fall inside this sphere, we compute 
    the mean velocity and velocity dispersion of the stellar particles, and the total stellar mass
    inside the halo. This function is highly optimized since speed is highly important when reading in 
    millions of subhalos. Some print statements are commented out to save extra time. 

    Note that we use two masks, the first is to find all particles inside the current group of halos, 
    which is done specifically so that the second (np.where) mask does not have to scan an array of length 
    5 billion, but instead on the order of 100k.  

    Parameters:
    ---------- 
    halo: integer
        The number of the current subhalo in ascending order 
    radius: float (in units of Mpc)
        The maximum distance that stellar particles are allowed to be from the centre of potential 
        to be included in the calculations
    
    Returns:
    -------
    mean_vel: 1-dimensional array of floats with shape (1,3)
        Mean velocity of the stellar particles inside the halo in the form (v_x,v_y,v_z)
    vel_disp : 1-dimensional array of floats with shape (1,3)
        Variance of the stellar particles inside the halo (sigma_v^2)
    galaxy_mass : float
        Total mass of the stellar mass of the galaxy, will be 0 if no stellar particles present

    """
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
        #print('No stellar particles in this group')
        return [0.,0.,0.],[0.,0.,0.],0.
    
    #Apply the mask to first find all particles inside the group 
    positions_in_group = positions_sorted[mask_[0]:mask_[1]]
    velocities_in_group = velocities_sorted[mask_[0]:mask_[1]]
    masses_in_group = masses_sorted[mask_[0]:mask_[1]]

    #Next, find out which particles are within r^2 of the CentreOfPot
    #Note that we compare the squares since taking the square root of both
    #values would take additional, unnecassary FLOPs 
    mask2 = np.where(distance(halo_pos,positions_in_group) < radius*radius)

    #Apply this mask over the particles in the group 
    positions_in_halo = positions_in_group[mask2]
    velocities_in_halo = velocities_in_group[mask2]
    masses_in_halo = masses_in_group[mask2]

    #Number of particles in this subhalo 
    #print("The number of stellar particles in subhalo {} is".format(halo),positions_in_halo.shape)

    #Calculate the mass of the galaxy 
    particle_mass = 1e10 #mass per stellar particle 
    galaxy_mass = particle_mass * np.sum(masses_in_halo)
    #print('The total mass of the galaxy is {:.2e} M_sun.'.format(galaxy_mass))
    #print('The inclusive-sphere stellar mass according to the data is {:.2e} M_sun.'.format(halo_stellar_mass))

    #print('Compare calculated mass to mass according to data:')
    #print('{:.3e}'.format(galaxy_mass))
    #print('{:.3e}'.format(halo_stellar_mass))

    #Compute the mean stellar velocity and the velocity dispersion 
    if len(velocities_in_halo) == 0:
        return [0.,0.,0.],[0.,0.,0.],0.
    else:
        vel_disp = np.var(velocities_in_halo,axis=0)
        mean_vel = np.mean(velocities_in_halo,axis=0)

    return mean_vel,vel_disp,galaxy_mass
    
SH_meanVel = np.zeros((num_halos-1,3))
SH_velDisp = np.zeros((num_halos-1,3))
SH_stellarMass = np.zeros((num_halos-1))
#Loop through all halo groups 
for i in range(num_groups-1):
    mean_vel,vel_disp,gal_mass = SH_properties(i)
    SH_meanVel[i] = mean_vel
    SH_velDisp[i] = vel_disp
    SH_stellarMass[i] = gal_mass
end = timeit.default_timer()-begin
print('end timer:',end)

#Save the arrays containing the velocities, dispersions and masses 
np.save('STRONGEST_AGN/Vel_STRONGEST_AGN.npy',SH_meanVel)
np.save('STRONGEST_AGN/Disp_STRONGEST_AGN.npy',SH_velDisp)
np.save('STRONGEST_AGN/Mass_STRONGEST_AGN.npy',SH_stellarMass)







