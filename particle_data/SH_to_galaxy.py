import numpy as np
import matplotlib.pyplot as plt 
import h5py

#first we read in the data from ALL particles. More specifically, the positions, velocities, masses and particle IDs
with h5py.File('/disks/cosmodm/vanveenhuyzen/stellar_properties.h5', 'r') as data:
    data.keys()
    particle_ids = data.get('ParticleIDs')
    positions = data.get('Coordinates')
    velocities = data.get('Velocities')
    masses = data.get('Masses')
    particle_ids = np.array(particle_ids)
    positions = np.array(positions)
    velocities = np.array(velocities)
    masses = np.array(masses)
data.close()

def read_subhalo(halo,group,particles):
    group_path = "/net/hypernova/data2/FLAMINGO" + group
    particles_path = "/net/hypernova/data2/FLAMINGO" + particles

    with h5py.File(group_path,"r") as group_file:
        halo_start_position = group_file["Offset"][halo]
        halo_end_position = group_file["Offset"][halo+1]
    group_file.close()

    with h5py.File(particles_path,"r") as particles_file:
        particle_ids_in_halo = particles_file["Particle_IDs"][halo_start_position:halo_end_position]
    particles_file.close()

    print('Start intersection to find IDs in halo...')
    _, indices_v, indices_p = np.intersect1d(particle_ids_in_halo,particle_ids,assume_unique=True,return_indices=True)

    print('Finding the positions,velocities and masses of the particles in halo {}...'.format(halo))
    positions_in_halo = positions[indices_p]
    velocities_in_halo = velocities[indices_p]
    masses_in_halo = masses[indices_p]

    print('The amount of particles in the halo is {}'.format(masses_in_halo.shape))

    return positions_in_halo,velocities_in_halo,masses_in_halo

#read_subhalo(0,"/L1000N1800/HYDRO_FIDUCIAL/VR/catalogue_0077/vr_catalogue_0077.catalog_groups.0",\
#             "/L1000N1800/HYDRO_FIDUCIAL/VR/catalogue_0077/vr_catalogue_0077.catalog_particles.0")

def centre_of_mass(pos,mass):
    """
    Calculates the centre of mass position using input positions and masses
    """
    return np.matmul(mass,pos)/np.sum(mass)

def centre_of_mass_vel(vel,mass):
    """
    Calculates the centre of mass velocity using input velocities and masses
    """
    if np.sum(mass) == 0:
        return 0 
    else: 
        return np.matmul(mass,vel)/np.sum(mass)

def halo_filter(pos,vel,mass,radius,figure_name,show_com):
    """
    Mask the particle positions and velocities inside the halo such that only those within a certain radius are included. 

    Parameters:
        pos: particle positions 
        mass: particle masses 
        vel: particle velocities
        radius: radius of the spherical region we define as within the halo
        figure_name: string input for the name of the produced figure
        show_com: True or False, whether or not to display the CoM 

    Returns: 
        pos_masked: particle positions within the radius
        vel_masked: particle velocities within the radius
    """

    CoM = centre_of_mass(pos,mass) #calculate the centre of mass 
    distances = np.linalg.norm(pos - CoM, axis=1)

    print('Using a radius of {} kpc to mask the positions and velocities...'.format(1000*radius))
    pos_masked = pos[distances < radius] #mask the halo positions: include only the positions within the radius
    mass_masked = mass[distances < radius]
    vel_masked = vel[distances < radius]

    #check if there are any positions at all, if not return None 
    if len(pos_masked[:,0]) == 0:
        print('There are no particles within {} kpc!'.format(1000*radius))
        return None,None,None,None,None,None

    ratio = len(pos_masked[:,0])/len(pos[:,0])
    print('{:.2f}% of the initial particle sample is used!'.format(100*ratio))

    CoM_vel = centre_of_mass_vel(vel_masked,mass_masked)

    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(projection='3d')
    ax.scatter(pos_masked[:,0], pos_masked[:,1], pos_masked[:,2],s=5,zorder=1) 
    if show_com == True:
        ax.scatter(CoM[0],CoM[1],CoM[2],color='green',s=100,zorder=100) #centre of mass
    ax.set_xlim([CoM[0]-radius,CoM[0]+radius])
    ax.set_ylim([CoM[1]-radius,CoM[1]+radius])
    ax.set_zlim([CoM[2]-radius,CoM[2]+radius])
    ax.set_xlabel('x-position [Mpc]')
    ax.set_ylabel('y-position [Mpc]')
    ax.set_zlabel('z-position [Mpc]')
    fig.savefig(figure_name)
    plt.close()

    vel_dispersion = np.std(vel_masked) #take the standard deviation of the masked velocities to find the velocity dispersion 

    return pos_masked,vel_masked,mass_masked,CoM,CoM_vel,vel_dispersion

pos_mask_total = []
for i in range(10):
    print("-"*60)
    print('Halo number {}...'.format(i))
    pos,vel,mass = read_subhalo(i,"/L1000N1800/HYDRO_FIDUCIAL/VR/catalogue_0077/vr_catalogue_0077.catalog_groups.0",\
            "/L1000N1800/HYDRO_FIDUCIAL/VR/catalogue_0077/vr_catalogue_0077.catalog_particles.0")
    pos_mask,vel_mask,mass_mask,com,com_vel,vel_disp = halo_filter(pos,vel,mass,0.05,'halo{}_50kpc.png'.format(i),False)

    #add all the masked postions to the main position halo 
    if pos_mask is not None:
        for j in range(len(pos_mask)):
            pos_mask_total.append(pos_mask[i,:].tolist())

pos_mask_total = np.array(pos_mask_total)
print(pos_mask_total.shape)
np.savetxt('haloPos_0_10.txt',pos_mask_total)


    
    
    
    