import numpy as np
import matplotlib.pyplot as plt
import h5py
from matplotlib.patches import Circle

positions = np.genfromtxt('positions_halo0.txt',delimiter='')

path_hydro = "/net/hypernova/data2/FLAMINGO/L1000N1800/DMO_FIDUCIAL/SOAP/halo_properties_0077.hdf5"
with h5py.File(path_hydro, "r") as handle:
    R200m = handle["SO/200_mean/SORadius"][:]
    R500c = handle["SO/500_crit/SORadius"][:]
    center = handle["VR/CentreOfPotential"][:]
    
R200m_halo0 = R200m[0]
R500c_halo0 = R500c[0]

halo0_center = (np.mean(positions[:,0]),np.mean(positions[:,1]))
circle_200m = Circle(halo0_center,R200m_halo0,facecolor='gray',edgecolor='black',alpha=0.2,label='200m')
circle_500c = Circle(halo0_center,R500c_halo0,facecolor='green',edgecolor='black',alpha=0.2,label='500c') 

fig, ax = plt.subplots()
ax.scatter(positions[:,0],positions[:,1],s=1,label='CDM')
ax.add_patch(circle_200m)
ax.add_patch(circle_500c)
ax.set_aspect('equal')
ax.set_xlabel('x coordinate [Mpc]')
ax.set_ylabel('y coordinate [Mpc]')
ax.set_title('Dark matter particles inside halo 0')
ax.set_title('x-y positions of the CDM- and stellar particles in halo 0')
ax.legend()
fig.savefig('halo0_DMO.png')
plt.close()

