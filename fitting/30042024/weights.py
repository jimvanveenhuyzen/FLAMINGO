import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors

def load_files(file):
    path = '/net/draco/data2/vanveenhuyzen/rsd_project/nbodykit_sourcecode/griddata/'
    m_pos = np.load(path+'grid_pos_{}.npy'.format(file))
    m_rsd = np.load(path+'grid_rsd_{}.npy'.format(file))
    return m_pos,m_rsd,(m_rsd/m_pos)

#Load in the data: 
mall_pos,mall_rsd,mall_div = load_files('mall')

fig, ax = plt.subplots(figsize=(10,10))
cax1 = ax.imshow(mall_rsd, origin='lower',extent=(0,1,0,1),cmap='nipy_spectral',norm=matplotlib.colors.LogNorm(vmin=50,vmax=1.1e5))
ax.set_xlabel(r'$k = \sqrt{k_x^2 + k_y^2}$')
ax.set_ylabel(r'$k_z$')
ax.set_title(r'$P_{RSD}/P_{real}$ for all galaxies')
ax.grid(visible=True)
fig.colorbar(cax1,label=r'$P_{RSD}(k)/P_{Pos}(k)$')
plt.savefig('mall_rsd.png')
plt.show()
plt.close()

factor = 2
No = 64 
Nn = No//factor

avg = np.zeros((32,32))
weights = np.zeros((64,64))
for i in range(Nn):
    for j in range(Nn):
        tl,tr,bl,br = mall_rsd[2*i,2*j],mall_rsd[2*i,2*j+1],mall_rsd[2*i+1,2*j],mall_rsd[2*i+1,2*j+1]
        mean = 0.25*(tl+tr+bl+br)

        diff_sq = (np.array([tl,tr,bl,br])-mean)**2
        var = np.mean(diff_sq)
        print(var)

        avg[i,j] = mean 
        weights[2*i,2*j],weights[2*i,2*j+1],weights[2*i+1,2*j],weights[2*i+1,2*j+1] = 1/var,1/var,1/var,1/var 

fig, ax = plt.subplots(figsize=(10,10))
cax1 = ax.imshow(avg, origin='lower',extent=(0,1,0,1),cmap='nipy_spectral',norm=matplotlib.colors.LogNorm(vmin=50,vmax=1.1e5))
ax.set_xlabel(r'$k = \sqrt{k_x^2 + k_y^2}$')
ax.set_ylabel(r'$k_z$')
ax.set_title(r'$P_{RSD}/P_{real}$ for all galaxies')
ax.grid(visible=True)
fig.colorbar(cax1,label=r'$P_{RSD}(k)/P_{Pos}(k)$')
plt.savefig('mall_rsd_32x.png')
plt.show()

fig, ax = plt.subplots(figsize=(10,10))
cax1 = ax.imshow(weights, origin='lower',extent=(0,1,0,1),cmap='nipy_spectral',norm=matplotlib.colors.LogNorm(vmin=1e-4,vmax=1))
ax.set_xlabel(r'$k = \sqrt{k_x^2 + k_y^2}$')
ax.set_ylabel(r'$k_z$')
ax.set_title(r'$P_{RSD}/P_{real}$ for all galaxies')
ax.grid(visible=True)
fig.colorbar(cax1,label=r'$P_{RSD}(k)/P_{Pos}(k)$')
plt.savefig('weights_rsd.png')
plt.show()

np.save('fitting/weights_2804.npy',weights)

print(mall_pos[-1,:])

