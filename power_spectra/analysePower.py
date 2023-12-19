import numpy as np
import matplotlib.pyplot as plt
import h5py

k_val = np.load('kVAL_51.npy')
Pk_val = np.load('pkVAL_51.npy')
mu_val = np.load('muVAL_51.npy')

k_val = np.load('k_val_avg.npy')
Pk_val = np.load('Pk_val_avg.npy')
mu_val = np.load('mu_val_avg.npy')


print(k_val.shape)
print(Pk_val.shape)

k_value_80 = np.nanmean(k_val[:,10])
#print(k_val[:,10])
Pk_val_k80 = Pk_val[:,10]

plt.plot(mu_val,Pk_val_k80)
plt.xlabel('mu values')
plt.ylabel('Pk values')
#plt.yscale('log')
plt.title('Plot of the Pk values as a function of mu at k={:.2f}'.format(k_value_80))
plt.savefig('Pk_mu_19122023.png')

#Plot the full power spectrum for various mu values:
plt.plot(k_val,Pk_val)
plt.xlabel('k values')
plt.ylabel('Pk values')
plt.yscale('log')
plt.title('Plot of the whole power spectrum')
plt.savefig('fullspectrum_19122023.png')