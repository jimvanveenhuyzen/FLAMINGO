import numpy as np
import matplotlib.pyplot as plt 

power_real = np.genfromtxt('/net/draco/data2/vanveenhuyzen/rsd_project/nbodykit_sourcecode/power_spectra/power_real_0122.txt')
print(power_real)
power_matter = np.genfromtxt('/net/hypernova/data2/FLAMINGO/L1000N1800/HYDRO_FIDUCIAL/power_spectra/power_matter_0122.txt')
print(power_matter)
print(power_matter.shape)

power_realk = power_real[:,0]
power_realPk = power_real[:,1]
power_realSN = power_real[0,2]

power_matterk = power_matter[:,1]
power_matterPk = power_matter[:,2]

#Now, lets interpolate the spectrum we made so it overlaps with the k-values from the power_matter spectrum
power_matterlowk = power_matter[:100,1]

power_realInterp = np.interp(power_matterlowk,power_realk,power_realPk)

xlow,xhigh = np.min(power_matterk),np.max(power_realk)

plt.loglog(power_matterk,power_matterPk,label='matter')
#plt.loglog(power_realk,power_realPk,label='real')
plt.loglog(power_matterlowk,power_realInterp,label='real (interpolated)')
plt.xlim(xlow,xhigh)
plt.ylim(1e2,np.max(power_matterPk)*1.5)
plt.xlabel('k')
plt.ylabel('P(k)')
plt.legend()
plt.show()

#Use low k values
power_realPklowk = power_realInterp[np.where(power_matterlowk < 0.15)]
power_matterPklowk = power_matterPk[np.where(power_matterlowk < 0.15)]

#Compute the bias per k and take the mean value 
bias = np.sqrt(power_realPklowk/power_matterPklowk)
print(np.mean(bias))

