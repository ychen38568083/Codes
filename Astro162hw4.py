import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.signal import argrelextrema
import math

data1 = np.loadtxt('HW4_prob1_ice_core_data.csv',delimiter =',',dtype='int32')[:,0] # Sameple Number
data2 = np.loadtxt('HW4_prob1_ice_core_data.csv',delimiter =',',dtype='int32')[:,1] # Depth Below Glacier surface - samples at the top of the glacier are younger than at the bottom.
data3 = np.loadtxt('HW4_prob1_ice_core_data.csv',delimiter =',',dtype='int32')[:,2] # Derived age
data4 = np.loadtxt('HW4_prob1_ice_core_data.csv',delimiter =',',dtype='int32')[:,3] # Measured fractional deuterium issotope content
data5 = np.loadtxt('HW4_prob1_ice_core_data.csv',delimiter =',',dtype='int32')[:,4] # Derived Temperature

plt.plot(data3, data4)

plt.xlabel('Age (Years)')
plt.ylabel('Deuterium Content')
plt.title('Deuterium Content vs Age')
plt.show()

plt.plot(data3, data5)


a =0
a1 =4
a2 =3
p1 =216
p2 =100000
p3 =27000
To = -4
x = np.arange(0,800000)
T = a*np.sin((2*np.pi*x)/p1) + a1*np.sin((2*np.pi*x)/p2) + a2*np.sin((2*np.pi*x)/p3) + To

plt.plot(T, 'k-')
plt.xlabel('Age (Years)')
plt.ylabel('Temperature')
plt.title('Actual data vs Theoratical Prediction')
plt.show()

x = (59000/ np.sqrt(((1.38*10**(-23))*900)/(1.6737*10**(-27))))**2
y = (2*np.sqrt(np.pi)*np.exp(x))/(np.sqrt(((1.38*10**(-23))*900)/(1.6737*10**(-27)))*(1+x))
print(x)
print(y)