import numpy as np
import matplotlib.pyplot as plt

data2y = np.loadtxt('WaveIncan.txt')[:,1]
data2x = np.loadtxt('WaveIncan.txt')[:,0]
plt.plot(data2x, data2y)
plt.xlabel('Wavelength(NM)', fontsize = 17)
plt.ylabel('Intensity (Counts)', fontsize = 17)
plt.title('Incandescent Lamp', fontsize = 17)
plt.show()

data2y = np.loadtxt('Waveflour.txt')[:,1]
data2x = np.loadtxt('Waveflour.txt')[:,0]
plt.plot(data2x, data2y)
plt.xlabel('Wavelength(NM)', fontsize = 17)
plt.ylabel('Intensity (Counts)', fontsize = 17)
plt.title('Fluorescent Strip Light', fontsize = 17)
plt.show()

data11y = np.loadtxt('WaveGreenLight.txt')[:,1]
data11x = np.loadtxt('WaveGreenLight.txt')[:,0]
plt.plot(data11x, data11y)
plt.xlabel('Wavelength(NM)', fontsize = 17)
plt.ylabel('Intensity (Counts)', fontsize = 17)
plt.title('Green Light', fontsize = 17)
plt.show()

data8y = np.loadtxt('WaveNeon.txt')[:,1]
data8x = np.loadtxt('WaveNeon.txt')[:,0]
plt.plot(data8x, data8y)
plt.xlabel('Wavelength(NM)', fontsize = 17)
plt.ylabel('Intensity (Counts)', fontsize = 17)
plt.title('Neon Gas Discharge Lamp', fontsize = 17)
plt.show()

data13y = np.loadtxt('WaveSunlight.txt')[:,1]
data13x = np.loadtxt('WaveSunlight.txt')[:,0]
plt.plot(data13x, data13y)
plt.xlabel('Wavelength(NM)', fontsize = 17)
plt.ylabel('Intensity (Counts)', fontsize = 17)
plt.title('Sunlight', fontsize = 17)
plt.show()

####################################################################################################################

