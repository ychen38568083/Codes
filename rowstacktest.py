import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.signal import argrelextrema
import math
from pylab import *


fluorpixelx = np.loadtxt('flour.txt')[:,0]
fluorpixely = np.loadtxt('flour.txt')[:,1]

#Background Noises in pixel and wavelength for calibration
########################################################################################################################################################################


bias1 = np.genfromtxt('./data/bias_3ms/bias00.txt',skip_header=17, skip_footer=1)[:,1]
bias2 = np.genfromtxt('./data/bias_3ms/bias01.txt',skip_header=17, skip_footer=1)[:,1]
bias3 = np.genfromtxt('./data/bias_3ms/bias02.txt',skip_header=17, skip_footer=1)[:,1]
bias4 = np.genfromtxt('./data/bias_3ms/bias03.txt',skip_header=17, skip_footer=1)[:,1]
bias5 = np.genfromtxt('./data/bias_3ms/bias04.txt',skip_header=17, skip_footer=1)[:,1]
bias6 = np.genfromtxt('./data/bias_3ms/bias05.txt',skip_header=17, skip_footer=1)[:,1]
bias7 = np.genfromtxt('./data/bias_3ms/bias06.txt',skip_header=17, skip_footer=1)[:,1]
bias8 = np.genfromtxt('./data/bias_3ms/bias07.txt',skip_header=17, skip_footer=1)[:,1]
bias9 = np.genfromtxt('./data/bias_3ms/bias08.txt',skip_header=17, skip_footer=1)[:,1]
bias10 = np.genfromtxt('./data/bias_3ms/bias09.txt',skip_header=17, skip_footer=1)[:,1]

bg1 = np.loadtxt('BG1.txt')[:,1]
bg2 = np.loadtxt('BG2.txt')[:,1]
bg3 = np.loadtxt('BG3.txt')[:,1]
bg4 = np.loadtxt('BG4.txt')[:,1]
bg5 = np.loadtxt('BG5.txt')[:,1]
bg6 = np.loadtxt('BG6.txt')[:,1]
bg7 = np.loadtxt('BG7.txt')[:,1]
bg8 = np.loadtxt('BG8.txt')[:,1]
bg9 = np.loadtxt('BG9.txt')[:,1]
bg10 = np.loadtxt('BG10.txt')[:,1]

meanbg = (bg1 +bg2 +bg3 +bg4 +bg5 +bg6 +bg7 +bg8 +bg9 +bg10) / 10


meanbias = (bias1+bias2+bias3+bias4+bias5+bias6+bias7+bias8+bias9+bias10)/10

def peak_finder(signal_array):
    threshold = 200
    peaks = []                            #x positions of the peaks, or rather, their index
    for i in range(2,len(signal_array)-2): 
        if signal_array[i] > signal_array[i+2]  and signal_array[i] > signal_array[i+1] and signal_array[i] > signal_array[i-1] and signal_array[i] > signal_array[i-2]:  #four conditions to be a peak (see description)
            if signal_array[i] > threshold:                      #is the value of the spectrum at i higher than our threshold?
                peaks.append(i)
    return peaks
    
#print('peaks at:',peak_finder(data1y))
#print('peak intensities are:', peak_finder(data1y))

def centroid(x_range,y_range):
    '''A function to return the centroid given equally sized x and y ranges over which to perform the calculation'''
    x_range = np.array(x_range) #make sure these are arrays if they aren't already
    y_range = np.array(y_range) #make sure these are arrays if they aren't already
    ... #convert the math formula for a centroid into code in these lines
    x_centroid = sum(x_range*y_range)/(sum(y_range))
    return x_centroid

def find_all_centroids(x_range,y_range):
	peaks = peak_finder(y_range) #define the peak positions in x indicies
	multicen = [] #empty array to append
	for i in peaks: #for loops for indicies in peaks
		y = y_range[i] #define the y which uses the y-axis indicies
		halfmax = y/2 #half of each peaks
		#print(halfmax)
		#multicen.append(centroid(x_range[i-4:i+4],y_range[i-4:i+4]))
		#The following codes are for more general way:
		dr = np.where(y_range[i:] < halfmax)[0][0] # everything to the right after half of each peaks
		dl = np.where(y_range[:i] < halfmax)[0][-1] # everything to the left
		multicen.append(centroid(x_range[dl:i+dr], y_range[dl:i+dr])) #append centroid back
	return multicen #returns multicen = [] with each newl updated centroid

########################################################################################################################################################################

neon0 = np.genfromtxt('./data/neon_16ms/red00.txt',skip_header=17, skip_footer=1)[:,1]-meanbg
neon1 = np.genfromtxt('./data/neon_16ms/red01.txt',skip_header=17, skip_footer=1)[:,1]-meanbg
neon2 = np.genfromtxt('./data/neon_16ms/red02.txt',skip_header=17, skip_footer=1)[:,1]-meanbg
neon3 = np.genfromtxt('./data/neon_16ms/red03.txt',skip_header=17, skip_footer=1)[:,1]-meanbg
neon4 = np.genfromtxt('./data/neon_16ms/red04.txt',skip_header=17, skip_footer=1)[:,1]-meanbg
neon5 = np.genfromtxt('./data/neon_16ms/red05.txt',skip_header=17, skip_footer=1)[:,1]-meanbg
neon6 = np.genfromtxt('./data/neon_16ms/red06.txt',skip_header=17, skip_footer=1)[:,1]-meanbg
neon7 = np.genfromtxt('./data/neon_16ms/red07.txt',skip_header=17, skip_footer=1)[:,1]-meanbg
neon8 = np.genfromtxt('./data/neon_16ms/red08.txt',skip_header=17, skip_footer=1)[:,1]-meanbg
neon9 = np.genfromtxt('./data/neon_16ms/red09.txt',skip_header=17, skip_footer=1)[:,1]-meanbg
neon10 = np.genfromtxt('./data/neon_16ms/red21.txt',skip_header=17, skip_footer=1)[:,1]-meanbg
neon11 = np.genfromtxt('./data/neon_16ms/red22.txt',skip_header=17, skip_footer=1)[:,1]-meanbg
neon12 = np.genfromtxt('./data/neon_16ms/red23.txt',skip_header=17, skip_footer=1)[:,1]-meanbg
neon13 = np.genfromtxt('./data/neon_16ms/red24.txt',skip_header=17, skip_footer=1)[:,1]-meanbg
neon14 = np.genfromtxt('./data/neon_16ms/red25.txt',skip_header=17, skip_footer=1)[:,1]-meanbg
neon15 = np.genfromtxt('./data/neon_16ms/red26.txt',skip_header=17, skip_footer=1)[:,1]-meanbg
neon16 = np.genfromtxt('./data/neon_16ms/red27.txt',skip_header=17, skip_footer=1)[:,1]-meanbg
neon17 = np.genfromtxt('./data/neon_16ms/red28.txt',skip_header=17, skip_footer=1)[:,1]-meanbg
neon18 = np.genfromtxt('./data/neon_16ms/red29.txt',skip_header=17, skip_footer=1)[:,1]-meanbg
neon19 = np.genfromtxt('./data/neon_16ms/red30.txt',skip_header=17, skip_footer=1)[:,1]-meanbg
neon20 = np.genfromtxt('./data/neon_16ms/red31.txt',skip_header=17, skip_footer=1)[:,1]-meanbg
neon21 = np.genfromtxt('./data/neon_16ms/red32.txt',skip_header=17, skip_footer=1)[:,1]-meanbg
neon22 = np.genfromtxt('./data/neon_16ms/red33.txt',skip_header=17, skip_footer=1)[:,1]-meanbg
neon23 = np.genfromtxt('./data/neon_16ms/red34.txt',skip_header=17, skip_footer=1)[:,1]-meanbg
neon24 = np.genfromtxt('./data/neon_16ms/red35.txt',skip_header=17, skip_footer=1)[:,1]-meanbg
neon25 = np.genfromtxt('./data/neon_16ms/red36.txt',skip_header=17, skip_footer=1)[:,1]-meanbg
neon26 = np.genfromtxt('./data/neon_16ms/red37.txt',skip_header=17, skip_footer=1)[:,1]-meanbg
neon27 = np.genfromtxt('./data/neon_16ms/red38.txt',skip_header=17, skip_footer=1)[:,1]-meanbg
neon28 = np.genfromtxt('./data/neon_16ms/red39.txt',skip_header=17, skip_footer=1)[:,1]-meanbg
neon29 = np.genfromtxt('./data/neon_16ms/red40.txt',skip_header=17, skip_footer=1)[:,1]-meanbg
neon30 = np.genfromtxt('./data/neon_16ms/red41.txt',skip_header=17, skip_footer=1)[:,1]-meanbg
neon31 = np.genfromtxt('./data/neon_16ms/red42.txt',skip_header=17, skip_footer=1)[:,1]-meanbg

col1stack = np.vstack((neon0, neon1, neon2, neon3,neon4, neon5, neon6,neon7,neon8, neon9, neon10,neon11,neon12, neon13, neon14,neon15,neon16, neon17, neon18,neon19, neon20, neon21, neon22, neon23,neon24,neon25, neon26, neon27, neon28, neon29, neon30, neon31))

transposecol1 = np.transpose(col1stack)
print(transposecol1)
print(np.arange(len(transposecol1)))

means = np.array([])
for i in np.arange(len(transposecol1)):
    mean = np.mean(transposecol1[i])
    means = np.append(means, mean)

variances = np.array([])
for i in np.arange(len(transposecol1)):
    variance = np.var(transposecol1[i])
    variances = np.append(variances, variance)
print('variances=', variances)
print('means = ', means)


tfit, tcov = np.polyfit(means, variances, 2, full=False, cov = True)
tp = np.poly1d(tfit)
tys = tp(means)
plt.plot(means, variances, 'o')
plt.title('Variances vs. Means', fontsize = 17)
plt.xlabel('Means [ADU]', fontsize = 17)
plt.ylabel('Variances [ADU^2]', fontsize = 17)
plt.plot(means, tys, 'k-')
plt.xscale('log')
plt.yscale('log')
plt.show()

print('Covarience for neon:', tcov)
print('Variance of Slope for neon: ', tcov[0][0])
print('Standard Deviation of Slope for neon:', np.sqrt(tcov[0][0]))

print('Variance of y-intercept for neon:', tcov[1][1])
print('Standard Deviation of y-intercept for neon:', np.sqrt(tcov[1][1]))
print('slope for neon:', tp[1])
print('Y-intercept: ', tp[0])
print('Coeff for neon:', tfit)