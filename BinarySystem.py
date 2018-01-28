
# coding: utf-8

# In[1]:

import rebound
import matplotlib.pyplot as plt
import numpy as np

n = 100;

sim = rebound.Simulation()
sim.add(m=1.1) #Alpha Centauri A
sim.add(m=0.9, a=11) #Alpha Centauri B

semimajor = np.random.uniform(0.0,20.0,n)
incl = np.random.uniform(0.0,np.pi/2,n)
ecc = np.random.random(n)
for i in range(n):
    sim.add(m=1e-9, a=semimajor[i], e=ecc[i], T=i, inc=incl[i])
sim.status()


# In[2]:

#Static Orbit Plot
sim.move_to_com()
rebound.OrbitPlot(sim, lim=25, figsize=(10,10), trails=True, lw=0.3, slices=True)


# In[3]:

# Animated Orbit
w = sim.getWidget(size=(500,500))
w


# In[4]:

#Scatter Plot
plt.clf()
a = np.zeros(n-1) #semimajor axis x-axis [0,0,0,0,0...] len=102
e = np.zeros(n-1) #eccentricty y-axis
for i in range(1,len(a)): #iterate through i = 0, i<102, i++
    a[i] = sim.particles[i].calculate_orbit(sim.particles[0]).a #sim.calculate_orbits(particle[i])
    e[i] = sim.particles[i].calculate_orbit(sim.particles[0]).e
fig, ax = plt.subplots()
ax.scatter(a,e, s=1)
plt.xlim(0, 30)
plt.ylim(0, 2)
plt.show()


# In[ ]:

sim.integrator = "whfast"
sim.dt = 1.e-2 #sets timestep
sim.integrate(10000) #run to time t=100


# In[ ]:

# After integrate use any of the cell above to plot

