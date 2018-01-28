import numpy as np
import matplotlib.pyplot as plt

G = 6.67408e-11 # in m^3⋅kg^−1⋅s^−2
M = 1.989e30 # sun's mass in kg
r1 = 149597870700 # 1 AU in meter
dvr = sqrt((GM(r-r1))/(r*r1))

plt.plot(dvr)
plt.show()