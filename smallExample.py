import pylgn
import numpy as np
import quantities as pq

create=pylgn.stimulus.create_flashing_spot(contrast=0.2, patch_diameter=2*pq.deg,
                         delay=0*pq.ms, duration=10*pq.ms)
x=np.arange(-3,3,0.1).tolist()
y=x
t=np.arange(0,20,1).tolist()
Z = np.zeros([len(t),len(x), len(y)])



for dt in t:
    for i in x:
        for j in y:
            Z[t.index(dt)][x.index(i)][y.index(j)] = create(dt*pq.ms, i*pq.deg, j*pq.deg)

pylgn.plot.animate_cube(Z)

'''
X, Y = np.meshgrid(x, y)
R = np.sqrt(X ** 2 + Y ** 2)
R = np.sin(R)

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = Axes3D(fig)
ax.plot_surface(X, Y, Z)
plt.show()
pylgn.plot.animate_cube()

'''