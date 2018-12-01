
import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
import pylgn
from pylgn import kernels as kernel
import quantities as pq
import testFunction as tf
import time as tm

# initial setup
since = tm.time()
time = 9
fieldSize = 7
matrixSize = [13, int(2**time)]
matrix = np.zeros(matrixSize)
patchSize = [0, 0.2, 0.4, 0.6, 0.8, 1., 1.5, 2., 3., 5., 6., 8., 10.]

# fetch data from allFinal
data = sio.loadmat('allFinal.mat')
cell = data['allcell']['offsta'][0][0][1][0][:, 0:int(2**time)]

# create dict for parameters

paraDict_tp = {'excPhase': 50., 'excPhaseDelay': 20., 'inhPhase': 80., 'inhPhaseDelay': 40., 'excDamping': 0.4, 'inhDamping': 0.8}
paraDict_sp = {'a': 0.62, 'B': 0.85, 'b': 1.26}
paraDict_weight = {'offWeight': 0.5}

# loop


# generate matrix
for size in patchSize:

    # create network
    network = pylgn.Network()

    # create integrator
    integrator = network.create_integrator(nt=time, nr=fieldSize, dt=1 * pq.ms, dr=0.1 * pq.deg)

    # create and set stimulus
    stimulus = pylgn.stimulus.create_flashing_spot(contrast=10, patch_diameter=size * pq.deg,
                                                   delay=0 * pq.ms, duration=500 * pq.ms)
    network.set_stimulus(stimulus, True)

    # create kernel function
    # temporal
    tp_bi_exc = kernel.temporal.create_biphasic_ft(phase=50 * (1.0 - 0.05 * size) * pq.ms, damping=0.4,
                                                   delay=20 * (1.0 - 0.02 * size) * pq.ms)
    tp_bi_inh = kernel.temporal.create_biphasic_ft(phase=80 * (1.0 - 0.05 * size) * pq.ms, damping=0.8,
                                                   delay=40 * (1.0 - 0.02 * size) * pq.ms)

    # spatial
    Wg_r_on = kernel.spatial.create_dog_ft(A=1, a=0.62 * pq.deg, B=0.85, b=1.26 * pq.deg)
    Wg_r_off = kernel.spatial.create_dog_ft(A=-1, a=0.62 * pq.deg, B=-0.85, b=1.26 * pq.deg)

    # create neuron
    receptNeuron = network.create_ganglion_cell()
    ganglionOn = network.create_relay_cell(background_response=6 / pq.s)

    # connect neuron
    network.connect(receptNeuron, ganglionOn, (Wg_r_on, tp_bi_exc), weight=1 * size ** 0.8)
    network.connect(receptNeuron, ganglionOn, (Wg_r_off, tp_bi_inh), weight=0.5 * size ** 0.8)

    # set kernel function for neurons
    receptNeuron.set_kernel((kernel.spatial.create_delta_ft(), kernel.temporal.create_delta_ft()))

    # network.compute_response(receptNeuron)
    network.compute_response(ganglionOn)

    # calculate centre response
    matrix[patchSize.index(size)] = ganglionOn.center_response

# normalization process for 2 matrices
matrix = matrix/tf.find_max(matrix)
cell = cell/tf.find_max(cell)
'''
# calculate R square
RSquare = 0
for i in range(0,13):
    slope, intercept, r_value, p_value, std_err = st.linregress(matrix[i], cell[i])
    RSquare = RSquare + r_value**2
RSquare = RSquare/13
print('R square')
print(RSquare)

# calculate correlation
correlation = 0
for i in range(0,13):
    s1 = pd.Series(cell[i])
    s2 = pd.Series(matrix[i])
    correlation = correlation + s2.corr(s1)
correlation = correlation/13
print('correlation coefficient')
print(correlation)

'''
# time
now = tm.time()-since
print('time consume')
print(now)

# plotting
X, Y = np.meshgrid(range(0,int(2**time)),patchSize)
fig = plt.figure()
ax1 = fig.add_subplot(121)
ax1.contourf(X, Y, cell)
ax2 = fig.add_subplot(122)
ax2.contourf(X, Y, matrix)

plt.show()



