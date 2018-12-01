import quantities as pq
import numpy as np
import matplotlib.pyplot as plt
import pylgn.tools as tls
import pylgn
import pylgn.kernels as kernel

responseGanglionOn = np.zeros([11,1])
responseGanglionOff = np.zeros([11,1])
responseRelayOn = np.zeros([11,1])
responseRelayOff = np.zeros([11,1])
for size in [0,0.2,0.4,0.6,0.8,1,1.5,2,3,4,6,8,10]:

    # create network
    network = pylgn.Network()
    # create integrator
    integrator = network.create_integrator(nt=7, nr=7, dt=1*pq.ms, dr=0.1*pq.deg)

    # create spatial kernels
    Wg_r_on = kernel.spatial.create_dog_ft(A=1, a=0.62*pq.deg, B=0.85, b=1.26*pq.deg)
    Wg_r_off = kernel.spatial.create_dog_ft(A=-1, a=0.62*pq.deg, B=-0.85, b=1.26*pq.deg)
    Krig_r = kernel.spatial.create_gauss_ft(A=1.2, a=0.88*pq.deg)
    Krg_r = kernel.spatial.create_gauss_ft()

    # create temporal kernels
    ''''
    # create stimulus
    stimulus = pylgn.stimulus.create_flashing_spot(contrast=10, patch_diameter=3*pq.deg,
                                delay=5*pq.ms, duration=50*pq.ms)
    network.set_stimulus(stimulus,True)
    '''
    # create neuron
    ganglionOn = network.create_ganglion_cell(background_response=3/pq.s)
    ganglionOff = network.create_ganglion_cell(background_response=3/pq.s)
    #relayOn = network.create_relay_cell(background_response=1/pq.s)
    #relayOff = network.create_relay_cell(background_response=1/pq.s)

    # set kernels
    ganglionOn.set_kernel((Wg_r_on, kernel.temporal.create_biphasic_ft(phase=10*pq.ms, damping=0.2*(1+0.4*size), delay=20*(1.1-0.05*size)*pq.ms)))
    #ganglionOn.set_kernel((Wg_r_on, kernel.temporal.create_exp_decay_ft(tau=10*pq.ms,delay=20*(1.1-0.05*size)*pq.ms)))
    ganglionOff.set_kernel((Wg_r_off,
                            kernel.temporal.create_biphasic_ft(phase=5*pq.ms, damping=0.8, delay=20*(1.1-0.05*size)*pq.ms)))

    # connect neurons
    #network.connect(ganglionOn,relayOn, (Krg_r, kernel.temporal.create_exp_decay_ft(3*pq.ms,0*pq.ms)), weight=0.81)
    #network.connect(ganglionOn,relayOn, (Krig_r, kernel.temporal.create_exp_decay_ft(6*pq.ms,0*pq.ms)), weight=-0.56)
    #network.connect(ganglionOff,relayOff, (Krg_r, kernel.temporal.create_exp_decay_ft(3*pq.ms,0*pq.ms)), weight=0.81)
    #network.connect(ganglionOff,relayOff, (Krig_r, kernel.temporal.create_exp_decay_ft(6*pq.ms,0*pq.ms)), weight=-0.56)

    network.connect(ganglionOn,ganglionOff,(Krig_r, kernel.temporal.create_exp_decay_ft(6*pq.ms,50*pq.ms)),weight=0)

    # set stimulus
    stimulus = pylgn.stimulus.create_flashing_spot(contrast=10, patch_diameter=size*pq.deg,
                            delay=5*pq.ms, duration=50*pq.ms)
    network.set_stimulus(stimulus,True)

    # compute stimulus
    network.compute_response(ganglionOn)
    network.compute_response(ganglionOff)
    #network.compute_response(relayOn)
    #network.compute_response(relayOff)

    plt.plot(range(0,len(ganglionOn.center_response)),ganglionOn.center_response, label="On cell with patch size={}".format(size))
plt.legend()
plt.show()


# visualize

# pylgn.plot.animate_cube(ganglionOn.response, title="Ganglion On cell response")
# pylgn.plot.animate_cube(ganglionOff.response, title="Ganglion Off cell response")
# pylgn.plot.animate_cube(relayOn.response, title="Relay On cell response")
# pylgn.plot.animate_cube(relayOff.response, title="Relay Off cell response")


'''
# apply static nonlinearity and scale rates
rates = tls.heaviside_nonlinearity(relayOff.response)
rates = tls.scale_rates(rates, 100*pq.Hz)

# generate spike trains
spike_trains = tls.generate_spike_train(rates, integrator.times)

# visulize
pylgn.plot.animate_spike_activity(spike_trains,
                                  times=integrator.times,
                                  positions=integrator.positions,
                                  title="Spike activity")
'''