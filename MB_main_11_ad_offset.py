import numpy as np
import pylab as plt
#import spynnaker8 as sim
from pyNN.utility.plotting import Figure, Panel
from Connection_Generator_4 import *
import scipy.io as sio
import requests
import csv
import pyNN.utility.plotting as plot
from pyNN.utility import Timer
import matplotlib.pyplot as plt
try:
    import pyNN.spiNNaker as sim
except ImportError:
    import spynnaker8 as sim
from datetime import datetime
import os
# Import the extra classes
from spynnaker.pyNN.external_devices_models import (
    AbstractEthernetTranslator, AbstractMulticastControllableDevice)
from spynnaker.pyNN.external_devices_models\
    .abstract_multicast_controllable_device import SendType
from data_specification.enums import DataType
import DVS
import neo
import seaborn as sns
from quantities import ms, s, Hz
from elephant.statistics import mean_firing_rate
from elephant.statistics import time_histogram, instantaneous_rate
from matplotlib.gridspec import GridSpec
from pyNN.random import RandomDistribution, NumpyRNG
# class TestTranslator(AbstractEthernetTranslator):
#
#     def translate_control_packet(self, packet):
#         print("Received key={}, voltage={}".format(
#             packet.key, packet.payload / DataType.S1615.max))
#
# #from NoveltyIndex import *
# class TestDevice(AbstractMulticastControllableDevice):
#
#     @property
#     def device_control_partition_id(self):
#         # This should be unique to the device, but is otherwise unimportant
#         return "Test"
#
#     @property
#     def device_control_key(self):
#         # This should be unique to the device
#         return 0
#
#     @property
#     def device_control_uses_payload(self):
#         # This returns True to receive the voltage,
#         # or False if only the key is desired
#         return True
#
#     @property
#     def device_control_min_value(self):
#         # Return the minimum value accepted by the device.  If the membrane
#         # voltage is below this value, this value will be used.
#         return -70
#
#     @property
#     def device_control_max_value(self):
#         # Return the maximum value accepted by the device.  If the membrane
#         # voltage is above this value, this value will be used.
#         return 20
#
#     @property
#     def device_control_timesteps_between_sending(self):
#         # The number of timesteps between sending values.  Controls the
#         # update rate of the value.
#         return 10
#
#     @property
#     def device_control_send_type(self):
#         # The type of the value - one of the SendType values
#         return SendType.SEND_TYPE_ACCUM
#
#     @property
#     def device_control_scaling_factor(self):
#         # The amount to multiply the voltage by before transmission
#         return 1.0
np.random.seed(1)
timer = Timer()

#outputpath = '/media/le/121942f7-1af1-4374-bc84-6f60a0cc0259/PycharmProjects/Spinnaker/output/learning(0.5-0.6)PN2(840)KC5(2000)/'
figoutputpath = '/media/le/121942f7-1af1-4374-bc84-6f60a0cc0259/PycharmProjects/Spinnaker/output/231023/pngs01122022/'
recordoutputpath = '/media/le/121942f7-1af1-4374-bc84-6f60a0cc0259/PycharmProjects/Spinnaker/output/231023/records01122022/'

sns.color_palette("light:#5A9", as_cmap=True)

kpse_cp = subprocess.run(['kpsewhich', '-var-value', 'TEXMFDIST'], capture_output=True, check=True)
#font_loc1 = os.path.join(kpse_cp.stdout.decode('utf8').strip(), 'fonts', 'opentype', 'public', 'tex-gyre')
#print(f'loading TeX Gyre fonts from "{font_loc1}"')
#font_dirs = [font_loc1]
#font_files = font_manager.findSystemFonts(fontpaths=font_dirs)
#for font_file in font_files:
#    font_manager.fontManager.addfont(font_file)

#plt.rcParams['font.size'] = 16
#plt.rcParams['font.family'] = 'TeX Gyre Termes'
plt.rcParams["mathtext.fontset"] = "stix"
#from IPython.display import set_matplotlib_formats
#set_matplotlib_formats('svg')
plt.rcParams['svg.fonttype'] = 'path'


# SpiNNaker setup
node_id =  sim.setup(timestep=1.0, min_delay=1.0, max_delay=14)
# sim.set_number_of_neurons_per_core(sim.IF_curr_exp,50)

PN_cell_params = {
        'v_rest':   -60.0,  # Resting membrane potential in mV.
        'cm':         15.0*1e-3,  # Capacity of the membrane in nF  15pF
        'tau_m':     20.0,  # Membrane time constant in ms.
        'tau_refrac': 20.0,  # Duration of refractory period in ms.
        'tau_syn_E':  5.0,  # Decay time of excitatory synaptic current in ms.
        'tau_syn_I':  5.0,  # Decay time of inhibitory synaptic current in ms.
        #'i_offset':   0.015,  # Offset current in nA
        'i_offset':   0.01,  # Offset current in nA
        'v_reset':  -70.0,  # Reset potential after a spike in mV.
        'v_thresh': -40.0,  # Spike threshold in mV.
         'tau_ca2':  20.0,
         'i_ca2':    0.0,
         'i_alpha':  0.1,
}
KC_cell_params = {
        'v_rest':   -80.0,  # Resting membrane potential in mV.
        'cm':        6.0*1e-3,  # Capacity of the membrane in nF
        'tau_m':     10.0,  # Membrane time constant in ms.
        'tau_refrac': 20.0,  # Duration of refractory period in ms.
        'tau_syn_E':  1.5,  # Decay time of excitatory synaptic current in ms.
        'tau_syn_I':  0.5,  # Decay time of inhibitory synaptic current in ms.
        'i_offset':   0.0,  # Offset current in nA
        'v_reset':  -90.0,  # Reset potential after a spike in mV.
        'v_thresh': -40.0,  # Spike threshold in mV.
}
MBON_cell_params = {
        'v_rest':   -56.7,  # Resting membrane potential in mV.
        'cm':        16.06*1e-3,  # Capacity of the membrane in nF  15pF
        'tau_m':     16.0,  # Membrane time constant in ms.
        'tau_refrac': 4.0,  # Duration of refractory period in ms.
        'tau_syn_E':  1.0,  # Decay time of excitatory synaptic current in ms.
        'tau_syn_I':  1.0,  # Decay time of inhibitory synaptic current in ms.
        'i_offset':   0.0,  # Offset current in nA
        'v_reset':  -70.0,  # Reset potential after a spike in mV.
        'v_thresh': -35.0,  # Spike threshold in mV.
}
pn_ini_v_distr = RandomDistribution('normal', (PN_cell_params['v_rest'], 20.0), rng=NumpyRNG(seed=2022))
def plot_spikes(pn_spikes):

    t_start = pn_spikes.t_start
    t_stop = pn_spikes.t_stop

    inst_rate = instantaneous_rate(pn_spikes, sampling_period=10*ms)

    plt.figure(figsize=(8, 4), dpi=300)

    # plotting the original spiketrain
    plt.plot(pn_spikes, [0]*len(pn_spikes), 'black', alpha=0.5, marker=2, ms=10, markeredgewidth=1, lw=0, label='spike times')

    # # mean firing rate
    # plt.hlines(mean_firing_rate(pn_spikes), linestyle='--', label='mean firing rate')

    # time histogram
    # plt.bar(histogram_rate.times, histogram_rate.rescale(1/s).magnitude.flatten(), width=histogram_rate.sampling_period, align='edge', alpha=0.3, label='time histogram (rate)')

    # instantaneous rate
    plt.plot(inst_rate.times.rescale(ms), inst_rate.rescale(1/s).magnitude.flatten(), label='instantaneous rate')

    # axis labels and legend
    plt.xlabel('time [{}]'.format(pn_spikes.times.dimensionality.latex))
    plt.ylabel('firing rate (Hz)')
    plt.xlim(pn_spikes.t_start, pn_spikes.t_stop)
    plt.legend()
    plt.savefig(figoutputpath+str(datetime.now().strftime('%Y_%m_%d_%H_%M_%S'))+".svg", format='svg')
    plt.clf()
    plt.close()
def calculate_group_mean_rt(spike_trains,nb_n_group):
    single_mean_rate = []
    for i in np.arange(nb_n_group):
        if len(spike_trains[i]) == 0:
            single_mean_rate.append(0)
        else:
            single_spiketrain = neo.SpikeTrain(spike_trains[i], t_stop= spike_trains[i][-1], units='ms')
            single_mean_rate.append(mean_firing_rate(single_spiketrain))
    #print(single_mean_rate)
    group_mean_rate = np.nanmean(single_mean_rate)*1000
    return group_mean_rate
def file2input(filename, t_start=0, t_end=1):
    data = sio.loadmat(filename)
    idx_dvs = data['idx_p'][0]
    t_dvs = data['t_p'][0]
    t_dvs = t_dvs - min(t_dvs )
    max_t = np.max(t_dvs)

    t_start = t_start*max_t
    t_end = t_end*max_t

    index_start = (np.abs(t_dvs-t_start)).argmin()
    index_end = (np.abs(t_dvs-t_end)).argmin()

    n_list = idx_dvs.copy()[index_start:index_end]
    t_list = t_dvs.copy()[index_start:index_end]

    input_spikes = []
    for i in range(nb_pn):
        input_spikes.append(t_list[np.where(n_list==i)])

    sim_t = t_end-t_start
    return input_spikes,max_t
def odomfile2input(filename, t_start=0, t_end=1):
    data = sio.loadmat(filename)
    ex,ey,et = data['ex'][0],data['ey'][0],data['et'][0]
    x_c, y_c, t_c = DVS.chop_frame(ex, ey, et)
    idx, t = DVS.smooth(x_c, y_c, t_c)
    idx_dvs,t_dvs = DVS.align_start(idx, t)

    t_dvs = t_dvs -min(t_dvs )
    max_t = np.max(t_dvs)

    t_start = t_start*max_t
    t_end = t_end*max_t

    index_start = (np.abs(t_dvs-t_start)).argmin()
    index_end = (np.abs(t_dvs-t_end)).argmin()

    n_list = idx_dvs.copy()[index_start:index_end]
    t_list = t_dvs.copy()[index_start:index_end]

    input_spikes = []
    for i in range(nb_pn):
        input_spikes.append(t_list[np.where(n_list==i)])

    return input_spikes,max_t
class MB_LE(object):

    def __init__(self, filename, t_start=0, t_end=1 , learned = False, w_kc2kc=0):


        # if learned == False:
        #     sim.setup(timestep=3)
        # else:
        #     sim.setup(timestep=1)

        self.learned_w_kc2kc = None
        self.filename = filename
        self.t_start = t_start
        self.t_end = t_end
        input_spikes, self.sim_t = file2input(filename, t_start, t_end)
        self.sim_t = self.sim_t*t_end
        print ('simulation time=', self.sim_t)

        self.w_kc2kc = w_kc2kc
        self.learned = learned

        timer.start()

        self.spike_source = sim.Population(nb_pn, sim.SpikeSourceArray(spike_times = input_spikes),label = "DVS")
        #self.spike_source = sim.Population(nb_pn, sim.SpikeSourceArray(spike_times = [2000]),label = "DVS")

        # self.pns = sim.Population(nb_pn , sim.IF_curr_exp(**PN_cell_params),
        #                           label = "PN",initial_values={'v': PN_cell_params['v_rest']})
        # #
        self.pns = sim.Population(nb_pn, sim.extra_models.IFCurrExpCa2Adaptive(**PN_cell_params),
                                       label = "PN",
        #                 #initial_values={'v':pn_ini_v_distr})
                         initial_values={'v': PN_cell_params['v_rest']})

        self.kcs = sim.Population(nb_kc , sim.IF_curr_exp(**KC_cell_params) ,
                                  label = "KC",initial_values={'v': KC_cell_params['v_rest']})
        self.kcs_a = sim.Population(nb_kc ,sim.IF_curr_exp(**KC_cell_params),
                                    label = "KC-A",initial_values={'v': KC_cell_params['v_rest']})
        self.ens = sim.Population(nb_en , sim.IF_curr_exp(**MBON_cell_params) ,
                                  label = "MBON",initial_values={'v': MBON_cell_params['v_rest']})
        self.ens_a = sim.Population(nb_en , sim.IF_curr_exp(**MBON_cell_params),
                                    label = "MBON-A",initial_values={'v': MBON_cell_params['v_rest']})
        # self.ens = sim.external_devices.EthernetControlPopulation(
        #         n_neurons=nb_en,
        #         model=sim.external_devices.ExternalDeviceLifControl(
        #             devices=[TestDevice()],
        #             create_edges=False,
        #             translator=TestTranslator()),
        #         label="test")


        self.dvs2pn = sim.Projection(self.spike_source , self.pns , sim.OneToOneConnector() ,
                                     sim.StaticSynapse(weight = 2.0),
                                     receptor_type = 'excitatory')
        # self.pn2kc = sim.Projection(self.pns , self.kcs , sim.FromListConnector(S_pn2kc,  column_names=["weight"]),
        #                             receptor_type = 'excitatory')
        # self.pn2kc_a = sim.Projection(self.pns , self.kcs_a , sim.FromListConnector(S_pn2kc, column_names=["weight"]),
        #                             receptor_type = 'excitatory')
        self.pn2kc = sim.Projection(self.pns , self.kcs , sim.FromListConnector(S_pn2kc,  column_names=["weight"]),
                               receptor_type = 'excitatory')
        self.pn2kc_a = sim.Projection(self.pns , self.kcs_a ,sim.FromListConnector(S_pn2kc,  column_names=["weight"]),
                               receptor_type = 'excitatory')
        self.kc2en = sim.Projection(self.kcs , self.ens , sim.AllToAllConnector() ,sim.StaticSynapse(weight = 0.01) ,
                               receptor_type = 'excitatory')
        self.kc_a2en_a = sim.Projection(self.kcs_a , self.ens_a , sim.AllToAllConnector(), sim.StaticSynapse(weight = 0.01) ,
                               receptor_type = 'excitatory')
        if self.learned is False:
            stdp_model = sim.STDPMechanism(
                # timing_dependence=sim.SpikePairRule(tau_plus=1.25, tau_minus=0.1,
                #                                     A_plus=0.3, A_minus=0),

                timing_dependence=sim.SpikePairRule(tau_plus=1.25, tau_minus=0.1,
                                                    A_plus=0.3, A_minus=0.015),
                weight_dependence=sim.MultiplicativeWeightDependence(w_min=0, w_max=0.3),
                weight=0)
            self.kc2kc_a=sim.Projection(self.kcs, self.kcs_a, sim.FromListConnector(S_kc2kc),
                                        synapse_type=stdp_model,receptor_type='inhibitory')
        else:
            self.kc2kc_a = sim.Projection(self.kcs , self.kcs_a , sim.FromListConnector(w_kc2kc,column_names=["weight"]),
                                          receptor_type = 'inhibitory')
            print('len KC2KC', len(S_kc2kc),len(w_kc2kc), 'learned percentage:', len(w_kc2kc)/len(S_kc2kc) )
            # self.kc2kc_a = sim.Projection(self.kcs , self.kcs_a , sim.FromListConnector(S_kc2kc),
            #                               synapse_type=sim.StaticSynapse(weight = w_kc2kc, delay = 1.0) ,receptor_type = 'inhibitory')

        path,filename=os.path.split(self.filename)
        self.filename_label = str(datetime.now().strftime('%Y_%m_%d_%H_%M_%S'))+'_file='+filename+'_t='+\
                              str(self.t_start)+'~'+str(self.t_end) + '_learning='+str(learned)

        self.pns.record(["spikes"])
        self.kcs.record(["spikes"])
        self.kcs_a.record(["spikes"])
        self.kcs_a.sample(1).record(['v'])
        self.ens.record(['v'],to_file=(recordoutputpath+self.filename_label+"_en_v.pkl"))
        self.ens_a.record(['v'],to_file=(recordoutputpath+self.filename_label+"_en_v.pkl"))

        self.ens.record(['spikes'],to_file=(recordoutputpath+self.filename_label+"_en_spikes.pkl"))
        self.ens_a.record(['spikes'],to_file=(recordoutputpath+self.filename_label+"_en_a_spikes.pkl"))
        # self.ens.record(['spikes'])
        # self.ens_a.record(['spikes'])
        self.buildCPUTime = timer.diff()

    def run_sim(self):
        # Setup the Simulator
        sim.run(self.sim_t)

        simCPUTime = timer.diff()

        pn_spikes = self.pns.get_data(variables=["spikes"]).segments[0].spiketrains
        kc_spikes = self.kcs.get_data(variables=["spikes"]).segments[0].spiketrains
        kc_a_spikes = self.kcs_a.get_data(variables=["spikes"]).segments[0].spiketrains
        en_spikes = self.ens.get_data(variables=["spikes"]).segments[0].spiketrains
        en_a_spikes = self.ens_a.get_data(variables=["spikes"]).segments[0].spiketrains
        mbon_v = self.ens.get_data(variables=["v"]).segments[0].filter(name='v')[0]
        mbon_a_v = self.ens_a.get_data(variables=["v"]).segments[0].filter(name='v')[0]
        kc_a_v = self.kcs_a.get_data(variables=["v"]).segments[0].filter(name='v')[0]
        #if self.learned is False:
        self.learned_w_kc2kc = self.kc2kc_a.get(['weight'], format='array',with_address=False)
        #kc2kc_w = self.kc2kc_a.get('weight', format='list')

        writeCPUTime = timer.diff()

        if node_id == 0:
             print("\n--- Vogels-Abbott Network Simulation ---")
             # print("Nodes                  : %d" % np)
             # # print("Simulation type        : %s" % benchmark)
             # print("Number of Neurons      : %d" % nb_pn+nb_kc+nb_en)
             # print("Number of KC-KC Synapses     : %s" % self.kc2kc_a)
             # print("Excitatory conductance : %g nS" % Gexc)
             # print("Inhibitory conductance : %g nS" % Ginh)
             print("Build time             : %g s" % self.buildCPUTime)
             print("Simulation time        : %g s" % simCPUTime)
             print("Writing time           : %g s" % writeCPUTime)

        sim.end()



        plot.Figure(

         plot.Panel(pn_spikes, data_labels=[self.pns.label], xticks =True, yticks=True, markersize=1, xlim=(0,self.sim_t)),
         plot.Panel(kc_spikes, data_labels=[self.kcs.label],xticks =True, yticks=True, markersize=1, xlim=(0,self.sim_t)),
         plot.Panel(kc_a_spikes, data_labels=[self.kcs_a.label],xticks =True, yticks=True, markersize=1, xlim=(0,self.sim_t)),
         plot.Panel(kc_a_v, ylabel="Membrane potential (mV)",
                    data_labels=[self.kcs_a.label], xlabel="Time (ms)",  xticks =True, yticks=True, xlim=(0,self.sim_t)),

         plot.Panel(mbon_v, ylabel="Membrane potential (mV)",
                    data_labels=[self.ens.label], xticks =True, yticks=True, xlim=(0,self.sim_t)),


         # plot voltage for first ([0]) neuron
         plot.Panel(mbon_a_v, ylabel="Membrane potential (mV)",
                    data_labels=[self.ens_a.label],xticks =True, yticks=True, xlim=(0,self.sim_t)),

         # plot spikes (or in this case spike)


         #plot.Panel(en_spikes, xticks =True, yticks=True, markersize=1, xlim=(0,self.sim_t)),
         #plot.Panel(en_a_spikes, xticks =True, yticks=True, markersize=1, xlim=(0,self.sim_t)),

         ).save(figoutputpath+self.filename_label+".png")
        plt.clf()
        plt.close()


        en_spikes= neo.SpikeTrain(en_spikes[0], t_start=en_spikes[0][0], t_stop=en_spikes[0][-1], units='ms')

        en_a_spikes= neo.SpikeTrain(en_a_spikes[0], t_start=en_a_spikes[0][0], t_stop=en_a_spikes[0][-1], units='ms')

        en_inst_rate = instantaneous_rate(en_spikes, sampling_period=10*ms)
        en_a_inst_rate = instantaneous_rate(en_a_spikes, sampling_period=10*ms)

        plt.figure(figsize=(8, 4), dpi=300)

        plt.subplot(211)
        pn_mean = calculate_group_mean_rt(pn_spikes, nb_pn)
        kc_mean = calculate_group_mean_rt(kc_spikes, nb_kc)
        kc_a_mean = calculate_group_mean_rt(kc_a_spikes, nb_kc)


        # plt.hlines(kc_a_mean, linestyle='-', label='KC-A mean firing rate', linewidth = 2, colors='black',  alpha = 0.5, xmin = (kc_spikes[0][0]), xmax = (pn_spikes[0][-1]))
        plt.xlim(en_spikes.t_start, en_spikes.t_stop)
        plt.ylim(0,40)
        plt.legend( loc='upper right')
        plt.subplot(212)
        # instantaneous rate
        plt.plot(en_inst_rate.times.rescale(ms), en_inst_rate.rescale(1/s).magnitude.flatten(), label='MBON instantaneous rate')
        plt.plot(en_a_inst_rate.times.rescale(ms), en_a_inst_rate.rescale(1/s).magnitude.flatten(), label='MBON-A instantaneous rate')
        plt.hlines(0, linestyle='-', linewidth = 250, colors='black',  alpha = 0.2,  xmin = (self.sim_t* self.t_start), xmax = (self.sim_t* self.t_end))
        # axis labels and legend
        plt.xlabel('time (ms)')
        plt.ylabel('firing rate (Hz)')
        plt.xlim(en_spikes.t_start, en_spikes.t_stop)
        plt.ylim(0,250)
        plt.legend( loc='upper right')
        plt.savefig(figoutputpath+str(datetime.now().strftime('%Y_%m_%d_%H_%M_%S'))+self.filename_label+".svg", format='svg')
        plt.clf()
        plt.close()




