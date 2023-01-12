
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib
np.random.seed(2022)
from MB_main_11_ad_offset import *
import os

data_file_path = '/media/le/121942f7-1af1-4374-bc84-6f60a0cc0259/videos/data_mat/231023newdataset/840/'
#data_file_path = '/media/le/121942f7-1af1-4374-bc84-6f60a0cc0259/videos/data_mat/1015originaltimestamp/'
#data_file_path = '/media/le/121942f7-1af1-4374-bc84-6f60a0cc0259/videos/1215_outdoor_displacement_data/normal_angle/mid/dvs_eventsframes/frames_parallel/'
files_list = os.listdir(data_file_path)
files_list.sort(reverse= False)
#learningfiles = os.listdir('/media/le/121942f7-1af1-4374-bc84-6f60a0cc0259/videos/data_mat/BW_MSc (3rd copy)/angular/840/wide_angle_learning_file/')
learningfiles = files_list
print(files_list)

###change learning location:
for i in [0]:
    ###change learning length:
    for l in [0.5]:
        t_start = i
        t_end = l+t_start
        for files in learningfiles:
            sim.setup(timestep=1.0, min_delay=1.0, max_delay=14)
            sim.set_number_of_neurons_per_core(sim.IF_curr_exp,50)
            learning_filename = data_file_path+files
            mb_learning = MB_LE(filename=learning_filename,t_start=t_start,t_end=t_end)
            mb_learning.run_sim()
            print('learned connections:', mb_learning.learned_w_kc2kc[1], len(mb_learning.learned_w_kc2kc))
            #weights = mb_learning.kc2kc_a.get(["weight"], format="array")
            weights = np.array(mb_learning.learned_w_kc2kc)
            print(np.nansum(weights))
            #scatter_hist(weights)
            weights = generate_pnlist(weights)
            print ('len CK2KC:', len(weights) )   
            for f in files_list:
                print(f)
                testing_filename = data_file_path + f
                sim.setup(timestep=1.0, min_delay=1.0, max_delay=14)
                sim.set_number_of_neurons_per_core(sim.IF_curr_exp,100)
                mb_testing = MB_LE(filename=testing_filename,learned=True,w_kc2kc=weights)
                mb_testing.run_sim()



# mb =MB_LE(filename='dvsdata.mat',t_start=0.5,t_end=0.6)
# mb.run_sim()
# print('kc2kc:', len(mb.learned_w_kc2kc), sum(mb.learned_w_kc2kc[15:20]))
# #for i in mb.learned_w_kc2kc:
#     #print(len(i))
#
# sim.setup(timestep=1.0, min_delay=1.0, max_delay=14)
# sim.set_number_of_neurons_per_core(sim.IF_curr_exp,50)
#
# mb2 = MB_LE(filename='dvsdata.mat',learned=True,w_kc2kc=mb.learned_w_kc2kc)
# mb2.run_sim()
# print('kc2kc:', len(mb2.learned_w_kc2kc), sum(mb2.learned_w_kc2kc[15:20]))
