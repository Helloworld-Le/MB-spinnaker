## this code test one-shot learing by:
##    learning one segment on a long route, then testing all the other runs.
##    also change the location and length of learning segment, testing all the other runs

from MB_main_11_ad_offset import *
import os
np.random.seed(2022)
data_file_path = '/media/le/121942f7-1af1-4374-bc84-6f60a0cc0259/videos/whole_route_all_together_events_840/selected_longroute_LMH/'
#data_file_path = '/media/le/121942f7-1af1-4374-bc84-6f60a0cc0259/videos/high_route_odom/high_routes_events/'
files_list = os.listdir(data_file_path)
files_list.sort()

print(files_list)

###change learning location:
for i in [0.0, 0.333]:
    ###change learning length:
    for l in [0.2, 0.333, 0.5]:
        t_start = i
        t_end = l+t_start
        for files in files_list:
            sim.setup(timestep=1.0, min_delay=1.0, max_delay=14)
            sim.set_number_of_neurons_per_core(sim.IF_curr_exp,50)
            learning_filename = data_file_path+files
            mb_learning = MB_LE(filename=learning_filename,t_start=t_start,t_end=t_end)
            mb_learning.run_sim()
            print('learned connections:', type(mb_learning.learned_w_kc2kc),len(mb_learning.learned_w_kc2kc))
            weights = np.array(mb_learning.learned_w_kc2kc)
            #scatter_hist(weights)
            weights = generate_pnlist(weights)
            print ('len CK2KC:', len(weights) )   

            testing_filename = learning_filename
            sim.setup(timestep=1.0, min_delay=1.0, max_delay=14)
            sim.set_number_of_neurons_per_core(sim.IF_curr_exp,100)
            mb_testing = MB_LE(filename=testing_filename,learned=True,w_kc2kc=weights)
            mb_testing.run_sim()

