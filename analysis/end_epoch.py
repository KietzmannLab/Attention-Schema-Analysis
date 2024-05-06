import matplotlib.pyplot as plt

import pandas as pd
from scipy import interpolate
from scipy.signal import savgol_filter

import numpy as np

file_name6 = "results_agent_with_emergence_ar_no_mem_ppo_3_act_1e-05_lr_1000_units_3_windowsize_0.9_discount_factor" + ".csv"
file_name1 = "results_agent_with_noise_0_random_emergence_ar_no_mem_ppo_3_act_1e-05_lr_1000_units_3_windowsize_0.9_discount_factor" + ".csv"
file_name2 = "results_agent_with_noise_0.25_random_emergence_ar_no_mem_ppo_3_act_1e-05_lr_1000_units_3_windowsize_0.9_discount_factor" + ".csv"
file_name3 = "results_agent_with_noise_0.75_random_emergence_ar_no_mem_ppo_3_act_1e-05_lr_1000_units_3_windowsize_0.9_discount_factor" + ".csv"
file_name4 = "results_agent_with_noise_1_random_emergence_ar_no_mem_ppo_3_act_1e-05_lr_1000_units_3_windowsize_0.9_discount_factor" + ".csv"
epochs_considered = 4

def find_stop_point(file_name):
    changes = []
    smaller_0_1 = []
    smaller_0_00005 = []
    smaller_0_01 = []
    more_than_5_9 = []
    file = pd.read_csv(
            "./csvs/"+ file_name
        )
    for idx in range(file[:-epochs_considered*2].shape[0]):
        first_file_part = file[idx:idx+epochs_considered]
        first_file_part = first_file_part["training_return"]
        second_file_part = file[idx+epochs_considered:idx+epochs_considered*2]
        second_file_part = second_file_part["training_return"]
        first_mean = np.mean(first_file_part)
        second_mean = np.mean(second_file_part)
        ratio = second_mean/first_mean
        change = np.abs((1 - ratio))
        changes.append(ratio)
        #smaller_0_01.append((change < 0.1))   
        #smaller_0_1.append((change < 0.01))
        #smaller_0_00005.append((change < 0.00000001))
        more_than_5_9.append((first_mean>5.95))


    #print(np.where(np.array(smaller_0_00005) > 0)[0][0])
    #print(np.where(np.array(smaller_0_00005) > 0)[0])
    print(np.where(np.array(more_than_5_9) > 0)[0][0])

find_stop_point(file_name1)
find_stop_point(file_name2)
find_stop_point(file_name6)
find_stop_point(file_name3)
find_stop_point(file_name4)