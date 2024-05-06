import matplotlib.pyplot as plt

import pandas as pd
from scipy import interpolate
from scipy.signal import savgol_filter

import numpy as np
file_name = "results_agent_with_emergence_no_mem_ppo_3_act_1e-05_lr_1000_units_3_windowsize_0.9_discount_factor" + ".csv"
new_name= "emergence_no_ar_2_vertical"
# file_name = "results_agent_with_as_no_mem_new_ppo_3_act_1e-05_lr_3_windowsize_0.9_discount_factor_500_units" + ".csv"
# new_name= "ppo_vertical"

file_name2 = "results_agent_with_noise_0_random_emergence_ar_no_mem_ppo_3_act_1e-05_lr_1000_units_3_windowsize_0.9_discount_factor" + ".csv"
new_name2= "0_noise_emergence_no_mem_vertical"
file_name3 = "results_agent_with_noise_0.5_random_emergence_ar_no_mem_ppo_3_act_1e-05_lr_1500_units_3_windowsize_0.9_discount_factor" + ".csv"
new_name3= "0-5_noise_1500_units_emergence_no_mem_vertical"
file_name4 = "results_agent_with_noise_0.5_random_emergence_ar_no_mem_ppo_3_act_1e-05_lr_2000_units_3_windowsize_0.9_discount_factor" + ".csv"
new_name4= "0-5_noise_2000_units_emergence_no_mem_vertical"
file_name5 = "results_agent_with_noise_1_random_emergence_ar_no_mem_ppo_3_act_1e-05_lr_1000_units_3_windowsize_0.9_discount_factor" + ".csv"
new_name5= "1_noise_emergence_no_mem_vertical"
file_name6 = "results_agent_with_noise_0.5_no_as_ar_no_mem_ppo_1000_units_3_windowsize_0.9_disc_factor" + ".csv"
new_name6 = "0-5_noise_no_as"
file_name7 = "results_agent_with_noise_0.25_random_emergence_ar_no_mem_ppo_3_act_1e-05_lr_1000_units_3_windowsize_0.9_discount_factor" + ".csv"
new_name7= "0-25_noise_emergence_no_mem_vertical"
file_name8 = "results_agent_with_noise_0.75_random_emergence_ar_no_mem_ppo_3_act_1e-05_lr_1000_units_3_windowsize_0.9_discount_factor" + ".csv"
new_name8= "0-75_noise_emergence_no_mem_vertical"

alpha = 0.2
alpha2 = 1

file_name_no_as = "" + ".csv"

pol=4

def visualize_func_as(file_name, new_name, extra_controls=False):

    results_agent_with_as = pd.read_csv(
        "./csvs/"+ file_name
    )

    results_agent_with_as = results_agent_with_as

    
    fig, ax = plt.subplots(2, 1, figsize=(7, 10))

    #plt.subplots_adjust(hspace=0.5)
    
    fontsize=12
    font = {'size': fontsize}
 
    # using rc function
    plt.rc('font', **font)
    
    x = results_agent_with_as.index
    
    ax[0].plot(
        x,
        results_agent_with_as["attention_score"].values,
        label = "Normal Agent",

        alpha=alpha
    )
    y_new= savgol_filter(results_agent_with_as["attention_score"].values,results_agent_with_as.shape[0],pol)
    ax[0].plot(
        x,
        y_new,
        color="b",
        #label = "Normal Agent",
        alpha=alpha2

    )
    
    """ ax[0].plot(
        x,
        results_agent_with_as["attention_score_disabled_a"].values,
        alpha=alpha,
        color="brown"
        
    )
    y_new= savgol_filter(results_agent_with_as["attention_score_disabled_a"].values,results_agent_with_as.shape[0],pol)
    ax[0].plot(
        x,
        y_new,
        color="brown",
        label = "No Noise Reduction",
        alpha=alpha2


    )"""


    """ax[0].plot(
        results_agent_with_as.index,
        results_agent_with_as["attention_score_disabled_as"].values,
        alpha=alpha,
        
        color = 'green'

    )
    y_new= savgol_filter(results_agent_with_as["attention_score_disabled_as"].values,results_agent_with_as.shape[0],pol)
    ax[0].plot(
        x,
        y_new,
        color="g",
        label = "No Attention Schema",
        alpha=alpha2

    )"""

    
    
    if extra_controls:

        ax[0].plot(
            x,
            results_agent_with_as["attention_score_random_schema"].values,
            alpha=alpha,
            color = 'red'
        )    
        y_new= savgol_filter(results_agent_with_as["attention_score_random_schema"].values,results_agent_with_as.shape[0],pol)
        ax[0].plot(
            x,
            y_new,
            color="red",
            label = "Random Schema Actions",
            alpha=alpha2

        )
        
        ax[0].plot(
            x,
            results_agent_with_as["attention_score_fixed_schema"].values,
            alpha=alpha,
            color = 'purple'

        
        )
        y_new= savgol_filter(results_agent_with_as["attention_score_fixed_schema"].values,results_agent_with_as.shape[0],pol)
        ax[0].plot(
            x,
            y_new,
            label = "Fixed Schema Actions",
            color="purple",
            alpha=alpha2

        )
        
    elif "emergence" in file_name or "new" in file_name:
        ax[0].plot(
            x,
            results_agent_with_as["attention_score_random_schema"].values,
            alpha=alpha,
            label = "Random Schema Actions",

            color = 'red'

        )
        y_new= savgol_filter(results_agent_with_as["attention_score_random_schema"].values,results_agent_with_as.shape[0],pol)
        ax[0].plot(
            x,
            y_new,
            color="red",
            #label = "Random Schema Actions",
            alpha=alpha2

        )

        

    
    ax[0].set_ylabel('Attention Tracking Reward', fontsize=fontsize)
    ax[0].set_xlabel('Epochs', fontsize=fontsize)
    ax[0].legend(loc=8)

    
    ax[1].plot(
        x,
        results_agent_with_as["catch_score"].values,
        label = "Normal Agent",

        alpha=alpha,
    )
    y_new= savgol_filter(results_agent_with_as["catch_score"].values,results_agent_with_as.shape[0],pol)
    ax[1].plot(
        x,
        y_new,
        color="b",
        #label = "Normal Agent",
        alpha = alpha2
    
    )
    
    """ax[1].plot(
        results_agent_with_as.index,
        results_agent_with_as["catch_score_disabled_a"].values,
        alpha=alpha,
        color="brown"
    )
    y_new= savgol_filter(results_agent_with_as["catch_score_disabled_a"].values,results_agent_with_as.shape[0],pol)
    ax[1].plot(
        x,
        y_new,
        color="brown",
        label = "No Noise Reduction",
        alpha=alpha2,

    )"""
    
    """ax[1].plot(
        x,
        results_agent_with_as["catch_score_disabled_as"].values,
        alpha=alpha,
        color = 'green'

    )
    y_new= savgol_filter(results_agent_with_as["catch_score_disabled_as"].values,results_agent_with_as.shape[0],pol)
    ax[1].plot(
        x,
        y_new,
        color="g",
        label = "No Attention Schema",
        alpha=alpha2
    )"""
    
    
    if extra_controls:

        ax[1].plot(
            x,
            results_agent_with_as["catch_score_random_schema"].values,
            alpha=alpha,
            color = 'red'

        )
        y_new= savgol_filter(results_agent_with_as["catch_score_random_schema"].values,results_agent_with_as.shape[0],pol)
        ax[1].plot(
            x,
            y_new,
            color="red",
            label = "Random Schema Actions",
            alpha=alpha2

        )
        
        ax[1].plot(
            x,
            results_agent_with_as["catch_score_fixed_schema"].values,
            alpha=alpha,
            color = 'purple'

        )
        y_new= savgol_filter(results_agent_with_as["catch_score_fixed_schema"].values,results_agent_with_as.shape[0],pol)
        ax[1].plot(
            x,
            y_new,
            color="purple",
            label = "Fixed Schema Actions",
            alpha=alpha2

        )
        

    elif "emergence" in file_name or "new" in file_name:
        ax[1].plot(
            x,
            results_agent_with_as["catch_score_random_schema"].values,
            alpha=alpha,
            label = "Random Schema Actions",

            color = 'red'

        )
        y_new= savgol_filter(results_agent_with_as["catch_score_random_schema"].values,results_agent_with_as.shape[0],pol)
        ax[1].plot(
            x,
            y_new,
            color="red",
            #label = "Random Schema Actions",
            alpha=alpha2
        )
    
    
    

    ax[1].set_ylabel('Catching Ball Reward', fontsize=fontsize)
    ax[1].set_xlabel('Epochs', fontsize=fontsize)
    ax[1].legend(loc=8)


    
    fig.savefig("./plots/"+new_name)




def visualize_func_no_as(file_name_no_as, new_name, extra_controls=False):
    results_agent_without_as = pd.read_csv(
        "./csvs/"+ file_name_no_as
    )

    fig, ax = plt.subplots(4, 1, figsize=(8, 20))
    plt.subplots_adjust(hspace=0.5)
    ax[0].plot(
        results_agent_without_as.index,
        results_agent_without_as["training_return"].values,
        
    )
    ax[0].set_ylabel("Overall Reward")
    ax[0].set_xlabel("Epochs")
    ax[3].plot(
        results_agent_without_as.index, results_agent_without_as["loss"].values
    )
    ax[3].set_ylabel("Loss")
    ax[3].set_xlabel("Epochs")

    ax[1].plot(
        results_agent_without_as.index,
        results_agent_without_as["attention_score"].values,
        alpha=0.5,
        label = "Normal Agent"
    )
    ax[1].plot(
        results_agent_without_as.index,
        results_agent_without_as["attention_score_enabled_as"].values,
        alpha=0.5,
        label = "No Attention Schema"
    )

    if extra_controls:
        ax[1].plot(
            results_agent_without_as.index,
            results_agent_without_as["attention_score_random_schema"].values,
            alpha=0.5,
            label = "Random Schema Actions"
        )
        
        ax[1].plot(
            results_agent_without_as.index,
            results_agent_without_as["attention_score_fixed_schema"].values,
            alpha=0.5,
            label = "Fixed Schema Actions"
        )
        
    """ax[2].set_title(
        "Reward of attention tracking (maximum would be 4)."
        + " \n In orange, the attention schema was enabled"
    )"""

    ax[1].set_ylabel('Attention Score Reward')
    ax[1].set_xlabel('Epochs')
    ax[1].legend(loc="best")

    ax[2].plot(
        results_agent_without_as.index,
        results_agent_without_as["catch_score"].values,
        alpha=0.5,
        label = "Normal Agent"
    )
    ax[2].plot(
        results_agent_without_as.index,
        results_agent_without_as["catch_score_enabled_as"].values,
        alpha=0.5,
        label = "No Attention Schema"
    )
    """ax[3].set_title(
        "Reward of catching ball (maximum would be 2). "
        #+ " \n In orange, the attention schema was enabled"
    )"""

    if extra_controls:
        ax[2].plot(
            results_agent_without_as.index,
            results_agent_without_as["catch_score_random_schema"].values,
            alpha=0.5,
            label = "Random Schema Actions"
        )
        
        ax[2].plot(
            results_agent_without_as.index,
            results_agent_without_as["catch_score_fixed_schema"].values,
            alpha=0.5,
            label = "Fixed Schema Actions"
        )
        

    ax[2].set_ylabel('Catching Ball Reward')
    ax[2].set_xlabel('Epochs')
    ax[2].legend(loc="best")

    # fig.savefig("./plots/"+file_name_no_as[:-4]+".png")
    fig.savefig("./plots/"+new_name+".png")


visualize_func_as(file_name, new_name=new_name, extra_controls=False)
pol=6
alpha =  0.7
alpha2 = 0
visualize_func_as(file_name2, new_name=new_name2, extra_controls=False)
visualize_func_as(file_name3, new_name=new_name3, extra_controls=False)
visualize_func_as(file_name4, new_name=new_name4, extra_controls=False)
visualize_func_as(file_name5, new_name=new_name5, extra_controls=False)
visualize_func_no_as(file_name6, new_name=new_name6, extra_controls=False)
visualize_func_as(file_name7, new_name=new_name7, extra_controls=False)
visualize_func_as(file_name8, new_name=new_name8, extra_controls=False)
