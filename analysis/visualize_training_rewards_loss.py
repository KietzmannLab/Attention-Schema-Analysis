import matplotlib.pyplot as plt

# numpy for handeling matrix operations
import pandas as pd

file_name = "results_agent_with_as_dqn_8_act_mem_fixed_0.001_lr_3_windowsize_0.9_discount_factor_200_units" + ".csv"
file_name2 = "results_agent_with_as_dqn_8_act_mem_random_0.001_lr_3_windowsize_0.9_discount_factor_200_units" + ".csv"
file_name3 = "results_agent_with_as_dqn_8_act_mem0.001_lr_3_windowsize_0.9_discount_factor_200_units" + ".csv"
file_name4 = "results_agent_with_as_no_ar_ppo_3_act_mem_fixed_1e-05_lr_1000_units_3_windowsize_0.99_discount_factor" + ".csv"
file_name5 = "results_agent_with_as_ppo_3_act_mem_random_1e-05_lr_3_windowsize_0.9_discount_factor_1000_units" + ".csv"
file_name6 = "results_agent_with_as_no_ar_ppo_3_act_no_mem_1e-05_lr_1000_units_3_windowsize_0.9_discount_factor" + ".csv"
file_name7 = "results_agent_with_as_ppo_3_act_mem1e-05_lr_3_windowsize_0.9_discount_factor_1000_units" + ".csv"
file_name8 = "results_agent_with_no_as_no_ar_ppo_3_act_mem_random_1e-05_lr_3_windowsize_0.99_discount_factor_1000_units" + ".csv"
file_name9 = "results_agent_with_as_no_ar_ppo_3_act_mem1e-05_lr_1000_units_3_windowsize_0.99_discount_factor" + ".csv"
file_name10 = "results_agent_with_as_dqn_8_act_mem_fixed_new_0.001_lr_3_windowsize_0.9_discount_factor_200_units" + ".csv"
file_name11 = "results_agent_with_as_dqn_8_act_mem_random__new0.001_lr_3_windowsize_0.9_discount_factor_200_units" + ".csv"
file_name12 = "results_agent_with_as_dqn_8_act_mem_new_0.001_lr_3_windowsize_0.9_discount_factor_200_units" + ".csv"
file_name26 = "results_agent_with_emergence_no_mem_ppo_3_act_1e-05_lr_1000_units_3_windowsize_0.9_discount_factor" + ".csv"
file_name13 = "results_agent_with_as_no_ar_ppo_3_act_no_mem_random_1e-05_lr_1000_units_3_windowsize_0.9_discount_factor" + ".csv"
file_name14 = "results_agent_with_as_ppo_3_act_no_mem_random_1e-05_lr_3_windowsize_0.9_discount_factor_1000_units" + ".csv"
file_name15 = "results_agent_with_as_dqn_8_act_mem_random_8_0.001_lr_3_windowsize_0.9_discount_factor_200_units" + ".csv"

file_name_no_as = "" + ".csv"


def visualize_func_as(file_name, extra_controls=False):

    results_agent_with_as = pd.read_csv(
        "./csvs/"+ file_name
    )

    
    fig, ax = plt.subplots(4, 1, figsize=(8, 20))
    plt.subplots_adjust(hspace=0.5)
    
    ax[0].plot(
        results_agent_with_as.index,
        results_agent_with_as["training_return"].values,
    )
    ax[0].set_ylabel('Overall Training Return')
    ax[0].set_xlabel('Epochs')
    ax[3].plot(
        results_agent_with_as.index, results_agent_with_as["loss"].values
    )
    ax[3].set_ylabel("Loss")
    ax[3].set_xlabel('Epochs')


    ax[1].plot(
        results_agent_with_as.index,
        results_agent_with_as["attention_score"].values,
        alpha=0.5,
        label = "Normal Agent"
    )
    """ax[1].plot(
        results_agent_with_as.index,
        results_agent_with_as["attention_score_disabled_a"].values,
        alpha=0.5,
        label = "No Noise Reduction"

        
    )"""
    """ax[1].plot(
        results_agent_with_as.index,
        results_agent_with_as["attention_score_disabled_as"].values,
        alpha=0.5,
        label = "No Attention Schema",
        color = 'green'

    )"""

    
    
    if extra_controls:

        ax[1].plot(
            results_agent_with_as.index,
            results_agent_with_as["attention_score_random_schema"].values,
            alpha=0.5,
            label = "Random Schema Actions",
            color = 'red'
        )    
        ax[1].plot(
            results_agent_with_as.index,
            results_agent_with_as["attention_score_fixed_schema"].values,
            alpha=0.5,
            label = "Fixed Schema Actions",
            color = 'purple'

        
        )

    elif "emergence" in file_name or "new" in file_name:
        ax[1].plot(
            results_agent_with_as.index,
            results_agent_with_as["attention_score_random_schema"].values,
            alpha=0.5,
            label = "Random Schema Actions",
            color = 'red'

        )

    """ax[1].set_title(
        "In blue, the return of the trained agent is shown (maximum would be 4). \n In orange, the attention window doesnt remove noise"
        + "\n In green, the Attention Schema was disabled."
        + "\n In red, random actions of Attention Schema."
    )"""
    ax[1].set_ylabel('Attention Tracking Return')
    ax[1].set_xlabel('Epochs')
    ax[1].legend(loc="best")



    ax[2].set_alpha(0.5)

    ax[2].plot(
        results_agent_with_as.index,
        results_agent_with_as["catch_score"].values,
        alpha=0.5,
        label = "Normal Agent"
    )
    """ax[2].plot(
        results_agent_with_as.index,
        results_agent_with_as["catch_score_disabled_a"].values,
        alpha=0.5,
        label = "No Noise Reduction"
    )"""
    """ax[2].plot(
        results_agent_with_as.index,
        results_agent_with_as["catch_score_disabled_as"].values,
        alpha=0.5,
        label = "No Attention Schema",
        color = 'green'

    )"""
    

    if extra_controls:

        ax[2].plot(
            results_agent_with_as.index,
            results_agent_with_as["catch_score_random_schema"].values,
            alpha=0.5,
            label = "Random Schema Actions",
            color = 'red'

        )

        ax[2].plot(
            results_agent_with_as.index,
            results_agent_with_as["catch_score_fixed_schema"].values,
            alpha=0.5,
            label = "Fixed Schema Actions",
            color = 'purple'

        )
        

    elif "emergence" in file_name or "new" in file_name:
        ax[2].plot(
            results_agent_with_as.index,
            results_agent_with_as["catch_score_random_schema"].values,
            alpha=0.5,
            label = "Random Schema Actions",
            color = 'red'

        )
    
    
    """ax[2].set_title(
        "In blue, the return of the trained agent is shown (maximum would be 4). \n In orange, the attention window doesnt remove noise"
        + "\n In green, the Attention Schema was disabled."
        + "\n In red, random actions of Attention Schema."

    )"""

    ax[2].set_ylabel('Catching Ball Return')
    ax[2].set_xlabel('Epochs')
    ax[2].legend(loc="best")


    
    fig.savefig("./plots/"+file_name[:-4]+".png")




def visualize_func_no_as(file_name_no_as, extra_controls=False):
    results_agent_without_as = pd.read_csv(
        "./csvs/"+ file_name_no_as
    )

    fig, ax = plt.subplots(4, 1, figsize=(8, 20))
    plt.subplots_adjust(hspace=0.5)
    ax[0].plot(
        results_agent_without_as.index,
        results_agent_without_as["training_return"].values,
        
    )
    ax[0].set_ylabel("Overall return")
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
        "Return of attention tracking (maximum would be 4)."
        + " \n In orange, the attention schema was enabled"
    )"""

    ax[1].set_ylabel('Attention Score Return')
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
        "Return of catching ball (maximum would be 2). "
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
        

    ax[2].set_ylabel('Catching Ball Return')
    ax[2].set_xlabel('Epochs')
    ax[2].legend(loc="best")

    fig.savefig("./plots/"+file_name_no_as[:-4]+".png")


visualize_func_as(file_name, extra_controls=True)

visualize_func_as(file_name2, extra_controls=True)
visualize_func_as(file_name3, extra_controls=True)
visualize_func_as(file_name4, extra_controls=True)
visualize_func_as(file_name5, extra_controls=True)
visualize_func_as(file_name6, extra_controls=True)
visualize_func_as(file_name7, extra_controls=True)
visualize_func_no_as(file_name8, extra_controls=True)
visualize_func_as(file_name9, extra_controls=True)
visualize_func_as(file_name10, extra_controls=True)
visualize_func_as(file_name11, extra_controls=True)
visualize_func_as(file_name12, extra_controls=True)
visualize_func_as(file_name13, extra_controls=True)
visualize_func_as(file_name14, extra_controls=True)
visualize_func_as(file_name15, extra_controls=True)

visualize_func_as(file_name26)


#visualize_func_no_as(file_name_no_as)
