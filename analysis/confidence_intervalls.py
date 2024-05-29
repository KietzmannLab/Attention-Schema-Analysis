
import copy
import math
import os
from pathlib import Path
import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

noises = [0,0.25,0.5,0.75,1]
reward_type = ["ar", "cr"]
schema_type = ["random_8_acts", "not_random"]

def draw_bs_replicates(data,func,size):
    """creates a bootstrap sample, computes replicates and returns replicates array"""
    # Create an empty array to store replicates
    bs_replicates = np.empty(size)
    
    # Create bootstrap replicates as much as size
    for i in range(size):
        # Create a bootstrap sample
        bs_sample = np.random.choice(data,size=len(data))
        # Get bootstrap replicate and append to bs_replicates
        bs_replicates[i] = func(bs_sample)
    
    return bs_replicates

for noise in noises:
    for schema in schema_type:
        for r,reward in enumerate(reward_type):
            if schema == "not_random":
                path = Path("./csvs/test_8_random_acts/not_random_noise_"+str(noise)+".csv")
            else:
                path = Path("./csvs/test_8_random_acts/3_actions_noise_"+str(noise)+".csv")

            data = np.loadtxt(path, dtype=int, delimiter=",")[1:]
            data = data[:,r]

            bs_replicates_heights = draw_bs_replicates(data,np.mean,10000)
            ci_low = np.percentile(bs_replicates_heights,[2.5])
            ci_high = np.percentile(bs_replicates_heights,[97.5])

            print(noise, schema, reward, ci_low, ci_high, ci_high-ci_low  )

