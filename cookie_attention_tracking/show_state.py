import copy
import math
import os

import matplotlib.pyplot as plt

# numpy for handeling matrix operations
import numpy as np
import PIL.Image
import random
# Keras is a deep learning libarary
import tensorflow as tf
from tf_agents.environments import py_environment, tf_py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts

from environment import Train

noise = 0.5
look_back = 10
discount_factor=0.99
window_size=9
attention_reward = True
attention_schema = True
catch_reward = True
noise_removing = True

as_env = Train(
	noise = noise,
	look_back = look_back,
	discount_factor=discount_factor,
	window_size=window_size,
	attention_reward = attention_reward,
	attention_schema = attention_schema,
	catch_reward = catch_reward,
	noise_removing = noise_removing,
	)

def visualize(env):
    canvas = env._draw_state()
    print(canvas)
    canvas = np.copy(np.asarray(canvas))
    
    
    upscaled_canvas = PIL.Image.fromarray(canvas).resize(
        [100, 200], resample=PIL.Image.NEAREST
    )
    upscaled_canvas = np.asarray(upscaled_canvas)
    plt.imsave(
        "./environments/as/"
        + str(window_size)
        + "_window_size"
        + ".png",
        upscaled_canvas,
    )

    

#visualize(as_env)

def visualize_10_steps(env):
    for i in range(1):
        time_step = env.reset()

        for step in range(9):
            action = random.randint(0,15)
            time_step = env.step(action)
            state = time_step.observation
    print(state)
    canvas = np.split(np.asarray(state), 10, axis=0)
    print(canvas)
    
    # canvas = np.reshape(np.asarray(canvas), newshape=(200,10))
    for step in range(10):
        canvas_arr = np.asarray(canvas[step])
        upscaled_canvas = PIL.Image.fromarray(canvas_arr).resize(
            [100, 200], resample=PIL.Image.NEAREST
        )
        upscaled_canvas = np.asarray(upscaled_canvas)
        plt.imsave(
            "./environments/as/"
        + str(window_size)
        + "_window_size/"
        + str(step)
        + ".png",
        upscaled_canvas,
        )

visualize_10_steps(as_env)