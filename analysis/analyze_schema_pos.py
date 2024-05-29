import copy
import math
import os
from pathlib import Path
import pandas as pd

import matplotlib.pyplot as plt

import numpy as np
import PIL.Image

# Keras is a deep learning libarary
import tensorflow as tf
from tf_agents.environments import py_environment, tf_py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts


policy1 = tf.saved_model.load("../policy/policy_with_emergence_ar_no_mem_ppo_3_act_1e-05_lr_1000_units_3_windowsize_0.9_discount_factor")
save_file_name = "fully_squeezed_aw_as_as_emergence_ar_no_mem_ppo_3_act_1e-05_lr_1000_units_3_windowsize_0.9_discount_factor"
file_name = "emergence_ar_no_mem_ppo_3_act_1e-05_lr_1000_units_3_windowsize_0.9_discount_factor"

#specify the training parameters for the environment if not default
attention_reward = True
catch_reward = True
noise_removing = True
attention_schema = True
window_size = 3
discount_factor = 0.99
look_back = 1
noise = 0.5
random_schema_action = False
decoupling = True
actions = 3

# this is the environment class that the agent has to learn to play in
# the only change to the environment is that it saves the attention positions
# and the schema positions

class Env(py_environment.PyEnvironment):
    """
    The environment in which the agent has to learn to play the game.
    The environment is a simple game where the agent has to move a paddle to catch a ball.
    The visual field is noisy and the agent has to attend to the ball to catch it.
    """
    def __init__(
        self,
        attention_reward=True,
        catch_reward=True,
        noise_removing=True,
        attention_schema=True,
        window_size:int=3,
        discount_factor=0.99,
        look_back=10,
        noise=0.5,
        random_schema_action = False,
        actions = 3,
        decoupling = True
    ):
        if attention_reward:
            self.attention_reward = tf.constant(0.5, dtype=tf.float32)
        else:
            self.attention_reward = tf.constant(0, dtype=tf.int32)

        if catch_reward:
            self.catch_reward = tf.constant(2, dtype=tf.int32)
        else:
            self.catch_reward = tf.constant(0, dtype=tf.int32)

        self.noise_removing = noise_removing
        self.attention_schema = attention_schema
        self.window_size = window_size
        self.half_window = math.floor(self.window_size / 2)
        self.look_back = look_back
        self.noise = noise
        self.discount_factor = discount_factor
        self.random_schema_action = random_schema_action
        self.actions = actions
        self.decoupling = decoupling

        self.grid_size = 10
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(),
            dtype=np.int32,
            minimum=0,
            maximum=(self.actions * 2 * self.actions) - 1, #(8 * 2 * 8) - 1,
            name="action",
        )  
        
        if decoupling == False:
            self._action_spec = array_spec.BoundedArraySpec(
                shape=(),
                dtype=np.int32,
                minimum=0,
                maximum=(self.actions * 2) - 1,
                name="action",
            )

        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(self.grid_size * 2 * self.look_back, self.grid_size),
            dtype=np.int32,
            minimum=0,
            maximum=20,
            name="observation",
        )  # The observation space is equal to two stacked game grids

        self.ball_row = 0  # Start the ball at the top of the screen
        self.ball_col = int(
            np.random.randint(0, self.grid_size, size=1)
        )  # What column does the ball appear in
        self.paddle_loc = math.floor(
            self.grid_size / 2 - 1
        )  # Starts paddle in the middle, column of leftmost corner of paddle
        self.attn_row = (
            copy.deepcopy(self.ball_row) + self.half_window #was 1
        )  # Attention starts fixated on the ball
        self.attn_col = copy.deepcopy(self.ball_col)
        
        if (
            self.attn_col < self.half_window  
        ):  # Check to make sure attention field is within bounds
            self.attn_col = self.half_window  
        if self.attn_col > self.grid_size - 1 - self.half_window:  
            self.attn_col = self.grid_size - 1 - self.half_window  
        self.landing = np.random.randint(
            0, 10
        )  # Randomly predetermine where the ball will land

        self.attn_rowspan = list(range(self.attn_row - self.half_window, self.attn_row + 1 + self.half_window)) #was-1 and +2
        self.attn_colspan = list(range(self.attn_col - self.half_window, self.attn_col + 1 + self.half_window)) #was-1 and +2

        self.schema_row = self.attn_row
        self.schema_col = self.attn_col
        self.schema_rowspan = self.attn_rowspan
        self.schema_colspan = self.attn_colspan

        self.memory_buffer = np.zeros(
            (self.grid_size * 2 * self.look_back, self.grid_size),
            dtype=np.int32,
        )
        self.step_count = 0  # internal counter for the current step

        self._state = self._append_to_memory(self._draw_state())

        self._episode_ended = False

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        self._episode_ended = False
        self.ball_row = 0  # Start the ball at the top of the screen
        self.ball_col = int(
            np.random.randint(0, self.grid_size, size=1)
        )  # What column does the ball appear in
        self.paddle_loc = math.floor(
            self.grid_size / 2 - 1
        )  # Starts paddle in the middle, column of leftmost corner of paddle
        self.attn_row = (
            copy.deepcopy(self.ball_row) + self.half_window #was 1
        )  # Attention starts fixated on the ball
        self.attn_col = copy.deepcopy(self.ball_col)
        
        if (
            self.attn_col < self.half_window  
        ):  # Check to make sure attention field is within bounds
            self.attn_col = self.half_window  
        if self.attn_col > self.grid_size - 1 - self.half_window:  
            self.attn_col = self.grid_size - 1 - self.half_window  
        self.landing = np.random.randint(
            0, 10
        )  # Randomly predetermine where the ball will land

        self.attn_rowspan = list(range(self.attn_row - self.half_window, self.attn_row + 1 + self.half_window)) #was-1 and +2
        self.attn_colspan = list(range(self.attn_col - self.half_window, self.attn_col + 1 + self.half_window)) #was-1 and +2

        self.schema_row = self.attn_row
        self.schema_col = self.attn_col
        self.schema_rowspan = self.attn_rowspan
        self.schema_colspan = self.attn_colspan

        self.memory_buffer = np.zeros(
            (self.grid_size * 2 * self.look_back, self.grid_size),
            dtype=np.int32,
        )
        self._state = self._append_to_memory(self._draw_state())
        self.step_count = 0

        return ts.restart(self._state)

    def _step(self, action):
        if self._episode_ended:
            return self.reset()

        self.step_count += 1  # Increment the step counter

        # here we define how action selection affects the location of the paddle

        if self.actions == 8 and self.decoupling:
            #8 actions
            if action in np.arange(0, 64):  # left
                move = -1
            elif action in np.arange(64, 128):
                move = 1  # right

            if action > 63:
                action = action - 64
            # Here we define how action selection affects the locus of attention
            # Rescale action selection to exclude the chosen move
            
            attn = action // 8
            schema = action % 8 

            if self.random_schema_action:
                schema = np.random.randint(0,8)
            # Attention movement options are stationary or 8 possible directions
            moves = np.array(
                [
                    (0, 1),
                    (1, 1),
                    (1, 0),
                    (1, -1),
                    (0, -1),
                    (-1, -1),
                    (-1, 0),
                    (-1, 1),
                ]
            )

        elif self.actions == 3 and self.decoupling:

            if action in np.arange(0, 9):  # left
                move = -1
            elif action in np.arange(9, 18):
                move = 1  # right

            if action > 8:
                action = action - 9

            attn = action // 3
            schema = action % 3 

            if self.random_schema_action:
                schema = np.random.randint(0,3)
            # Attention movement options are stationary or 8 possible directions
            moves = np.array(
                [
                    (0, 1),
                    (1, 1),
                    (-1, 1),
                ]
            )

        elif self.actions == 8 and self.decoupling == False:
            if action in np.arange(0, 8):  # left
                move = -1
            elif action in np.arange(8, 16):
                move = 1  # right

            if action > 7:
                action = action - 8
            

            if self.random_schema_action:
                schema = np.random.randint(0,8)
            # Attention movement options are stationary or 8 possible directions
            moves = np.array(
                [
                    (0, 1),
                    (1, 1),
                    (1, 0),
                    (1, -1),
                    (0, -1),
                    (-1, -1),
                    (-1, 0),
                    (-1, 1),
                ]
            )

            attn = action
            schema = action
        
        elif self.actions == 3 and self.decoupling == False:
            if action in np.arange(0, 3):
                move = -1
            elif action in np.arange(3, 6):
                move = 1  # right

            if action > 2:
                action = action - 3
            
            if self.random_schema_action:
                schema = np.random.randint(0,3)

            moves = np.array(
                [
                    (0, 1),
                    (1, 1),
                    (-1, 1),
                ]
            )

            attn = action
            schema = action


        attn_delta_col, attn_delta_row = moves[attn]
        # Apply the change in attention locus
        self.attn_row = self.attn_row + attn_delta_row
        self.attn_col = self.attn_col + attn_delta_col

        schema_delta_col, schema_delta_row = moves[schema]
        # Apply the change in attention locus
        self.schema_row = self.schema_row + schema_delta_row
        self.schema_col = self.schema_col + schema_delta_col



        if (
            self.attn_row < self.half_window  
        ):  # Check to make sure attention field is within bounds
            self.attn_row = self.half_window  
        if self.attn_row > self.grid_size - 1 - self.half_window:  
            self.attn_row = self.grid_size - 1 - self.half_window  
        if (
            self.attn_col < self.half_window  
        ):  # Check to make sure attention field is within bounds
            self.attn_col = self.half_window  
        if self.attn_col > self.grid_size - 1 - self.half_window:  
            self.attn_col = self.grid_size - 1 - self.half_window  

        if (
            self.schema_row < self.half_window  
        ):  # Check to make sure attention field is within bounds
            self.schema_row = self.half_window  
        if self.schema_row > self.grid_size - 1 - self.half_window:  
            self.schema_row = self.grid_size - 1 - self.half_window  
        if (
            self.schema_col < self.half_window  
        ):  # Check to make sure attention field is within bounds
            self.schema_col = self.half_window  
        if self.schema_col > self.grid_size - 1 - self.half_window:  
            self.schema_col = self.grid_size - 1 - self.half_window  
        
        

        if (
            self.ball_col < self.landing
        ):  # adjust to the right if ball is left of landing zone
            self.ball_col = self.ball_col + 1
        elif (
            self.ball_col > self.landing
        ):  # adjust to the left if ball is right of landing zone
            self.ball_col = self.ball_col - 1

        # Don't let the ball leave the playing field
        if self.ball_col < 0:  # Check to make sure the ball is within bounds
            self.ball_col = 0  # undo the mistake
        if self.ball_col > self.grid_size - 1:
            self.ball_col = self.grid_size - 1

        if self.decoupling == False:
            self.schema_row = self.attn_row
            self.schema_col = self.attn_col

        # Represent attention location:
        self.attn_rowspan = list(range(self.attn_row - self.half_window, self.attn_row + 1 + self.half_window)) #was-1 and +2
        self.attn_colspan = list(range(self.attn_col - self.half_window, self.attn_col + 1 + self.half_window)) #was-1 and +2

        self.schema_rowspan = list(range(self.schema_row - self.half_window, self.schema_row + 1 + self.half_window)) #was-1 and +2
        self.schema_colspan = list(range(self.schema_col - self.half_window, self.schema_col + 1 + self.half_window)) #was-1 and +2

        # Update the positions of the moving pieces
        self.paddle_loc = self.paddle_loc + move
        if (
            self.paddle_loc < 1 or self.paddle_loc > self.grid_size - 2
        ):  # Check to make sure paddle is within bounds
            self.paddle_loc = self.paddle_loc - move  # undo the mistake

        # Update ball position
        self.ball_row = (
            self.ball_row + 1
        )  # ball decends one space per timestep

       

        # Update the game state in the model
        self._state = self._append_to_memory(self._draw_state())

        # Scoring
        if (
            self.ball_row == self.grid_size - 1
        ):  # Check if the ball has hit the bottom
            self._episode_ended = True
            if (
                abs(self.ball_col - self.paddle_loc) <= 1
            ):  # Did the player catch the ball
                return ts.termination(
                    self._state, reward=self.catch_reward
                )  # Good!
            else:
                return ts.termination(
                    self._state, reward=-self.catch_reward
                )  # Bad!
        elif (
            self.ball_row in self.attn_rowspan
            and self.ball_col in self.attn_colspan
        ):  # small reward for attending the ball
            return ts.transition(
                np.array(self._state, dtype=np.int32),
                reward=self.attention_reward,
                #discount=self.discount_factor,
            )
        else:  # Punishment for attending empty space
            return ts.transition(
                np.array(self._state, dtype=np.int32),
                reward=-self.attention_reward,
                #discount=self.discount_factor,
            )

    def _draw_state(self):
        attentional_canvas = np.zeros(
            (self.grid_size, self.grid_size), dtype=np.int32
        )

        # Draw attention on the attentional space
        if self.attention_schema:
            attentional_canvas[
                self.schema_rowspan[0] : self.schema_rowspan[-1] + 1,
                self.schema_colspan[0] : self.schema_colspan[-1] + 1,
            ] = 1  # attention locus is a 3 by 3 square

        # Draw a noisy visual space
        noise_level = (
            self.noise
        )  # between 0 and 0.5, gives the percentage of playing feild to be filled with inserted while pixels
        noise_array = np.concatenate(
            (
                np.repeat(1, noise_level * (self.grid_size**2)),
                np.repeat(0, (1 - noise_level) * (self.grid_size**2)),
            )
        )

        visual_canvas = np.random.permutation(noise_array).reshape(
            (self.grid_size, self.grid_size)
        )
        visual_canvas = visual_canvas.astype("int32")
        visual_canvas[self.grid_size - 1, :] = 0  # Paddle row has no noise

        # Remove noise from attended spotlight
        if self.noise_removing:
            visual_canvas[
                self.attn_rowspan[0] : self.attn_rowspan[-1] + 1,
                self.attn_colspan[0] : self.attn_colspan[-1] + 1,
            ] = 0

        # Draw objects

        if (
            self.ball_row in self.attn_rowspan
            and self.ball_col in self.attn_colspan
        ):
            visual_canvas[self.ball_row, self.ball_col] = 2  # draw ball to make it distinguishable
        else:
            visual_canvas[self.ball_row, self.ball_col] = 2  # draw ball to make it distinguishable

        visual_canvas[
            self.grid_size - 1, self.paddle_loc - 1 : self.paddle_loc + 2
        ] = 1  # draw paddle, which always attended

        canvas = np.concatenate((visual_canvas, attentional_canvas), axis=0)


        # saving the positions to a file
        pos_a = (self.attn_row * 10) + self.attn_col
        pos_as = (self.schema_row * 10) + self.schema_col

        pos_list = np.asarray([int(pos_a), int(pos_as)], dtype=int)
        pos_list = np.reshape(pos_list, newshape=(1,2))

        path = Path(
                "./csvs/"
				+ "as_pos/"
				+ file_name
				+ ".csv")
    
        if path.is_file():
            results = np.loadtxt(path, dtype=int, delimiter=",")
            if len(results.shape) < 2:
                results = np.expand_dims(results, axis=0)
            results = np.concatenate([results, pos_list], axis=0)
            
        else:
            results = pd.DataFrame(pos_list) 

        # Save results df to file
        np.savetxt(path, results,fmt="%d", delimiter=",")

        return canvas

    def _append_to_memory(self, state):
        memory = copy.deepcopy(self.memory_buffer)
        memory = np.delete(
            memory, np.arange(self.grid_size * 2), 0
        )  # Delete the oldest memory
        updated_memory = np.append(
            memory, state, axis=0
        )  # Append most recent observation
        self.memory_buffer = updated_memory  # Update the memory buffer
        return updated_memory


def generate_plays(env, number_of_games, policy):
    for n in range(number_of_games):
        if(n) == 0:
            time_step = env._reset()

        for i in range(10):
            action_step = policy.action(time_step)
            time_step = env.step(action_step.action)


def make_heatmap(path):
    """
    makes a scatterplot of positions
    """
    path_to_store= Path("./csvs/"
            + "as_pos/" + "fully_squeezed_"
            + file_name
            + ".png")
    data = np.loadtxt(path, dtype=int, delimiter=",")

    x = data[:,0]
    x = x % 10
    y = data[:,1]
    y = y % 10

    combinations_counter = np.zeros((8,8))
    for idx, datapoint in enumerate(x):
        combinations_counter[datapoint-1, y[idx]-1] += 1

    relative_combinations = combinations_counter / np.sum(combinations_counter)
    relative_combinations_flat = np.reshape(relative_combinations, newshape=64)
    print(relative_combinations)
    plt.figure(figsize=(6,6))

    #plt.plot(a,a,'r',  alpha = 0.5)
    #plt.scatter(x,y, s=150, alpha=0.005)
    #plt.xlabel("Attention Window Column", fontsize=13)
    #plt.ylabel("Attention Schema Column", fontsize=13)
    #plt.savefig(path_to_store)

    path_to_store= Path("./csvs/"
            + "as_pos/" + "fully_squeezed2_"
            + file_name
            + ".png")
    
    axes = np.arange(8) + 1
    plt.scatter(y=(np.repeat(axes,8)), x=(np.arange(64)%8)+1, s=150, cmap="Blues", c=relative_combinations_flat) 

    cbar = plt.colorbar() 
    cbar.ax.tick_params(labelsize=12)
    cbar.set_label("Relative Frequency", fontsize=15)
    plt.xticks(axes, fontsize=13)
    plt.yticks(axes, fontsize=13)
    plt.ylabel("Attention Window Column", fontsize=15)
    plt.xlabel("Additional Resource Column", fontsize=15)

    plt.savefig(path_to_store)

# path where the positions are or will be stored
path = Path(
            "./csvs/"
            + "as_pos/"
            + file_name
            + ".csv")


environment = Env(attention_reward=attention_reward, catch_reward=catch_reward, noise_removing=noise_removing, attention_schema=attention_schema, 
                 window_size=window_size, discount_factor=discount_factor, look_back= look_back, noise=noise, 
                 decoupling=decoupling , random_schema_action=random_schema_action, actions=actions)
env = tf_py_environment.TFPyEnvironment(environment)


generate_new_plays = False #wheter to generate new plays or not
number_of_games = 4000 #number of games to be generated to gather data

if generate_new_plays:  
    generate_plays(env, number_of_games=number_of_games, policy=policy1)

make_heatmap(path)
