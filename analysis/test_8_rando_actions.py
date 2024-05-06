# train a nn to infer attention window position from attention schema

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
from tensorflow.keras import layers

random_schema_action = False
eight_actions = False

policy1 = tf.saved_model.load("../policy/policy_with_noise_"+"0"+"_random_emergence_ar_no_mem_ppo_3_act_1e-05_lr_"+str(1000)+"_units_3_windowsize_0.9_discount_factor")
policy2 = tf.saved_model.load("../policy/policy_with_noise_"+"0.25"+"_random_emergence_ar_no_mem_ppo_3_act_1e-05_lr_"+str(1000)+"_units_3_windowsize_0.9_discount_factor")
policy4 = tf.saved_model.load("../policy/policy_with_noise_"+"0.75"+"_random_emergence_ar_no_mem_ppo_3_act_1e-05_lr_"+str(1000)+"_units_3_windowsize_0.9_discount_factor")
policy5 = tf.saved_model.load("../policy/policy_with_noise_"+"1"+"_random_emergence_ar_no_mem_ppo_3_act_1e-05_lr_"+str(1000)+"_units_3_windowsize_0.9_discount_factor")
policy3 = tf.saved_model.load("../policy/policy_with_emergence_ar_no_mem_ppo_3_act_1e-05_lr_1000_units_3_windowsize_0.9_discount_factor")
policy6 = tf.saved_model.load("../policy/policy_with_noise_0.5_no_as_ar_no_mem_ppo_1000_units_3_windowsize_0.9_disc_factor")
policy7 = tf.saved_model.load("../policy/policy_with_as_no_mem_new_ppo_3_act_1e-05_lr_3_windowsize_0.9_discount_factor_500_units")
policy8 = tf.saved_model.load("../policy/policy_with_emergence_no_mem_ppo_3_act_1e-05_lr_1000_units_3_windowsize_0.9_discount_factor")
policy9 = tf.saved_model.load("../policy/policy_with_emergence")
#policies = [policy1, policy2, policy3, policy4, policy5]
#policies = [policy6]
policies = [policy9]
#results_agent_with_as_no_mem_new_ppo_3_act_1e-05_lr_3_windowsize_0.9_discount_factor_500_units
# what is task?

# generate new data and if yes how many?
generate_new_plays = True
num_generate_epochs = 2000

# normal emergent environment, where just the _draw_state() function is modified to allow storing the 
# images and positions
class EmergEnv(py_environment.PyEnvironment):
    """
    Class catch is the actual game.
    In the game, balls, represented by white tiles, fall from the top.
    The goal is to catch the balls with a paddle
    """

    def __init__(
        self,
        attention_reward=True,
        catch_reward=True,
        noise_removing=True,
        attention_schema=True,
        window_size:int=3,
        discount_factor=0.99,
        look_back=1,
        noise=0.5,
        random_schema_action = False
    ):
        if attention_reward:
            self.attention_reward = tf.constant(0.5, dtype=tf.float32)
        else:
            self.attention_reward = tf.constant(0, dtype=tf.int32)

        if catch_reward:
            self.catch_reward = tf.constant(2, dtype=tf.int32)
        else:
            self.catch_reward = tf.constant(0, dtype=tf.int32)

        self.total_attention_reward = 0
        self.total_catch_reward = 0


        self.noise_removing = noise_removing
        self.attention_schema = attention_schema
        self.window_size = window_size
        self.half_window = math.floor(self.window_size / 2)
        self.look_back = look_back
        self.noise = noise
        self.discount_factor = discount_factor
        self.random_schema_action = random_schema_action

        self.grid_size = 10
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(),
            dtype=np.int32,
            minimum=0,
            maximum=(3 * 2 * 3) - 1, #(8 * 2 * 8) - 1,
            name="action",
        )  # The number of possible actions is equal to the number of grid tiles times 3
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
            self.attn_col < self.half_window # was 1
        ):  # Check to make sure attention field is within bounds
            self.attn_col = self.half_window # was 1
        if self.attn_col > self.grid_size - 1 - self.half_window: # was - 2
            self.attn_col = self.grid_size - 1 - self.half_window # was - 2
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

        self._state = self._draw_state()

        self._episode_ended = False

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):

        # path = Path(
        #         "./csvs/test_8_random_acts/trained_w_no_as_noise_"+str(self.noise)+".csv")
        
        path = Path("./csvs/test_8_random_acts/emergence"+".csv")
        
        rewards = np.array([self.total_attention_reward, self.total_catch_reward])
        rewards = np.expand_dims(rewards, axis=-1).T
        if path.is_file():
            
            
            results = np.loadtxt(path, dtype=int, delimiter=",")            
            if len(results.shape) < 2:
                results = np.expand_dims(results, axis=0)
            results = np.concatenate([results, rewards], axis=0)
            
        else:

            results = pd.DataFrame(rewards) 
        

        # Save results df to file
        np.savetxt(path, results,fmt="%d", delimiter=",")


        self.total_attention_reward = 0
        self.total_catch_reward = 0
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
            self.attn_col < self.half_window # was 1
        ):  # Check to make sure attention field is within bounds
            self.attn_col = self.half_window # was 1
        if self.attn_col > self.grid_size - 1 - self.half_window: # was - 2
            self.attn_col = self.grid_size - 1 - self.half_window # was - 2
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
        self._state = self._draw_state()
        self.step_count = 0

        return ts.restart(self._state)
    
    def get_ball_col(self):
        return self.ball_col
    
    def _step(self, action):
        if self._episode_ended:
            return self._reset()

        self.step_count += 1  # Increment the step counter

        # here we define how action selection affects the location of the paddle
        """
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
        attn_delta_col, attn_delta_row = moves[attn]
        # Apply the change in attention locus
        self.attn_row = self.attn_row + attn_delta_row
        self.attn_col = self.attn_col + attn_delta_col

        schema_delta_col, schema_delta_row = moves[schema]
        # Apply the change in attention locus
        self.schema_row = self.schema_row + schema_delta_row
        self.schema_col = self.schema_col + schema_delta_col"""

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


        attn_delta_col, attn_delta_row = moves[attn]
        # Apply the change in attention locus
        self.attn_row = self.attn_row + attn_delta_row
        self.attn_col = self.attn_col + attn_delta_col
        # for 8 random actions
        # moves = np.array(
        #     [
        #         (0, 1),
        #         (1, 1),
        #         (1, 0),
        #         (1, -1),
        #         (0, -1),
        #         (-1, -1),
        #         (-1, 0),
        #         (-1, 1),
        #     ]
        # )

        # schema = np.random.randint(0,8)
        #until here
        schema_delta_col, schema_delta_row = moves[schema]
        # Apply the change in attention locus
        self.schema_row = self.schema_row + schema_delta_row
        self.schema_col = self.schema_col + schema_delta_col



        if (
            self.attn_row < self.half_window # was 1
        ):  # Check to make sure attention field is within bounds
            self.attn_row = self.half_window # was 1
        if self.attn_row > self.grid_size - 1 - self.half_window: # was - 2
            self.attn_row = self.grid_size - 1 - self.half_window # was - 2
        if (
            self.attn_col < self.half_window # was 1
        ):  # Check to make sure attention field is within bounds
            self.attn_col = self.half_window # was 1
        if self.attn_col > self.grid_size - 1 - self.half_window: # was - 2
            self.attn_col = self.grid_size - 1 - self.half_window # was - 

        if (
            self.schema_row < self.half_window # was 1
        ):  # Check to make sure attention field is within bounds
            self.schema_row = self.half_window # was 1
        if self.schema_row > self.grid_size - 1 - self.half_window: # was - 2
            self.schema_row = self.grid_size - 1 - self.half_window # was - 2
        if (
            self.schema_col < self.half_window # was 1
        ):  # Check to make sure attention field is within bounds
            self.schema_col = self.half_window # was 1
        if self.schema_col > self.grid_size - 1 - self.half_window: # was - 2
            self.schema_col = self.grid_size - 1 - self.half_window # was - 
        
        

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

        """self.attn_row = self.ball_row
        self.attn_col = self.ball_col

        if (
            self.attn_row < self.half_window # was 1
        ):  # Check to 
        make sure attention field is within bounds
            self.attn_row = self.half_window # was 1
        if self.attn_row > self.grid_size - 1 - self.half_window: # was - 2
            self.attn_row = self.grid_size - 1 - self.half_window # was - 2
        if (
            self.attn_col < self.half_window # was 1
        ):  # Check to make sure attention field is within bounds
            self.attn_col = self.half_window # was 1
        if self.attn_col > self.grid_size - 1 - self.half_window: # was - 2
            self.attn_col = self.grid_size - 1 - self.half_window # was - 2


         """



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
        self._state = self._draw_state()

        # Scoring
        if (
            self.ball_row == self.grid_size - 1
        ):  # Check if the ball has hit the bottom
            self._episode_ended = True
            if (
                abs(self.ball_col - self.paddle_loc) <= 1
            ):  # Did the player catch the ball
                self.total_catch_reward += self.catch_reward
                return ts.termination(
                    self._state, reward=self.catch_reward
                )  # Good!
            else:
                self.total_catch_reward -= self.catch_reward

                return ts.termination(
                    self._state, reward=-self.catch_reward
                )  # Bad!
        elif (
            self.ball_row in self.attn_rowspan
            and self.ball_col in self.attn_colspan
        ):  # small reward for attending the ball
            self.total_attention_reward += self.attention_reward
            return ts.transition(
                np.array(self._state, dtype=np.int32),
                reward=self.attention_reward,
                #discount=self.discount_factor,
            )
        else:  # Punishment for attending empty space
            self.total_attention_reward -= self.attention_reward

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
        visual_canvas[
            self.grid_size - 1, self.paddle_loc - 1 : self.paddle_loc + 2
        ] = 1  # draw paddle, which always attended
        
        if (
            self.ball_row in self.attn_rowspan
            and self.ball_col in self.attn_colspan
        ):
            visual_canvas[self.ball_row, self.ball_col] = 1  # draw ball
        else:
            visual_canvas[self.ball_row, self.ball_col] = 1  # draw ball



        canvas = np.concatenate((visual_canvas, attentional_canvas), axis=0)

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
class Env(py_environment.PyEnvironment):
    """
    Class catch is the actual game.
    In the game, balls, represented by white tiles, fall from the top.
    The goal is to catch the balls with a paddle
    """

    def __init__(
        self,
        attention_reward=True,
        catch_reward=True,
        noise_removing=True,
        attention_schema=True,
        window_size:int=3,
        discount_factor=0.99,
        look_back=1,
        noise=0.5,
        actions=3,
        random_schema_action = False,
        fixed_attention_schema = False
    ):
        if attention_reward:
            self.attention_reward = tf.constant(0.5, dtype=tf.float32)
        else:
            self.attention_reward = tf.constant(0, dtype=tf.int32)

        if catch_reward:
            self.catch_reward = tf.constant(2, dtype=tf.int32)
        else:
            self.catch_reward = tf.constant(0, dtype=tf.int32)

        self.total_attention_reward = 0
        self.total_catch_reward = 0

        self.noise_removing = noise_removing
        self.attention_schema = attention_schema
        self.window_size = window_size
        self.half_window = math.floor(self.window_size / 2)
        self.look_back = look_back
        self.noise = noise
        self.actions=actions
        self.discount_factor = discount_factor
        self.random_schema_action = random_schema_action
        self.fixed_attention_schema = fixed_attention_schema

        self.grid_size = 10
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(),
            dtype=np.int32,
            minimum=0,
            maximum= (self.actions*2) - 1, #5, # (8 * 2) - 1,
            name="action",
        )  # The number of possible actions is equal to the number of grid tiles times 3
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(self.grid_size * 2 * self.look_back, self.grid_size),
            dtype=np.int32,
            minimum=0,
            maximum=1,
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
            self.attn_col < self.half_window # was 1
        ):  # Check to make sure attention field is within bounds
            self.attn_col = self.half_window # was 1
        if self.attn_col > self.grid_size - 1 - self.half_window: # was - 2
            self.attn_col = self.grid_size - 1 - self.half_window # was - 2
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

        path = Path("./csvs/test_8_random_acts/new_normal_ppo_3_acts_as_ar_"+str(noise)+".csv")
        rewards = np.array([self.total_attention_reward, self.total_catch_reward])
        rewards = np.expand_dims(rewards, axis=-1).T
        if path.is_file():
            
            
            results = np.loadtxt(path, dtype=int, delimiter=",")            
            if len(results.shape) < 2:
                results = np.expand_dims(results, axis=0)
            results = np.concatenate([results, rewards], axis=0)
            
        else:

            results = pd.DataFrame(rewards) 
        

        # Save results df to file
        np.savetxt(path, results,fmt="%d", delimiter=",")
        self.total_attention_reward = 0
        self.total_catch_reward = 0

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
            self.attn_col < self.half_window # was 1
        ):  # Check to make sure attention field is within bounds
            self.attn_col = self.half_window # was 1
        if self.attn_col > self.grid_size - 1 - self.half_window: # was - 2
            self.attn_col = self.grid_size - 1 - self.half_window # was - 2
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
       
        if self.actions == 8:
            if action in np.arange(0, 8):  # left
                move = -1
            elif action in np.arange(8, 16):
                move = 1  # right

            # Here we define how action selection affects the locus of attention
            # Rescale action selection to exclude the chosen move
            temp_vec = np.array([0, 8])
            temp_mat = np.array(
                [
                    temp_vec,
                    temp_vec + 1,
                    temp_vec + 2,
                    temp_vec + 3,
                    temp_vec + 4,
                    temp_vec + 5,
                    temp_vec + 6,
                    temp_vec + 7,
                ]
            )
            attn_action = np.argwhere(temp_mat == action)[0][0]
            # Attention movement options are stationary or 8 possible directions
            attn_moves = np.array(
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
        elif self.actions==3:
            if action in np.arange(0, 3):  # left
                move = -1
            elif action in np.arange(3, 6):
                move = 1  # right

            # Here we define how action selection affects the locus of attention
            # Rescale action selection to exclude the chosen move
            temp_vec = np.array([0, 3])
            temp_mat = np.array(
                [
                    temp_vec,
                    temp_vec + 1,
                    temp_vec + 2,
                    
                ]
            )
            attn_action = np.argwhere(temp_mat == action)[0][0]
            # Attention movement options are stationary or 8 possible directions
            attn_moves = np.array(
                [
                    (0, 1),
                    (1, 1),
                    (-1, 1),
                ]
            )

        
        if self.random_schema_action:
            schema = np.random.randint(0,3)

            schema_delta_col, schema_delta_row = attn_moves[schema]

            self.schema_row = self.schema_row + schema_delta_row
            self.schema_col = self.schema_col + schema_delta_col

            if (
                self.schema_row < self.half_window # was 1
            ):  # Check to make sure attention field is within bounds
                self.schema_row = self.half_window # was 1
            if self.schema_row > self.grid_size - 1 - self.half_window: # was - 2
                self.schema_row = self.grid_size - 1 - self.half_window # was - 2
            if (
                self.schema_col < self.half_window # was 1
            ):  # Check to make sure attention field is within bounds
                self.schema_col = self.half_window # was 1
            if self.schema_col > self.grid_size - 1 - self.half_window: # was - 2
                self.schema_col = self.grid_size - 1 - self.half_window # was -   

            self.schema_rowspan = list(range(self.schema_row - self.half_window, self.schema_row + 1 + self.half_window)) #was-1 and +2
            self.schema_colspan = list(range(self.schema_col - self.half_window, self.schema_col + 1 + self.half_window)) #was-1 and +2   

        elif self.fixed_attention_schema:  
            if self.step_count == 0:
                self.schema_row = self.ball_row
                self.schema_col = self.ball_col
            elif self.step_count == 1:
                self.schema_row += 1
            
                
            

        delta_col, delta_row = attn_moves[attn_action]
        # Apply the change in attention locus
        self.attn_row = self.attn_row + delta_row
        self.attn_col = self.attn_col + delta_col

        if (
            self.attn_row < self.half_window # was 1
        ):  # Check to make sure attention field is within bounds
            self.attn_row = self.half_window # was 1
        if self.attn_row > self.grid_size - 1 - self.half_window: # was - 2
            self.attn_row = self.grid_size - 1 - self.half_window # was - 2
        if (
            self.attn_col < self.half_window # was 1
        ):  # Check to make sure attention field is within bounds
            self.attn_col = self.half_window # was 1
        if self.attn_col > self.grid_size - 1 - self.half_window: # was - 2
            self.attn_col = self.grid_size - 1 - self.half_window # was - 
        
        

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

        
        # Represent attention location:
        self.attn_rowspan = list(range(self.attn_row - self.half_window, self.attn_row + 1 + self.half_window)) #was-1 and +2
        self.attn_colspan = list(range(self.attn_col - self.half_window, self.attn_col + 1 + self.half_window)) #was-1 and +2


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
            ):
                self.total_catch_reward += self.catch_reward

                  # Did the player catch the ball
                return ts.termination(
                    self._state, reward=self.catch_reward
                )  # Good!
            else:
                self.total_catch_reward -= self.catch_reward

                return ts.termination(
                    self._state, reward=-self.catch_reward
                )  # Bad!
        elif (
            self.ball_row in self.attn_rowspan
            and self.ball_col in self.attn_colspan
        ):  # small reward for attending the ball
            self.total_attention_reward += self.attention_reward

            return ts.transition(
                np.array(self._state, dtype=np.int32),
                reward=self.attention_reward,
                #discount=self.discount_factor,
            )
        else:  # Punishment for attending empty space
            self.total_attention_reward -= self.attention_reward
 
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
            if self.random_schema_action or self.fixed_attention_schema:
                attentional_canvas[
                    self.schema_rowspan[0] : self.schema_rowspan[-1] + 1,
                    self.schema_colspan[0] : self.schema_colspan[-1] + 1,
                ] = 1  # attention locus is a 3 by 3 square
            else:
                attentional_canvas[
                    self.attn_rowspan[0] : self.attn_rowspan[-1] + 1,
                    self.attn_colspan[0] : self.attn_colspan[-1] + 1,
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
            visual_canvas[self.ball_row, self.ball_col] = 1  # draw ball
        else:
            visual_canvas[self.ball_row, self.ball_col] = 1  # draw ball

        visual_canvas[
            self.grid_size - 1, self.paddle_loc - 1 : self.paddle_loc + 2
        ] = 1  # draw paddle, which always attended

        canvas = np.concatenate((visual_canvas, attentional_canvas), axis=0)

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

#noises = [0,0.25,0.5,0.75,1]
noises = [0.5]
if generate_new_plays:
    for i,policy in enumerate(policies):
        noise = noises[i]
        agent = EmergEnv(attention_schema=True,random_schema_action=random_schema_action)
        #agent = Env()
        env = tf_py_environment.TFPyEnvironment(agent)
        generate_plays(env, number_of_games=num_generate_epochs, policy=policy)

for i,policy in enumerate(policies):
        noise = noises[i]

        path = Path("./csvs/test_8_random_acts/emergence"+".csv")
        
        results = np.loadtxt(path, dtype=int, delimiter=",")[1:]

        print(np.mean(results[:,0]), np.mean(results[:,1]))

# for i,policy in enumerate(policies):
#         noise = noises[i]

#         path = Path("./csvs/test_8_random_acts/new_normal_ppo_3_acts_as_ar_"+str(noise)+".csv")
        
#         results = np.loadtxt(path, dtype=int, delimiter=",")[1:]

#         print(np.mean(results[:,0]), np.mean(results[:,1]))
