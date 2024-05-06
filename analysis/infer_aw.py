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

noise = 1
noise_str = str(noise)
units = 1000
random_schema_action = True
eight_actions = False
pseudo_random = False
decode_col = True
part_of_image = "lower_half" # either whole_image , upper_half, or lower_half
policy1 = tf.saved_model.load("../policy/policy_with_emergence_ar_no_mem_ppo_3_act_1e-05_lr_1000_units_3_windowsize_0.9_discount_factor")
# for perfect agent:
policy2 = tf.saved_model.load("../policy/policy_with_as_ppo_3_act_no_mem_random_1e-05_lr_3_windowsize_0.9_discount_factor_1000_units")

#policy3 = tf.saved_model.load("../policy/policy_with_noise_"+noise_str+"_random_emergence_ar_no_mem_ppo_3_act_1e-05_lr_"+str(units)+"_units_3_windowsize_0.9_discount_factor")
policy3 = tf.saved_model.load("../policy/policy_with_emergence_ar_no_mem_ppo_3_act_1e-05_lr_1000_units_3_windowsize_0.9_discount_factor")

# what is task?
get_image_and_ball_pos = False #predict ball position from image
get_image_and_aw_pos = True #predict aw position from image
get_aw_and_as_pos = False   #predict aw position from as position
get_image_and_as_row = False #predict aw position from image and attention schema row
get_image_ball_pos_and_as_pos = False #predict aw position from image and attention schema row
get_image_and_ball_row = False #predict ball position from vis input and as row

# generate new data and if yes how many?
generate_new_plays = False
num_generate_epochs = 2000

# nn training settings
nn_train_epochs = 30 # epochs

#train with visual input + schema input
#train with schema input only
#train with visual input only

#name of the file where the data is stored
if get_image_and_aw_pos:
    if random_schema_action:
        schema_str = "_random_schema"
    else:
        schema_str = ""
    if eight_actions:
        action_str = "_8_random_acts"
    else:
        action_str = ""
    if part_of_image == "lower_half":
        image_str = "_lower_half_"
    elif part_of_image == "upper_half":
        image_str = "_upper_half_"
    else:
        image_str = "_image_"

    if pseudo_random:
        pseudo_str = "_pseudo_random_"
    else:
        pseudo_str = ""

    if decode_col:
        col_str = "col_"
    else: 
        col_str = ""
    file_name = "aw_from_image_" + noise_str + "_noise"+schema_str+pseudo_str+action_str+"_as_ball_full_image_" + str(units) + "units"
    save_file_name = "aw_"+col_str+"from_image_" + noise_str + "_noise"+schema_str+pseudo_str+action_str+"_infer_ball_from_upper_half_and_as_(part)_pos_emergence_ar_no_mem_ppo_3_act_1e-05_lr_"+str(units)+"_units_3_windowsize_0.9_discount_factor"

elif get_image_and_ball_pos:
    file_name = "ball_full_image_perfect_tracking"
    save_file_name = "ball_"+part_of_image+"_perfect_tracking"

elif get_aw_and_as_pos:
    file_name = "aw_as_"+noise_str+"_noise_pos_as_emergence_ar_no_mem_ppo_3_act_1e-05_lr_1000_units_3_windowsize_0.9_discount_factor"
    save_file_name = "aw_as_"+noise_str+"_noise_pos_as_emergence_ar_no_mem_ppo_3_act_1e-05_lr_1000_units_3_windowsize_0.9_discount_factor"

elif get_image_and_as_row:
    file_name = "aw_image_as_pos_as_emergence_ar_no_mem_ppo_3_act_1e-05_lr_1000_units_3_windowsize_0.9_discount_factor"
    save_file_name = "infer_aw_from_upper_half_and_as_row_emergence_ar_no_mem_ppo_3_act_1e-05_lr_1000_units_3_windowsize_0.9_discount_factor"

elif get_image_ball_pos_and_as_pos:
    if random_schema_action:
        schema_str = "_random_schema"
    else:
        schema_str = ""
    file_name = noise_str + "_noise"+schema_str+"_as_ball_full_image_" + str(units) + "units"
    save_file_name = noise_str + "_noise"+schema_str+"_infer_ball_from_upper_half_and_as_(part)_pos_emergence_ar_no_mem_ppo_3_act_1e-05_lr_"+str(units)+"_units_3_windowsize_0.9_discount_factor"

elif get_image_and_ball_row:
    file_name = "ball_full_image_emergence_ar_no_mem_ppo_3_act_1e-05_lr_1000_units_3_windowsize_0.9_discount_factor"
    save_file_name = "infer_ball_from_upper_half_and_timestep_emergence_ar_no_mem_ppo_3_act_1e-05_lr_1000_units_3_windowsize_0.9_discount_factor"


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
        random_schema_action = False,
        pseudo_random = False
    ):
        if attention_reward:
            self.attention_reward = tf.constant(0.5, dtype=tf.float32)
        else:
            self.attention_reward = tf.constant(0, dtype=tf.int32)

        if catch_reward:
            self.catch_reward = tf.constant(2, dtype=tf.int32)
        else:
            self.catch_reward = tf.constant(0, dtype=tf.int32)

        
        self.pseudo_random = pseudo_random

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

        if self.pseudo_random:
            self.pseudo_row = self.schema_row
            self.pseudo_col = self.schema_col
            self.pseudo_rowspan = self.schema_rowspan
            self.pseudo_colspan = self.schema_colspan

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

        if pseudo_random:
            self.pseudo_row = self.schema_row
            self.pseudo_col = self.schema_col
            self.pseudo_rowspan = self.schema_rowspan
            self.pseudo_colspan = self.schema_colspan


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

        #schema = np.random.randint(0,3)

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

        if self.pseudo_random:
            pseudo_schema = np.random.randint(0,3)
        # Attention movement options are stationary or 8 possible directions
            moves = np.array(
                [
                    (0, 1),
                    (1, 1),
                    (-1, 1),
                ]
            )

            pseudo_schema_delta_col, pseudo_schema_delta_row = moves[pseudo_schema]
            # Apply the change in attention locus
            self.pseudo_row = self.pseudo_row + pseudo_schema_delta_row
            self.pseudo_col = self.pseudo_col + pseudo_schema_delta_col






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
        if self.pseudo_random:

            if (
                self.pseudo_row < self.half_window # was 1
            ):  # Check to make sure attention field is within bounds
                self.pseudo_row = self.half_window # was 1
            if self.pseudo_row > self.grid_size - 1 - self.half_window: # was - 2
                self.pseudo_row = self.grid_size - 1 - self.half_window # was - 2
            if (
                self.pseudo_col < self.half_window # was 1
            ):  # Check to make sure attention field is within bounds
                self.pseudo_col = self.half_window # was 1
            if self.pseudo_col > self.grid_size - 1 - self.half_window: # was - 2
                self.pseudo_col = self.grid_size - 1 - self.half_window # was - 
            
        

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
        if self.pseudo_random:
            self.pseudo_rowspan = list(range(self.pseudo_row - self.half_window, self.pseudo_row + 1 + self.half_window)) #was-1 and +2
            self.pseudo_colspan = list(range(self.pseudo_col - self.half_window, self.pseudo_col + 1 + self.half_window)) #was-1 and +2

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

        if self.pseudo_random:
            pseudo_attentional_canvas = np.zeros(
                (self.grid_size, self.grid_size), dtype=np.int32)

            pseudo_attentional_canvas[
                self.pseudo_rowspan[0] : self.pseudo_rowspan[-1] + 1,
                self.pseudo_colspan[0] : self.pseudo_colspan[-1] + 1,
            ] = 1

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

        # for single position values
        if get_aw_and_as_pos:
            pos_a = (self.attn_row * 10) + self.attn_col
            pos_as = (self.schema_row * 10) + self.schema_col
            pos_list = np.asarray([int(pos_a), int(pos_as)], dtype=int)
            pos_list = np.reshape(pos_list, newshape=(1,2))

        # for whole image and aw pos
        if get_image_and_aw_pos:
            image_1d = np.concatenate((np.reshape(visual_canvas, newshape=(100)), np.reshape(attentional_canvas, newshape=(100))), axis=0)

            if self.pseudo_random:
                image_1d = np.concatenate((np.reshape(visual_canvas, newshape=(100)), np.reshape(pseudo_attentional_canvas, newshape=(100))), axis=0)


            pos_a = (self.attn_row * 10) + self.attn_col
            pos_list = np.concatenate((np.asarray([pos_a]), image_1d))
            pos_list = np.reshape(pos_list, newshape=(1,201))

        # for whole image and ball pos
        if get_image_and_ball_pos:
            pos_ball = (self.ball_row * 10) + self.ball_col
            image_1d = np.concatenate((np.reshape(visual_canvas, newshape=(100)), np.reshape(attentional_canvas, newshape=(100))), axis=0)
            pos_list = np.concatenate((np.asarray([pos_ball]), image_1d))
            pos_list = np.reshape(pos_list, newshape=(1,201))

        if get_image_ball_pos_and_as_pos:
            pos_ball = (self.ball_row * 10) + self.ball_col
            image_1d = np.concatenate((np.reshape(visual_canvas, newshape=(100)), np.reshape(attentional_canvas, newshape=(100))), axis=0)
            col = self.schema_col
            row = self.schema_row
            pos_list = np.concatenate((np.asarray([col, row]), image_1d))
            pos_list = np.concatenate((np.asarray([pos_ball]), pos_list))
            pos_list = np.reshape(pos_list, newshape=(1,203))


        path = Path(
                "./csvs/"
				+ "aw_as_pos/"
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

# matplotlib for rendering


class PerfectEnv(py_environment.PyEnvironment):
    """
    Class catch is the actual game.
    In the game, balls, represented by white tiles, fall from the top.
    The goal is to catch the balls with a paddle
    """

    def __init__(
        self,
        attention_reward=False,
        catch_reward=True,
        noise_removing=True,
        attention_schema=True,
        window_size:int=3,
        discount_factor=0.9,
        look_back=1,
        noise=0.5,
        actions=3,
        random_schema_action = False,
        fixed_attention_schema = False,
        perfect_actions = True
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
        self.actions=actions
        self.discount_factor = discount_factor
        self.random_schema_action = random_schema_action
        self.fixed_attention_schema = fixed_attention_schema
        self.perfect_actions = perfect_actions

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
            # for 8 random actions
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

            schema = np.random.randint(0,8)
            #until here

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

        if self.perfect_actions:  
        
            self.attn_row = self.ball_row
            self.attn_col = self.ball_col

            self.schema_row = self.ball_row
            self.schema_col = self.ball_col

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

        if get_aw_and_as_pos:
            pos_a = (self.attn_row * 10) + self.attn_col
            pos_as = (self.schema_row * 10) + self.schema_col

            pos_list = np.asarray([int(pos_a), int(pos_as)], dtype=int)
            pos_list = np.reshape(pos_list, newshape=(1,2))

        # for whole image and aw pos
        if get_image_and_aw_pos:
            pos_a = (self.attn_row * 10) + self.attn_col
            image_1d = np.concatenate((np.reshape(visual_canvas, newshape=(100)), np.reshape(attentional_canvas, newshape=(100))), axis=0)
            pos_list = np.concatenate((np.asarray([pos_a]), image_1d))
            pos_list = np.reshape(pos_list, newshape=(1,201))

        # for whole image and ball pos
        if get_image_and_ball_pos:
            pos_ball = (self.ball_row * 10) + self.ball_col
            image_1d = np.concatenate((np.reshape(visual_canvas, newshape=(100)), np.reshape(attentional_canvas, newshape=(100))), axis=0)
            pos_list = np.concatenate((np.asarray([pos_ball]), image_1d))
            pos_list = np.reshape(pos_list, newshape=(1,201))

        if get_image_ball_pos_and_as_pos:
            pos_ball = (self.ball_row * 10) + self.ball_col
            image_1d = np.concatenate((np.reshape(visual_canvas, newshape=(100)), np.reshape(attentional_canvas, newshape=(100))), axis=0)
            col = self.schema_col
            row = self.schema_row
            pos_list = np.concatenate((np.asarray([col, row]), image_1d))
            pos_list = np.concatenate((np.asarray([pos_ball]), pos_list))
            pos_list = np.reshape(pos_list, newshape=(1,203))

        path = Path(
                "./csvs/"
				+ "aw_as_pos/"
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


path = Path(
            "./csvs/"
            + "aw_as_pos/"
            + file_name
            + ".csv")


if generate_new_plays:
    agent = EmergEnv(noise=noise, random_schema_action=random_schema_action, pseudo_random=pseudo_random)
    #agent = PerfectEnv(look_back=1)
    env = tf_py_environment.TFPyEnvironment(agent)
    #generate_plays(env, number_of_games=num_generate_epochs, policy=policy1)
    # generate_plays(env, number_of_games=num_generate_epochs, policy=policy2)
    generate_plays(env, number_of_games=num_generate_epochs, policy=policy3)

#make nn
data = np.loadtxt(path, dtype=int, delimiter=",")


y = data[:,0] #a window position or ball pos

if part_of_image == "whole_image":
    x = data[:,1:]
elif part_of_image == "upper_half":
    x = data[:,1:101]
elif part_of_image == "lower_half":
    x = data[:,101:]

#when x is a position
if get_aw_and_as_pos:

    # as_col = x % 10
    # as_col -= 1
    # print(as_col[0])
    # as_col = tf.cast(as_col, tf.int64)

    # one_hot = tf.one_hot(as_col,depth=8, axis=-1)
    # x_vis = tf.cast(one_hot, tf.int64)


    x_vis = np.zeros(shape=(x.shape[0], 100))
    for x_pos in x:
        image = np.zeros(shape=(10,10))
        col = int(x_pos % 10)
        row = int(x_pos // 10)
        cols = range(col-1, col+2)
        rows = range(row-1, row+2)
        for c in cols:
            for r in rows:
                image[r,c] = 1
        #make 2d
        image = tf.reshape(image, shape=(100))
        x_vis[x_pos,:] = image
    x_vis = tf.convert_to_tensor(x_vis)
    x_vis =tf.cast(x_vis, tf.int64)
    x_vis = tf.expand_dims(x_vis, axis=1) 

elif get_image_and_aw_pos:
    # schema = data[:,101:]
    # x = data[:,1:101]

    # schema_idxes = np.where(np.array(schema) > 0)
    # print(np.array(schema).shape)
    # print(np.array(schema_idxes).shape)

    # schema_idxes = np.reshape(schema_idxes[1], newshape=(x.shape[0], 9))
    # schema_idxes = np.array(schema_idxes)
    # col = ((schema_idxes[:,0] % 10) +1)
    # row = ((schema_idxes[:,0] // 10) +1)
    # col = np.reshape(col, newshape=(x.shape[0],1))
    # row = np.reshape(row, newshape=(x.shape[0],1))
    # x_vis = tf.concat((x,row), axis=1)
    # x_vis = tf.concat((x_vis,col), axis=1)

    x_vis = x


elif get_image_and_as_row:
    as_row = y // 10
    as_row = as_row - 1
    one_hot = tf.one_hot(as_row,8,dtype=tf.int64)
    x_vis = tf.cast(x, tf.int64)
    x_vis = tf.concat((x_vis, one_hot), axis=1)

elif get_image_ball_pos_and_as_pos:
    if part_of_image == "upper_half":

        columns = tf.concat((tf.convert_to_tensor([1]), tf.range(3,103)), axis=-1)
        x = data[:,columns]
        x = data[:,1:103]
        # print(data)
    as_col = y % 10
    one_hot = tf.one_hot(as_col,10,dtype=tf.int64)
    x_vis = tf.cast(x, tf.int64)
    x_vis = tf.concat((x_vis, one_hot), axis=1)

elif get_image_and_ball_row:
    time_step = y // 10
    print(time_step)
    print(y)
    time_step = np.where(time_step>7,7,time_step)
    print(time_step)
    one_hot = tf.one_hot(time_step,8,dtype=tf.int64)
    x_vis = tf.cast(x, tf.int64)
    x_vis = tf.concat((x_vis, one_hot), axis=1)

else:
    #when x is not aw position
    x_vis = tf.cast(x, tf.int64)

if get_image_and_ball_pos or get_image_and_ball_row or get_image_ball_pos_and_as_pos: 
    # y_values = 100
    # y = tf.one_hot(y,y_values)

    aw_col = y % 10
    aw_col -= 1
    aw_col = tf.cast(aw_col, tf.int64)

    y = tf.one_hot(aw_col,8,dtype=tf.int64)
    y_values = 8
elif get_aw_and_as_pos:

    aw_col = y % 10
    aw_col -= 1
    aw_col = tf.cast(aw_col, tf.int64)

    y = tf.one_hot(aw_col,8,dtype=tf.int64)
    y_values = 8
    y = tf.expand_dims(y, axis=1)

elif get_image_and_aw_pos:
    if decode_col:
        aw_col = y % 10
        aw_col -= 1
        print(aw_col)

        aw_col = tf.cast(aw_col, tf.int64)

        y = tf.one_hot(aw_col,8,dtype=tf.int64)
        y_values = 8
        #y = tf.expand_dims(y, axis=1)


else:
    #if y is attention window pos
    y_row = y // 10
    y_col = y % 10
    y = (y_row-1)*8 + y_col-1
    y_values = 64
    y = tf.one_hot(y,y_values)



y=tf.cast(y,tf.int64)


#length dataset
length_test = int(x_vis.shape[0] /10)
x_test = x_vis[:length_test,:]
y_test = y[:length_test,:]
x_train = x_vis[length_test:,:]
y_train = y[length_test:,:]
print(x_test.shape, x_train.shape, y_test.shape, y_train.shape)


nn = tf.keras.Sequential([
    layers.Dense(1000),
    layers.Dense(1000),
    layers.Dense(1000),
    layers.Dense(1000),
    layers.Dense(y_values, 'softmax')
])

acc = tf.keras.metrics.CategoricalAccuracy()
acc_5 = tf.keras.metrics.TopKCategoricalAccuracy()
nn.compile(loss=tf.keras.losses.CategoricalCrossentropy(), optimizer=tf.keras.optimizers.Adam(), metrics=[acc, acc_5])

nn.fit(x_train,y_train,epochs=26)
nn.evaluate(x_test,y_test)

nn.fit(x_train,y_train,epochs=1)
nn.evaluate(x_test,y_test)
nn.fit(x_train,y_train,epochs=1)
nn.evaluate(x_test,y_test)
nn.fit(x_train,y_train,epochs=1)
nn.evaluate(x_test,y_test)
nn.fit(x_train,y_train,epochs=1)
nn.evaluate(x_test,y_test)


model_path = Path("./csvs/"
            + "aw_as_pos/"
            + "model_"
            + save_file_name)

nn.save_weights(model_path)

