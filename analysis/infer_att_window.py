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


from sklearn.model_selection import KFold


"""
This file is used to train a neural network to infer the attention window position from the visual input.
The K-fold cross validation is used to evaluate the model. 
The data can be generated in this file as well. 
"""


decode_col = True #decode only column (True) or whole position (False)
part_of_image = "lower_half" # which part of the image to decode fro: can be whole_image , upper_half, or lower_half
policy1 = tf.saved_model.load("../policy/policy_with_emergence_ar_no_mem_ppo_3_act_1e-05_lr_1000_units_3_windowsize_0.9_discount_factor")

# generate new data and if yes how many?
generate_new_plays = False
num_generate_epochs = 2000

# nn training settings
nn_train_epochs = 30 # epochs

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

noise_str = str(noise)

#name of the file where the data is stored
if random_schema_action:
    schema_str = "_random_schema"
else:
    schema_str = ""
if part_of_image == "lower_half":
    image_str = "_lower_half_"
elif part_of_image == "upper_half":
    image_str = "_upper_half_"
else:
    image_str = "_image_"

if decode_col:
    col_str = "col_"
else: 
    col_str = ""
file_name = "aw_from_image_" + noise_str + "_noise"+schema_str+"_as_ball_full_image_"
save_file_name = "aw_"+col_str+"from_"+image_str+"image_" + noise_str + "_noise"+schema_str+"_as"



# normal emergent environment, where just the _draw_state() function is modified to allow storing the 
# images and positions

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
            visual_canvas[self.ball_row, self.ball_col] = 1  # draw ball
        else:
            visual_canvas[self.ball_row, self.ball_col] = 1  # draw ball

        visual_canvas[
            self.grid_size - 1, self.paddle_loc - 1 : self.paddle_loc + 2
        ] = 1  # draw paddle, which always attended

        canvas = np.concatenate((visual_canvas, attentional_canvas), axis=0)

        # save image and attentional window position

        # for whole image and aw pos
        image_1d = np.concatenate((np.reshape(visual_canvas, newshape=(100)), np.reshape(attentional_canvas, newshape=(100))), axis=0)

        pos_a = (self.attn_row * 10) + self.attn_col
        pos_list = np.concatenate((np.asarray([pos_a]), image_1d))
        pos_list = np.reshape(pos_list, newshape=(1,201))

    

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
    environment = Env(attention_reward=attention_reward, catch_reward=catch_reward, noise_removing=noise_removing, attention_schema=attention_schema, 
                 window_size=window_size, discount_factor=discount_factor, look_back= look_back, noise=noise, 
                 decoupling=decoupling , random_schema_action=random_schema_action, actions = actions)
    env = tf_py_environment.TFPyEnvironment(environment)
    generate_plays(env, number_of_games=num_generate_epochs, policy=policy1)

#make nn
data = np.loadtxt(path, dtype=int, delimiter=",")

y = data[:,0] #attention window position

if part_of_image == "whole_image":
    x = data[:,1:]
elif part_of_image == "upper_half":
    x = data[:,1:101]
elif part_of_image == "lower_half":
    x = data[:,101:]

if decode_col:
    aw_col = y % 10
    aw_col -= 1

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

# cross validation
targets = y
inputs = x


# K-fold Cross Validation model evaluation
fold_no = 1
acc_per_fold = []
num_folds = 10


# Define the K-fold Cross Validator
kfold = KFold(n_splits=num_folds, shuffle=True)

for train, test in kfold.split(inputs, targets):

    # Define the model architecture
    nn = tf.keras.Sequential([
    tf.keras.layers.Dense(1000),
    tf.keras.layers.Dense(1000),
    tf.keras.layers.Dense(1000),
    tf.keras.layers.Dense(1000),
    tf.keras.layers.Dense(y_values, 'softmax')
    ])

    acc = tf.keras.metrics.CategoricalAccuracy()
    nn.compile(loss=tf.keras.losses.CategoricalCrossentropy(), optimizer=tf.keras.optimizers.Adam(), metrics=[acc])


    # Generate a print
    print('------------------------------------------------------------------------')
    print(f'Training for fold {fold_no} ...')
    input = np.take(inputs, train, axis=0)
    target = np.take(targets, train, axis=0)
    # Fit data to model
    history = nn.fit(input, target,
                epochs=nn_train_epochs)

    input = np.take(inputs, test, axis=0)
    target = np.take(targets, test, axis=0)
    # Generate generalization metrics
    scores = nn.evaluate(input, target, verbose=0)
    acc_per_fold.append(scores[1] * 100)

    # Increase fold number
    fold_no = fold_no + 1
print("accuracy per fold:")
print(acc_per_fold)
print("mean accuracy: " + str(np.mean(acc_per_fold)))

