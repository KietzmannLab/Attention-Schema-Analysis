import os
import shutil
import zipfile
from pathlib import Path

import pandas as pd
import tensorflow as tf
import tf_agents
from .environment import Train
from tf_agents.agents.dqn import dqn_agent
from tf_agents.drivers import dynamic_step_driver
from tf_agents.environments import tf_py_environment, utils
from tf_agents.networks import q_network
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.utils import common


TESTING_ITERS = 1000
LOOK_BACK = 10
NOISE = 0.5

train_agent = Train(
    look_back=LOOK_BACK, noise=NOISE
)

train_agent_no_ar = Train(
    attention_reward=False, look_back=LOOK_BACK, noise=NOISE
)
test_attention_score = Train(
    catch_reward=False, look_back=LOOK_BACK, noise=NOISE
)  # can also be used to test the agent that learned without as
test_catch_score = Train(
    attention_reward=False, look_back=LOOK_BACK, noise=NOISE
)  # can also be used to test the agent that learned without as
test_attention_score_disabled_a = Train(
    catch_reward=False, noise_removing=False, look_back=LOOK_BACK, noise=NOISE
)
test_attention_score_disabled_as = Train(
    catch_reward=False,
    attention_schema=False,
    look_back=LOOK_BACK,
    noise=NOISE,
)
test_catch_score_disabled_a = Train(
    attention_reward=False,
    noise_removing=False,
    look_back=LOOK_BACK,
    noise=NOISE,
)
test_catch_score_disabled_as = Train(
    attention_reward=False,
    attention_schema=False,
    look_back=LOOK_BACK,
    noise=NOISE,
)

train_agent_without_as = Train(
    attention_schema=False, look_back=LOOK_BACK, noise=NOISE
)
train_agent_without_as_no_ar = Train(
    attention_reward=False,
    attention_schema=False,
    look_back=LOOK_BACK,
    noise=NOISE,
)
test_agent_without_as_attention_score = Train(
    catch_reward=False,
    attention_schema=False,
    look_back=LOOK_BACK,
    noise=NOISE,
)
test_agent_without_as_catch_score = Train(
    attention_reward=False,
    attention_schema=False,
    look_back=LOOK_BACK,
    noise=NOISE,
)

utils.validate_py_environment(train_agent, episodes=5)
utils.validate_py_environment(train_agent_no_ar, episodes=5)
utils.validate_py_environment(test_attention_score, episodes=5)
utils.validate_py_environment(test_catch_score, episodes=5)
utils.validate_py_environment(test_attention_score_disabled_as, episodes=5)
utils.validate_py_environment(test_catch_score_disabled_as, episodes=5)
utils.validate_py_environment(train_agent_without_as, episodes=5)
utils.validate_py_environment(train_agent_without_as_no_ar, episodes=5)
utils.validate_py_environment(
    test_agent_without_as_attention_score, episodes=5
)
utils.validate_py_environment(test_agent_without_as_catch_score, episodes=5)


train_agent_env = tf_py_environment.TFPyEnvironment(train_agent)
train_agent_no_ar_env = tf_py_environment.TFPyEnvironment(train_agent_no_ar)
test_attention_score_env = tf_py_environment.TFPyEnvironment(
    test_attention_score
)
test_catch_score_env = tf_py_environment.TFPyEnvironment(test_catch_score)
test_attention_score_disabled_a_env = tf_py_environment.TFPyEnvironment(
    test_attention_score_disabled_a
)
test_attention_score_disabled_as_env = tf_py_environment.TFPyEnvironment(
    test_attention_score_disabled_as
)
test_catch_score_disabled_a_env = tf_py_environment.TFPyEnvironment(
    test_catch_score_disabled_a
)
test_catch_score_disabled_as_env = tf_py_environment.TFPyEnvironment(
    test_catch_score_disabled_as
)

train_agent_without_as_env = tf_py_environment.TFPyEnvironment(
    train_agent_without_as
)
train_agent_without_as_no_ar_env = tf_py_environment.TFPyEnvironment(
    train_agent_without_as_no_ar
)
test_agent_without_as_attention_score_env = tf_py_environment.TFPyEnvironment(
    test_agent_without_as_attention_score
)
test_agent_without_as_catch_score_env = tf_py_environment.TFPyEnvironment(
    test_agent_without_as_catch_score
)



class training_with_as:
    def __init__(
        self,
        units_per_layer: int,
        training_iters: int,
        epsilon_greedy: float = 0.2,
        learning_rate: float = 0.001,
        max_length: int = 10000,
        max_to_keep: int = 1,
        checkpoint_dir: str = None,
        checkpoint_name: str = None,
    ):
        self.units_per_layer = units_per_layer
        self.training_iters = training_iters
        self.epsilon_greedy = epsilon_greedy
        self.learning_rate = learning_rate
        self.max_length = max_length
        self.max_to_keep = max_to_keep
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_name = checkpoint_name
        
        Path(self.checkpoint_dir).mkdir(exist_ok=True)
        self.save_checkpoint_path = os.path.join(
            self.checkpoint_dir, self.checkpoint_name
        )
        Path(self.save_checkpoint_path).mkdir(exist_ok=True)

    def test_agent(self):
        q_net = q_network.QNetwork(
            train_agent_env.time_step_spec().observation,
            train_agent_env.action_spec(),
            fc_layer_params=(
                self.units_per_layer,
                self.units_per_layer,
                self.units_per_layer,
            ),
        )

        agent = dqn_agent.DqnAgent(
            time_step_spec=train_agent_env.time_step_spec(),
            action_spec=train_agent_env.action_spec(),
            q_network=q_net,
            epsilon_greedy=self.epsilon_greedy,
            optimizer=tf.optimizers.Adam(self.learning_rate),
        )

        global_step = tf.compat.v1.train.get_or_create_global_step()

        agent.initialize()

        replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
            agent.collect_data_spec,
            batch_size=train_agent_without_as_no_ar_env.batch_size,  # where does the batch size come from??
            max_length=self.max_length,
        )

        train_checkpointer = common.Checkpointer(
            ckpt_dir=self.save_checkpoint_path,
            max_to_keep=self.max_to_keep,
            agent=agent,
            policy=agent.policy,
            replay_buffer=replay_buffer,
            global_step=global_step,
        )

        train_checkpointer.initialize_or_restore()

        testing_iterations = TESTING_ITERS
        training_loss = []

        training_return = []
        attention_score = []
        catch_score = []
        attention_score_disabled_a = []
        attention_score_disabled_as = []
        catch_score_disabled_a = []
        catch_score_disabled_as = []

        
        collect_training_data(train_agent_env, agent, replay_buffer)
        training_loss.append(train_agent(agent, replay_buffer))
        training_return.append(
            compute_avg_return(
                train_agent_env, agent.policy, testing_iterations
            )
        )  # Compute average return for actual environment
        attention_score.append(
            compute_avg_return(
                test_attention_score_env, agent.policy, testing_iterations
            )
        )
        catch_score.append(
            compute_avg_return(
                test_catch_score_env, agent.policy, testing_iterations
            )
        )
        attention_score_disabled_as.append(
            compute_avg_return(
                test_attention_score_disabled_as_env,
                agent.policy,
                testing_iterations,
            )
        )
        attention_score_disabled_a.append(
            compute_avg_return(
                test_attention_score_disabled_a_env,
                agent.policy,
                testing_iterations,
            )
        )
        catch_score_disabled_as.append(
            compute_avg_return(
                test_catch_score_disabled_as_env,
                agent.policy,
                testing_iterations,
            )
        )
        catch_score_disabled_a.append(
            compute_avg_return(
                test_catch_score_disabled_a_env,
                agent.policy,
                testing_iterations,
            )
        )
    
        

        # Save results to dataframe
        results = pd.DataFrame()
        results["training_return"] = training_return
        results["attention_score"] = attention_score
        results["catch_score"] = catch_score
        results["attention_score_disabled_a"] = attention_score_disabled_a
        results["attention_score_disabled_as"] = attention_score_disabled_as
        results["catch_score_disabled_a"] = catch_score_disabled_a
        results["catch_score_disabled_as"] = catch_score_disabled_as
        results["loss"] = training_loss

        # Save results df to file
        results.sort_index(axis=1, inplace=True)

        df = pd.DataFrame(results)
        df.to_csv(
            "results_agent_with_as"
            + str(self.units_per_layer)
            + "units_"
            + str(self.training_iters)
            + "_epochs.csv"
        )



class training_with_as_no_ar:
    def __init__(
        self,
        units_per_layer: int,
        training_iters: int,
        epsilon_greedy: float = 0.2,
        learning_rate: float = 0.001,
        max_length: int = 10000,
        max_to_keep: int = 1,
        checkpoint_dir: str = None,
        checkpoint_name: str = None,
    ):
        self.units_per_layer = units_per_layer
        self.training_iters = training_iters
        self.epsilon_greedy = epsilon_greedy
        self.learning_rate = learning_rate
        self.max_length = max_length
        self.max_to_keep = max_to_keep
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_name = checkpoint_name
        Path(self.checkpoint_dir).mkdir(exist_ok=True)
        self.save_checkpoint_path = os.path.join(
            self.checkpoint_dir, self.checkpoint_name
        )
        Path(self.save_checkpoint_path).mkdir(exist_ok=True)

    def test_agent(self):
        q_net = q_network.QNetwork(
            train_agent_no_ar_env.time_step_spec().observation,
            train_agent_no_ar_env.action_spec(),
            fc_layer_params=(
                self.units_per_layer,
                self.units_per_layer,
                self.units_per_layer,
            ),
        )

        agent = dqn_agent.DqnAgent(
            time_step_spec=train_agent_no_ar_env.time_step_spec(),
            action_spec=train_agent_no_ar_env.action_spec(),
            q_network=q_net,
            epsilon_greedy=self.epsilon_greedy,
            optimizer=tf.optimizers.Adam(self.learning_rate),
        )

        global_step = tf.compat.v1.train.get_or_create_global_step()

        agent.initialize()

        replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
            agent.collect_data_spec,
            batch_size=train_agent_without_as_no_ar_env.batch_size,  # where does the batch size come from??
            max_length=self.max_length,
        )

        train_checkpointer = common.Checkpointer(
            ckpt_dir=self.save_checkpoint_path,
            max_to_keep=self.max_to_keep,
            agent=agent,
            policy=agent.policy,
            replay_buffer=replay_buffer,
            global_step=global_step,
        )
        train_checkpointer.initialize_or_restore()

      
        train_checkpointer.initialize_or_restore()
        global_step = tf.compat.v1.train.get_global_step()

        testing_iterations = TESTING_ITERS

        training_loss = []
        training_return = []
        attention_score = []
        catch_score = []
        attention_score_disabled_a = []
        attention_score_disabled_as = []
        catch_score_disabled_a = []
        catch_score_disabled_as = []

        collect_training_data(train_agent_no_ar_env, agent, replay_buffer)
        training_loss.append(train_agent(agent, replay_buffer))
        training_return.append(
            compute_avg_return(
                train_agent_no_ar_env, agent.policy, testing_iterations
            )
        )  # Compute average return for actual environment
        attention_score.append(
            compute_avg_return(
                test_attention_score_env, agent.policy, testing_iterations
            )
        )
        catch_score.append(
            compute_avg_return(
                test_catch_score_env, agent.policy, testing_iterations
            )
        )
        attention_score_disabled_as.append(
            compute_avg_return(
                test_attention_score_disabled_as_env,
                agent.policy,
                testing_iterations,
            )
        )
        attention_score_disabled_a.append(
            compute_avg_return(
                test_attention_score_disabled_a_env,
                agent.policy,
                testing_iterations,
            )
        )
        catch_score_disabled_as.append(
            compute_avg_return(
                test_catch_score_disabled_as_env,
                agent.policy,
                testing_iterations,
            )
        )
        catch_score_disabled_a.append(
            compute_avg_return(
                test_catch_score_disabled_a_env,
                agent.policy,
                testing_iterations,
            )
        )



        # Save results to dataframe
        results = pd.DataFrame()
        results["training_return"] = training_return
        results["attention_score"] = attention_score
        results["catch_score"] = catch_score
        results["attention_score_disabled_a"] = attention_score_disabled_a
        results["attention_score_disabled_as"] = attention_score_disabled_as
        results["catch_score_disabled_a"] = catch_score_disabled_a
        results["catch_score_disabled_as"] = catch_score_disabled_as
        results["loss"] = training_loss

        # Save results df to file
        results.sort_index(axis=1, inplace=True)

        df = pd.DataFrame(results)
        df.to_csv(
            self.save_checkpoint_path
            + str(self.units_per_layer)
            + "_units_"
            + str(self.training_iters)
            + "_epochs.csv"
        )


def collect_training_data(env, agent, replay_buffer):
    dynamic_step_driver.DynamicStepDriver(
        env=env,
        policy=agent.collect_policy,
        observers=[replay_buffer.add_batch],
        num_steps=1000,
    ).run()


def train_agent(agent, replay_buffer):
    dataset = replay_buffer.as_dataset(sample_batch_size=100, num_steps=2)

    iterator = iter(dataset)

    loss = None
    for _ in range(100):
        trajectories, _ = next(iterator)
        loss = agent.train(experience=trajectories)

    # print('Training loss: ', loss.loss.numpy())
    return loss.loss.numpy()


def compute_avg_return(environment, policy, num_episodes=10):
    total_return = 0.0
    for _ in range(num_episodes):
        time_step = environment.reset()
        episode_return = 0.0

        while not time_step.is_last():
            action_step = policy.action(time_step)
            time_step = environment.step(action_step.action)
            episode_return += time_step.reward
        total_return += episode_return

    avg_return = total_return / num_episodes
    # print('Average Return: {0}'.format(avg_return.numpy()[0]))
    return avg_return.numpy()[0]


"""
agent_with_as_no_ar = training_with_as_no_ar(
    units_per_layer=200,
    training_iters=1,
    checkpoint_dir="/share/klab/sthorat/lpi_tf_agents/checkpoint_dir/",
    checkpoint_name="as_no_ar_5000_ep_200_units",
)
agent_with_as_no_ar.test_agent()"""



agent_with_as = training_with_as(
    units_per_layer=500,
    training_iters=1,
    checkpoint_dir="/share/klab/sthorat/lpi_tf_agents/checkpoint_dir/",
    checkpoint_name="as_5000_ep_500_units",
)
agent_with_as.test_agent()
