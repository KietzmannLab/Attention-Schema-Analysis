from pathlib import Path
import os
from tf_agents.agents.dqn import dqn_agent
from tf_agents.agents.ppo import ppo_agent
from tf_agents.networks.actor_distribution_network import ActorDistributionNetwork
from tf_agents.networks.value_network import ValueNetwork
from tf_agents.drivers import dynamic_step_driver
from tf_agents.environments import tf_py_environment, utils
from tf_agents.networks import q_network
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.utils import common
import tensorflow as tf
from .emergent_environment import Train
import pandas as pd
import tf_agents
from .utils import collect_training_data, compute_avg_return, train_agent_loss

class emergent_training_with_as:
    def __init__(
        self,
        units_per_layer: int,
        training_iters: int,
        epsilon_greedy: float = 0.2,
        learning_rate: float = 0.001,
        max_length: int = 10000,
        max_to_keep: int = 1,
        noise: float = 0.5,
        look_back: int = 10,
        testing_iters: int = 50,
        attention_reward: bool = False,
        attention_schema:bool = False,
        catch_reward:bool = False,
        noise_removing: bool = False, 
        window_size: int = 3,
		discount_factor: float = 0.99,
        episodes: int = 5,
        checkpoint_dir: str = None,
        checkpoint_name: str = None,
    ):
        self.units_per_layer = units_per_layer
        self.training_iters = training_iters
        self.epsilon_greedy = epsilon_greedy
        self.learning_rate = learning_rate
        self.max_length = max_length
        self.max_to_keep = max_to_keep
        self.episodes = episodes
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_name = checkpoint_name
        self.noise = noise 
        self.look_back = look_back
        self.window_size = window_size
        self.discount_factor = discount_factor
        self.attention_reward = attention_reward
        self.attention_schema = attention_schema
        self.testing_iters = testing_iters
        self.catch_reward = catch_reward
        self.noise_removing = noise_removing

        Path(self.checkpoint_dir).mkdir(exist_ok=True)
        self.save_checkpoint_path = os.path.join(
            self.checkpoint_dir, self.checkpoint_name
        )
        Path(self.save_checkpoint_path).mkdir(exist_ok=True)
        
        self.train_agent = Train(
			attention_reward = self.attention_reward,
			catch_reward = self.catch_reward,
			noise_removing = self.noise_removing,
			attention_schema=self.attention_schema,
			window_size=self.window_size,
			discount_factor=self.discount_factor,
	 		look_back=self.look_back, 
			noise=self.noise
		)
        self.train_agent_env = tf_py_environment.TFPyEnvironment(self.train_agent)
        
        

        self.test_attention_score = Train(
			attention_reward = True,
			catch_reward = False,
			noise_removing = self.noise_removing,
			attention_schema=self.attention_schema,
			window_size=self.window_size,
			discount_factor=self.discount_factor,
	 		look_back=self.look_back, 
			noise=self.noise			)
        
        self.test_attention_score_env = tf_py_environment.TFPyEnvironment(
			self.test_attention_score
		)
		
        self.test_catch_score = Train(
			attention_reward = False,
			catch_reward = True,
			noise_removing = self.noise_removing,
			attention_schema=self.attention_schema,
			window_size=self.window_size,
			discount_factor=self.discount_factor,
	 		look_back=self.look_back, 
			noise=self.noise
				)
        self.test_catch_score_env = tf_py_environment.TFPyEnvironment(self.test_catch_score)

        self.test_attention_score_random_schema = Train(
			attention_reward = True,
			catch_reward = False,
			noise_removing = self.noise_removing,
			attention_schema=self.attention_schema,
			window_size=self.window_size,
			discount_factor=self.discount_factor,
	 		look_back=self.look_back, 
			noise=self.noise,
            random_schema_action=True			)
        
        self.test_attention_score_random_schema_env = tf_py_environment.TFPyEnvironment(
			self.test_attention_score_random_schema
		)
		
        self.test_catch_score_random_schema = Train(
			attention_reward = False,
			catch_reward = True,
			noise_removing = self.noise_removing,
			attention_schema=self.attention_schema,
			window_size=self.window_size,
			discount_factor=self.discount_factor,
	 		look_back=self.look_back, 
			noise=self.noise,
            random_schema_action = True
				)
        self.test_catch_score_random_schema_env = tf_py_environment.TFPyEnvironment(self.test_catch_score_random_schema)

        self.test_attention_score_disabled_as = Train(
			attention_reward = True,
			catch_reward = False,
			noise_removing = self.noise_removing,
			attention_schema = False,
			window_size=self.window_size,
			discount_factor=self.discount_factor,
	 		look_back=self.look_back, 
			noise=self.noise
			)
        self.test_attention_score_disabled_as_env = tf_py_environment.TFPyEnvironment(
			self.test_attention_score_disabled_as
		)
		 
        self.test_attention_score_disabled_a = Train(
			attention_reward = True,
			catch_reward = False,
			noise_removing = False,
			attention_schema=self.attention_schema,
			window_size=self.window_size,
			discount_factor=self.discount_factor,
	 		look_back=self.look_back, 
			noise=self.noise
		) 
        self.test_attention_score_disabled_a_env = tf_py_environment.TFPyEnvironment(
				self.test_attention_score_disabled_a
			)
        self.test_catch_score_disabled_as = Train(
			attention_reward = False,
			catch_reward = True,
			noise_removing = self.noise_removing,
			attention_schema=False,
			window_size=self.window_size,
			discount_factor=self.discount_factor,
	 		look_back=self.look_back, 
			noise=self.noise
			)
        self.test_catch_score_disabled_as_env = tf_py_environment.TFPyEnvironment(
			self.test_catch_score_disabled_as
		)
        self.test_catch_score_disabled_a = Train(
			attention_reward = False,
			catch_reward = True,
			noise_removing = False,
			attention_schema=self.attention_schema,
			window_size=self.window_size,
			discount_factor=self.discount_factor,
	 		look_back=self.look_back, 
			noise=self.noise
			)
        self.test_catch_score_disabled_a_env = tf_py_environment.TFPyEnvironment(
				self.test_catch_score_disabled_a
			)
        
        self._validate()

    def _validate(self):
        
            utils.validate_py_environment(self.train_agent, episodes=self.episodes)
            utils.validate_py_environment(self.test_attention_score, episodes=self.episodes)
            utils.validate_py_environment(self.test_catch_score, episodes=self.episodes)
            utils.validate_py_environment(self.test_attention_score_disabled_as, episodes=self.episodes)
            utils.validate_py_environment(self.test_catch_score_disabled_as, episodes=self.episodes)
        


    def start_train(self):
        q_net = q_network.QNetwork(
            self.train_agent_env.time_step_spec().observation,
            self.train_agent_env.action_spec(),
            fc_layer_params=(
                self.units_per_layer,
                self.units_per_layer,
                self.units_per_layer,
            ),
        )

        """global_step = tf.compat.v1.train.get_or_create_global_step()
        start_epsilon = self.epsilon_greedy
        n_of_steps = 100000
        end_epsilon = 0.0001
        epsilon = tf.compat.v1.train.polynomial_decay(
            start_epsilon,
            global_step,
            n_of_steps,
            end_learning_rate=end_epsilon)

        agent = dqn_agent.DqnAgent(
            time_step_spec=self.train_agent_env.time_step_spec(),
            action_spec=self.train_agent_env.action_spec(),
            q_network=q_net,
            epsilon_greedy=epsilon,
            optimizer=tf.optimizers.Adam(self.learning_rate),
            gamma=self.discount_factor
        )
        """
        #diagonal initialization for ppo! make sure to initialize final layer (policy output layer) with particularly small weights
        actor_net = ActorDistributionNetwork(
             input_tensor_spec=self.train_agent_env.observation_spec(),
             output_tensor_spec=self.train_agent_env.action_spec(),
             fc_layer_params=(self.units_per_layer,self.units_per_layer,self.units_per_layer, self.units_per_layer),
             kernel_initializer=tf.keras.initializers.RandomNormal(0,0.001)
             

        )
        value_net = ValueNetwork(
             input_tensor_spec=self.train_agent_env.observation_spec(),
             fc_layer_params=(self.units_per_layer,self.units_per_layer, self.units_per_layer),
             kernel_initializer=tf.keras.initializers.RandomNormal(0,0.001)

        )
        agent = ppo_agent.PPOAgent(
             time_step_spec= self.train_agent_env.time_step_spec(),
             action_spec=self.train_agent_env.action_spec(),
             actor_net=actor_net,
             value_net=value_net,
             optimizer=tf.keras.optimizers.Adam(self.learning_rate),#could be the issue? PPO Adam is better with non-default hparams by far!
             num_epochs=5,
             discount_factor=self.discount_factor,
             #entropy check
        )

        global_step = tf.compat.v1.train.get_or_create_global_step()

        agent.initialize()

        replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
            agent.collect_data_spec,
            batch_size=self.train_agent_env.batch_size,  # where does the batch size come from??
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

        testing_iterations = self.testing_iters

        results_update = {}
        results_dict = {}
        i = int(global_step/100)
        while i < self.training_iters:
            global_step = tf.compat.v1.train.get_or_create_global_step()
            i = int(global_step/100)
            collect_training_data(self.train_agent_env, agent, replay_buffer)
            print(f"Step {str(i)}")
            training_loss=train_agent_loss(agent, replay_buffer)
            training_return= compute_avg_return(
                    self.train_agent_env, agent.policy, testing_iterations
                )
              # Compute average return for actual environment
            attention_score=compute_avg_return(
                    self.test_attention_score_env, agent.policy, testing_iterations
                )
            
            catch_score=compute_avg_return(
                    self.test_catch_score_env, agent.policy, testing_iterations
                )
            
            attention_score_random_schema=compute_avg_return(
                    self.test_attention_score_random_schema_env, agent.policy, testing_iterations
                )
            
            catch_score_random_schema=compute_avg_return(
                    self.test_catch_score_random_schema_env, agent.policy, testing_iterations
                )
            
            attention_score_disabled_as=compute_avg_return(
                    self.test_attention_score_disabled_as_env,
                    agent.policy,
                    testing_iterations,
                )
            
            attention_score_disabled_a=compute_avg_return(
                    self.test_attention_score_disabled_a_env,
                    agent.policy,
                    testing_iterations,
                )
            
            catch_score_disabled_as=compute_avg_return(
                    self.test_catch_score_disabled_as_env,
                    agent.policy,
                    testing_iterations,
                )
            
            catch_score_disabled_a=compute_avg_return(
                    self.test_catch_score_disabled_a_env,
                    agent.policy,
                    testing_iterations,
                )
            
            train_checkpointer.save(global_step)

            print(global_step, i)

            path = Path(
				"./analysis/csvs/"
				+ "results_agent_with_"
				+ self.checkpoint_name
				+ ".csv")
    

            if path.is_file():
                results = pd.read_csv(path, index_col=0)
                
                results_update["training_return"] = training_return
                results_update["attention_score"] = attention_score
                results_update["catch_score"] = catch_score
                results_update["attention_score_random_schema"] = attention_score_random_schema
                results_update["catch_score_random_schema"] = catch_score_random_schema
                results_update["attention_score_disabled_a"] = attention_score_disabled_a
                results_update["attention_score_disabled_as"] = attention_score_disabled_as
                results_update["catch_score_disabled_a"] = catch_score_disabled_a
                results_update["catch_score_disabled_as"] = catch_score_disabled_as
                results_update["loss"] = training_loss
                results_update_dict = pd.DataFrame(results_update, index=[i])
                results = pd.concat([results, results_update_dict])
                

            else:
                
                results_dict["training_return"] = training_return
                results_dict["attention_score"] = attention_score
                results_dict["catch_score"] = catch_score
                results_dict["attention_score_random_schema"] = attention_score_random_schema
                results_dict["catch_score_random_schema"] = catch_score_random_schema
                results_dict["attention_score_disabled_a"] = attention_score_disabled_a
                results_dict["attention_score_disabled_as"] = attention_score_disabled_as
                results_dict["catch_score_disabled_a"] = catch_score_disabled_a
                results_dict["catch_score_disabled_as"] = catch_score_disabled_as
                results_dict["loss"] = training_loss
                results = pd.DataFrame(results_dict, index=[i]) 

            # Save results df to file
            results.to_csv(path, index=True, index_label="epoch")

        tf_agents.policies.PolicySaver(agent.collect_policy).save(
			"policy/" + "policy_with_"
			+ self.checkpoint_name
		)

        