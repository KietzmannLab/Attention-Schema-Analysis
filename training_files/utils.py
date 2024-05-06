from tf_agents.drivers import dynamic_step_driver

def collect_training_data(env, agent, replay_buffer):
	dynamic_step_driver.DynamicStepDriver(
		env=env,
		policy=agent.collect_policy,
		observers=[replay_buffer.add_batch],
		num_steps=1000,
	).run()
	
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

def train_agent_loss(agent, replay_buffer):
	dataset = replay_buffer.as_dataset(sample_batch_size=100, num_steps=2)

	iterator = iter(dataset)

	loss = None
	for _ in range(100):
		trajectories, _ = next(iterator)
		loss = agent.train(experience=trajectories)

	# print('Training loss: ', loss.loss.numpy())
	return loss.loss.numpy()