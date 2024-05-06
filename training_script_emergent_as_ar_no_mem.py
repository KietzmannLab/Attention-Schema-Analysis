from cookie_attention_tracking import emergent_training_with_as


def main():

	units_per_layer = 1000 #change units here
	training_iters = 60000 #100000
	epsilon_greedy = 0.5
	learning_rate = 0.00001
	max_length = 25000
	max_to_keep = 1
	noise = 0.25 # change noise here
	look_back = 1
	testing_iters = 50
	window_size=3
	discount_factor=0.9
	attention_reward = True
	attention_schema = True
	catch_reward = True
	noise_removing = True
	checkpoint_dir="/share/klab/sthorat/lpi_tf_agents/checkpoint_directory/"
	checkpoint_name = "noise_" + str(noise) + "_random_emergence_ar_no_mem_" + "ppo_3_act_" + str(learning_rate) +"_lr_"+ str(units_per_layer) + "_units_" + str(window_size)+ "_windowsize_"+ str(discount_factor)+ "_discount_factor"
	#checkpoint_name = "emergence_ar_no_mem_ppo_3_act_1e-05_lr_1000_units_3_windowsize_0.9_discount_factor"
	
	agent_with_as = emergent_training_with_as(
	units_per_layer=units_per_layer,
	training_iters=training_iters,
	epsilon_greedy = epsilon_greedy,
	learning_rate = learning_rate,
	max_length = max_length,
	max_to_keep = max_to_keep,
	noise = noise,
	look_back = look_back,
	testing_iters = testing_iters,
	discount_factor=discount_factor,
	window_size=window_size,
	attention_reward = attention_reward,
	attention_schema = attention_schema,
	catch_reward = catch_reward,
	noise_removing = noise_removing,
	checkpoint_dir=checkpoint_dir,
	checkpoint_name=checkpoint_name,
)
	agent_with_as.start_train()

if __name__=='__main__':

	main()
