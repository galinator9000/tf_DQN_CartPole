#! -*- coding: UTF-8 -*-

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import gym, random, time
from collections import deque

weight_path = "model/dqn_cartpole_weights"
load_weights = True
skip_training = True

# Parameters.
alpha = 0.001		# Learning rate.
gamma = 0.95		# Future reward discount rate.

# Exploration / Exploitation rate.
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.995

batch_size = 64
max_episode = 250
max_timestep = 100.0
memory_max_size = 1000000
last_reward_record_count = 100

max_timestep_increment_after_episode = 100
max_timestep_increment = 1

env = gym.make("CartPole-v0")
observation = env.reset()

obvSpace_dim = env.observation_space.shape[0]
actSpace_dim = env.action_space.n

# Define Neural Network.
sess = tf.Session()

# Neural network config.
layer_cfg = [
	[32,			tf.nn.relu],
	[32,			tf.nn.relu],
	[actSpace_dim,	None]
]

# Agent class.
class DQNAgent:
	def __init__(self, name):
		with tf.variable_scope(name):
			# Create dense layers and store them in a list.
			layers = [
				tf.layers.Dense(
					units=layer_cfg[l][0],
					activation=layer_cfg[l][1],
					kernel_initializer=(
						lambda shape, dtype, partition_info: tf.Variable(tf.random_uniform(shape, -1.0, 1.0, dtype=dtype))
					)
				)
				for l in range(len(layer_cfg))
			]
			
			# Feed forward given matrix to the model.
			def feedForward(x):
				for layer in layers:
					x = layer(x)
				return x

			# Inputs.
			self.tf_state = tf.placeholder(tf.float32, shape=(None, obvSpace_dim))
			self.tf_action = tf.placeholder(tf.float32, shape=(None, actSpace_dim))

			# Feed-forward.
			self.tf_output = feedForward(self.tf_state)
			self.tf_q_pred = tf.reduce_sum(
				tf.multiply(self.tf_output, self.tf_action),
				axis=1
			)

			# Feed-backward.
			self.tf_q_tar = tf.placeholder(tf.float32, shape=(None))

			self.tf_loss = tf.reduce_mean(tf.square(self.tf_q_tar - self.tf_q_pred))
			self.tf_train = tf.train.GradientDescentOptimizer(alpha).minimize(self.tf_loss)

# Copies all trainable parameters from model mName to fName.
# This function gonna be called at every end of the episode.
# mName: Model to be trained.
# fName: Fixed model.
def update_parameters(mName, fName):
	updateOp = []
	for mW, fW in zip(
			tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, mName),
			tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, fName)
		):
		updateOp.append(fW.assign(mW))
	sess.run(updateOp)

model = DQNAgent("model")				# Agent's DQN model.
fixed_model = DQNAgent("fixed_model")	# Fixed weights model, only used for predicting 'next_state'. Q(s', a')

sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()

# Try to load weights.
if load_weights:
	try:
		saver.restore(sess, weight_path)
		print("Weights loaded.")

		if skip_training:
			print("Skipping training.")
	except:
		skip_training = False
		print("Weights couldn't loaded.")
else:
	skip_training = False

# Our agent's memory.
# Going to store experiences in [State, Action, Reward, NextState] form.
Memory = deque(maxlen=memory_max_size)

# Training.
plot_reward_record = []
reward_record = []
for episode in range(max_episode):
	if skip_training:
		break
	state = env.reset()
	episode_reward = 0

	for timestep in range(int(max_timestep)):
		# Comment this if you don't want to see the simulation while agent learns. (Slows training!)
		# env.render()

		# Explore / Exploit decision.
		# Based on epsilon value which is between 0 and 1.
		if random.random() < epsilon:
			# Explore, act randomly.
			action = env.action_space.sample()
		else:
			# Exploit, act the action which gives most reward.
			action = np.argmax(
				sess.run(
					model.tf_output,
					feed_dict={
						model.tf_state:np.expand_dims(state, axis=0)
					}
				)[0]
			)

		# Apply action on simulation.
		next_state, reward, done, info = env.step(action)

		# Penalize score if agent drops the pole (done = True) and it's not the end of the episode.
		if done and (timestep+1 < int(max_timestep)):
			reward = -10

		episode_reward += int(reward)

		# Store experience in memory. Only do this if it's not the end step.
		if not done:
			Memory.append(
				[state, action, reward, next_state]
			)

		state = next_state

		# Experience replay (training).
		# Waits until experiences accumulate much as batch size.
		if len(Memory) > batch_size:
			mini_batch = random.sample(Memory, batch_size)
			
			b_actions = np.array([np.array(b[1]) for b in mini_batch])
			b_actions_o = np.eye(actSpace_dim)[b_actions].astype(np.float32)
			b_states = np.array([np.array(b[0]) for b in mini_batch])

			# Get fixed model's output. And train the other model with this output.
			b_next_states = np.array([np.array(b[3]) for b in mini_batch])
			b_rewards = np.array([np.array(b[2]) for b in mini_batch])

			# Get the next state's most rewarding action for each experience.
			b_q_tars = b_rewards + gamma * np.max(
				sess.run(
					fixed_model.tf_output,
					feed_dict={
						fixed_model.tf_state:b_next_states
					}
				),
				axis=1
			)

			feed = {
				model.tf_action:b_actions_o,
				model.tf_state:b_states,
				model.tf_q_tar:b_q_tars
			}

			# Finally, train the model!
			sess.run(
				model.tf_train,
				feed_dict=feed
			)

		if epsilon > epsilon_min:
			epsilon *= epsilon_decay

		if done or (timestep+1 == int(max_timestep)):
			if len(reward_record) > last_reward_record_count:
				reward_record = reward_record[-last_reward_record_count:]

			try:
				avgRew = sum(reward_record) / float(len(reward_record))
			except:
				avgRew = 0.0

			if episode > max_timestep_increment_after_episode:
				max_timestep += max_timestep_increment

			print(
				"Ep %s, Epsilon %s, Ep Reward %s, Loss %s, Max Step %s, Avg reward %s" % (
					episode,
					"{0:.2f}".format(epsilon),
					episode_reward,
					(sess.run(
						model.tf_loss,
						feed_dict=feed
					) if len(Memory) > batch_size else None),
					"{0:.2f}".format(max_timestep),
					"{0:.2f}".format(avgRew)
				)
			)

			# Copy trained parameters to fixed model every end of the episode.
			update_parameters("model", "fixed_model")

			reward_record.append(episode_reward)
			plot_reward_record.append(episode_reward)
			break

# Plot reward for each episode.
if not skip_training:
	saver.save(sess, weight_path)
	plt.plot(range(len(plot_reward_record)), plot_reward_record)
	plt.show()

# Testing!
state = env.reset()
while True:
	env.render()

	# Selecting the best action for current state.
	action = np.argmax(
		sess.run(
			model.tf_output,
			feed_dict={
				model.tf_state:np.expand_dims(state, axis=0)
			}
		)[0]
	)

	state, reward, done, info = env.step(action)
	
	if done:
		state = env.reset()