import time
import numpy as np
import tensorflow as tf
import gym
import env
from model import VI_Block, VI_Untied_Block
from utils import fmt_row

# Parameters
tf.app.flags.DEFINE_float('lr',               0.001,                  'Learning rate for RMSProp')
tf.app.flags.DEFINE_integer('k',              10,                     'Number of value iterations')
tf.app.flags.DEFINE_integer('ch_i',           1,                      'Channels in input layer')
tf.app.flags.DEFINE_integer('ch_h',           150,                    'Channels in initial hidden layer')
tf.app.flags.DEFINE_integer('ch_q',           4,                     'Channels in q layer (~actions)')
tf.app.flags.DEFINE_integer('statebatchsize', 1,                     'Number of state inputs for each sample (real number, technically is k+1)')
tf.app.flags.DEFINE_boolean('untied_weights', False,                  'Untie weights of VI network')
# Misc.
tf.app.flags.DEFINE_integer('seed',           0,                      'Random seed for numpy')
tf.app.flags.DEFINE_integer('display_step',   1,                      'Print summary output every n epochs')
tf.app.flags.DEFINE_string('logdir',          '/tmp/vintf/',          'Directory to store tensorboard summary')
tf.app.flags.DEFINE_float('gamma', 0.98, "discount value")

config = tf.app.flags.FLAGS

np.random.seed(config.seed)

env = gym.make("GridWorld-v0")

grid_height = env.map_height
grid_width = env.map_width

# symbolic input image tensor where typically first channel is image, second is the reward prior
X  = tf.placeholder(tf.float32, name="X",  shape=[None, grid_height, grid_width, config.ch_i])
# symbolic input batches of vertical positions
S1 = tf.placeholder(tf.int32, name="S1", shape=[None, config.statebatchsize])
# symbolic input batches of horizontal positions
S2 = tf.placeholder(tf.int32, name="S2", shape=[None, config.statebatchsize])
sym_reward = tf.placeholder(tf.float32, name="r", shape=[None])
sym_action = tf.placeholder(tf.int32, name="action", shape=[None])

# Construct model (Value Iteration Network)
if config.untied_weights:
	logits, nn = VI_Untied_Block(X, S1, S2, config)
else:
	logits, nn = VI_Block(X, S1, S2, config)

chosen_action_log_output = tf.reduce_sum(tf.one_hot(sym_action, config.ch_q) * nn, axis=1)
cost_op = tf.reduce_sum(- sym_reward * tf.log(chosen_action_log_output))
optimizer = tf.train.RMSPropOptimizer(learning_rate=config.lr, epsilon=1e-6, centered=True).minimize(cost_op)

# Initializing the variables
init = tf.global_variables_initializer()
saver = tf.train.Saver()

def discount_rewards(rewards):
	discounted = np.zeros_like(rewards)
	running_add = 0
	for t in reversed(range(0, rewards.size)):
		running_add = running_add * config.gamma + rewards[t]
		discounted[t] = running_add
	return discounted


# Launch the graph
with tf.Session() as sess:
	for var in tf.trainable_variables():
		tf.summary.histogram(var.op.name, var)
	summary_op = tf.summary.merge_all()
	summary_writer = tf.summary.FileWriter(config.logdir, sess.graph)
	sess.run(init)

	last_100_episodes = np.zeros(100)

	for episode in range(10000):
		obs = env.reset().reshape(1, grid_height, grid_width)
		reward_prior = np.zeros_like(obs)
		observations = []
		states = []
		rewards = []
		actions = []
		length = 0
		done = False
		while not done:
			states.append(env.character_position)
			obs = obs.reshape(-1, grid_height, grid_width, 1)
			x_input = obs
			observations.append(obs)
			output = sess.run(nn, {
				X: x_input,
				S1: [[env.character_position[0]]],
				S2: [[env.character_position[1]]]
			})
			action = np.random.multinomial(1, output[0]).argmax()

			actions.append(action)
			obs, reward, done, _ = env.step(action)
			if episode % 100 == 0:
				env.render()
			rewards.append(reward)
			length += 1
		rewards = np.array(rewards)
		episode_reward = rewards.sum()

		last_100_episodes[episode % 100] = episode_reward
		states = np.array(states)
		observations = np.stack(observations).reshape(-1, grid_height, grid_width)
		x_input = observations.reshape(-1, grid_height, grid_width, 1)
		_, cost, _ = sess.run([optimizer, cost_op, summary_op], {
			X: x_input,
			S1: states[:, 0].reshape(-1, 1),
			S2: states[:, 1].reshape(-1, 1),
			sym_reward: discount_rewards(rewards),
			sym_action: actions
		})
		tf.summary.scalar("Cost", cost)

		print("cost: ", cost)
		mean_100 = last_100_episodes.mean()
		print('Average episode reward', mean_100)
		tf.summary.scalar("Mean 100 episode reward", mean_100)

