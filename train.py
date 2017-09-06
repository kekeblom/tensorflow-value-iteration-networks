import time
import numpy as np
import tensorflow as tf
import gym
import env
from data  import process_gridworld_data
from model import VI_Block, VI_Untied_Block
from utils import fmt_row

# Parameters
tf.app.flags.DEFINE_float('lr',               0.001,                  'Learning rate for RMSProp')
tf.app.flags.DEFINE_integer('epochs',         30,                     'Maximum epochs to train for')
tf.app.flags.DEFINE_integer('k',              10,                     'Number of value iterations')
tf.app.flags.DEFINE_integer('ch_i',           2,                      'Channels in input layer')
tf.app.flags.DEFINE_integer('ch_h',           150,                    'Channels in initial hidden layer')
tf.app.flags.DEFINE_integer('ch_q',           4,                     'Channels in q layer (~actions)')
tf.app.flags.DEFINE_integer('batchsize',      12,                     'Batch size')
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
sym_gamma = tf.placeholder(tf.float32, name="gamma", shape=[None])
sym_action = tf.placeholder(tf.int32, name="action", shape=[None])

# Construct model (Value Iteration Network)
if config.untied_weights:
	logits, nn = VI_Untied_Block(X, S1, S2, config)
else:
	logits, nn = VI_Block(X, S1, S2, config)

chosen_action_logit = tf.reduce_sum(tf.one_hot(sym_action, config.ch_q) * logits, axis=1)
cost_op = tf.reduce_sum(- sym_reward * sym_gamma * chosen_action_logit)
optimizer = tf.train.RMSPropOptimizer(learning_rate=config.lr, epsilon=1e-3, centered=True).minimize(cost_op)

# Initializing the variables
init = tf.global_variables_initializer()
saver = tf.train.Saver()

# Launch the graph
with tf.Session() as sess:
	for var in tf.trainable_variables():
		tf.summary.histogram(var.op.name, var)
	summary_op = tf.summary.merge_all()
	summary_writer = tf.summary.FileWriter(config.logdir, sess.graph)
	sess.run(init)

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
			obs = obs.reshape(1, grid_height, grid_width)
			x_input = np.stack([reward_prior, obs], axis=3)
			observations.append(obs)
			output = sess.run(nn, {
				X: x_input,
				S1: [[env.character_position[0]]],
				S2: [[env.character_position[1]]]
			})
			action = output.argmax()
			actions.append(action)
			obs, reward, done, _ = env.step(action)
			rewards.append(reward)
			length += 1

		gamma = np.ones(length) * config.gamma
		for i in range(length):
			gamma[i] = gamma[i] ** i

		states = np.array(states)
		observations = np.stack(observations).reshape(-1, grid_height, grid_width)
		x_input = np.stack([np.zeros_like(observations), observations], axis=3)
		_, cost, _ = sess.run([optimizer, cost_op, summary_op], {
			X: x_input,
			S1: states[:, 0].reshape(-1, 1),
			S2: states[:, 1].reshape(-1, 1),
			sym_gamma: gamma,
			sym_reward: rewards,
			sym_action: actions
		})
		tf.summary.scalar("Cost", cost)

		print("cost: ", cost)
		print("Episode reward:", sum(rewards))

