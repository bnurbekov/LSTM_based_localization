import tensorflow as tf
import numpy as np
#import tf.train, tf.contrib.rnn
import extract_data
import time
import math

DATA_FILE = "/media/atlancer/My Passport/Projects/summary.mat"
LOG_DIR = "/media/atlancer/My Passport/Projects/log_dir"

NUM_STEPS = 30
BATCH_SIZE = 20
NUM_HIDDEN = 500
NUM_LSTM_LAYERS = 2
INPUT_SIZE = None
OUTPUT_SIZE = 3
MAX_GRAD_NORM = 5
INITIALIZER_STD_DEV = 0.1

#Regularization:
KEEP_PROB = 0.35 #(Set to large percentage since the network needs to encode the map precisely)

LEARNING_RATE = 1
LEARNING_RATE_DECAY = 0.8
LEARNING_RATE_DECAY_START_EPOCH_I = 6


class Model:
	def __init__(self, is_training):
		self.lstm = tf.contrib.rnn.BasicLSTMCell(NUM_HIDDEN, forget_bias=0.0, state_is_tuple=True)

		if is_training and KEEP_PROB < 1:
			self.lstm = tf.contrib.rnn.DropoutWrapper(self.lstm, output_keep_prob=KEEP_PROB)

		self.stacked_lstm = tf.contrib.rnn.MultiRNNCell([self.lstm]*NUM_LSTM_LAYERS, state_is_tuple=True)

		self.initial_state = self.stacked_lstm.zero_state(BATCH_SIZE, dtype=tf.float32)

		self.epoch_data = tf.placeholder(tf.float32, shape=[NUM_STEPS, BATCH_SIZE, INPUT_SIZE])
		self.targets = tf.placeholder(tf.float32, shape=[NUM_STEPS, BATCH_SIZE, OUTPUT_SIZE])

		if is_training and KEEP_PROB < 1:
			self.targets = tf.nn.dropout(self.targets, keep_prob=KEEP_PROB)

		state = self.initial_state

		outputs = []
		out_weight = tf.get_variable("out_weight", [self.stacked_lstm.output_size, OUTPUT_SIZE], dtype=tf.float32)
		out_bias = tf.get_variable("out_bias", [OUTPUT_SIZE], dtype=tf.float32)
		with tf.variable_scope("RNN"):
			for step_i in range(NUM_STEPS):
				if step_i > 0:
					tf.get_variable_scope().reuse_variables()
				output, state = self.stacked_lstm(self.epoch_data[step_i, :, :], state)

				out_coordinates = tf.matmul(output, out_weight) + out_bias

				outputs.append(out_coordinates)

		self.difference = tf.subtract(self.targets, outputs)
		self.loss = tf.reduce_mean(tf.reduce_mean(tf.reduce_sum(tf.square(self.difference), axis=2), axis=1))

		self.final_state = state

		if not is_training:
			return

		self.learning_rate = tf.Variable(0.0, trainable=False)
		tvars = tf.trainable_variables()
		grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), MAX_GRAD_NORM)
		optimizer = tf.train.AdamOptimizer()
		self.train_op = optimizer.apply_gradients(zip(grads, tvars))

		self.new_learning_rate = tf.placeholder(tf.float32, shape=[])
		self.learning_rate_update = tf.assign(self.learning_rate, self.new_learning_rate)

	def assign_learning_rate(self, sess, new_value):
		sess.run(self.learning_rate_update, feed_dict={self.new_learning_rate:new_value})

def run(m, sess, data, targets, is_training, verbose=False):
	start_time = time.time()
	total_loss = 0.0
	iters = 0

	fetches = {"final_state":m.final_state, "loss":m.loss}
	if is_training:
		fetches["train_op"] = m.train_op
	else:
		fetches["diff"] = m.difference

	state = sess.run(m.initial_state)
	epoch_size = data.shape[0]

	for epoch_i in range(epoch_size):
		if is_training:
			learning_rate_decay = LEARNING_RATE_DECAY ** max(epoch_i - LEARNING_RATE_DECAY_START_EPOCH_I, 0)
			m.assign_learning_rate(sess, LEARNING_RATE * learning_rate_decay)

		feed_dict = {m.initial_state: state, m.epoch_data: data[epoch_i], m.targets: targets[epoch_i]}

		vals = sess.run(fetches, feed_dict)
		state = vals["final_state"]
		total_loss += vals["loss"]
		iters += NUM_STEPS

		if verbose:
			print("Iteration: %d, Speed: %.3f, Loss: %.3f" % (iters, iters*BATCH_SIZE/(time.time()-start_time), vals["loss"]))

		if not is_training and vals["loss"] > 0.005:
			print("Something is wrong")

	return total_loss/epoch_size

def main(_):
	data, targets = extract_data.get_data_from_file(DATA_FILE)

	print("Data shape:", data.shape)
	print("Targets shape:", targets.shape)

	prev = np.copy(targets[0])
	for i in range(targets.shape[0]):
		current = np.subtract(targets[i], prev)
		current[2] = math.atan2(math.sin(current[2]), math.cos(current[2]))
		prev = np.copy(targets[i])
		targets[i] = current

	print("New Data")

	data, targets = extract_data.slice_data(data, targets, BATCH_SIZE, NUM_STEPS)

	data = np.divide(data, data.max())

	print("Cut data shape:", data.shape)
	print("Cut targets shape:", targets.shape)

	data, targets = np.transpose(data, [0, 2, 1, 3]), np.transpose(targets, [0, 2, 1, 3])

	print("Transposed data shape:", data.shape)
	print("Transposed targets shape:", targets.shape)

	training_data, training_targets = data[data.shape[0]//10:, :, :, :], targets[targets.shape[0]//10:, :, :, :]
	valid_data, valid_targets = data[:data.shape[0]//10, :, :, :], targets[:targets.shape[0]//10, :, :, :]

	global INPUT_SIZE

	INPUT_SIZE = data.shape[-1]

	with tf.Graph().as_default():
		# Build model

		initializer = tf.truncated_normal_initializer(stddev=INITIALIZER_STD_DEV)

		with tf.name_scope("Train"):
			with tf.variable_scope("Model", reuse=None, initializer=initializer):
				train_model = Model(True)
		print("Done initializing training model!")

		with tf.name_scope("Valid"):
			with tf.variable_scope("Model", reuse=True, initializer=initializer):
				valid_model = Model(False)
		print("Done initializing valid model!")

		sv = tf.train.Supervisor(logdir=LOG_DIR)
		with sv.managed_session() as sess:
			av_training_loss = run(train_model, sess, training_data, training_targets, True, verbose=True)
			print("Training loss: %f", av_training_loss)
			av_valid_loss = run(valid_model, sess, valid_data, valid_targets, False, verbose=True)
			print("Valid loss: %f", av_valid_loss)

			sv.saver.save(sess, LOG_DIR, global_step=sv.global_step)

if __name__ == "__main__":
	tf.app.run()