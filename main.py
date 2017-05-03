import tensorflow as tf
import numpy as np
#import tf.train, tf.contrib.rnn
import extract_data
import time
import math
import socketserver
import json
import csv

DATA_FILE = "/media/atlancer/My Passport/Projects/summary.mat"
LOG_DIR = "/media/atlancer/My Passport/Projects/log_dir"

NUM_STEPS = 30
BATCH_SIZE = 20
NUM_HIDDEN = 500
NUM_LSTM_LAYERS = 2
INPUT_SIZE = 1081 # Size of the scan readings array
OUTPUT_SIZE = 3
MAX_GRAD_NORM = 5
INITIALIZER_STD_DEV = 0.1

#Regularization:
KEEP_PROB = 0.35 #(Set to large percentage since the network needs to encode the map precisely)

LEARNING_RATE = 1
LEARNING_RATE_DECAY = 0.8
LEARNING_RATE_DECAY_START_EPOCH_I = 6

ONLINE_MODE = False
#if ONLINE_MODE is True, then set the following:
HOST = '192.168.1.65'
PORT = 7779

class Model:
	def __init__(self, batch_size, num_steps, is_training=False):
		self.lstm = tf.contrib.rnn.BasicLSTMCell(NUM_HIDDEN, forget_bias=0.0, state_is_tuple=True)

		if is_training and KEEP_PROB < 1:
			self.lstm = tf.contrib.rnn.DropoutWrapper(self.lstm, output_keep_prob=KEEP_PROB)

		self.stacked_lstm = tf.contrib.rnn.MultiRNNCell([self.lstm]*NUM_LSTM_LAYERS, state_is_tuple=True)

		self.initial_state = self.stacked_lstm.zero_state(batch_size, dtype=tf.float32)

		self.epoch_data = tf.placeholder(tf.float32, shape=[num_steps, batch_size, INPUT_SIZE])
		self.targets = tf.placeholder(tf.float32, shape=[num_steps, batch_size, OUTPUT_SIZE])

		if is_training and KEEP_PROB < 1:
			self.targets = tf.nn.dropout(self.targets, keep_prob=KEEP_PROB)

		state = self.initial_state

		outputs = []
		out_weight = tf.get_variable("out_weight", [self.stacked_lstm.output_size, OUTPUT_SIZE], dtype=tf.float32)
		out_bias = tf.get_variable("out_bias", [OUTPUT_SIZE], dtype=tf.float32)
		with tf.variable_scope("RNN"):
			for step_i in range(num_steps):
				if step_i > 0:
					tf.get_variable_scope().reuse_variables()
				output, state = self.stacked_lstm(self.epoch_data[step_i, :, :], state)

				out_coordinates = tf.matmul(output, out_weight) + out_bias

				outputs.append(out_coordinates)

		self.outputs = outputs

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

	def run(self, sess, data, targets, is_training, verbose=False, record_outputs=False, init_state=None):
		start_time = time.time()
		total_loss = 0.0
		iters = 0

		## CSV 
		res_file = open("results.csv" if is_training else "valid.csv", "w")
		writer = csv.writer(res_file, delimiter=",")
		##

		fetches = {"final_state":self.final_state, "loss":self.loss}
		if is_training:
			fetches["train_op"] = self.train_op
		else:
			fetches["diff"] = self.difference
			if record_outputs:
				fetches["outputs"] = self.outputs

		if init_state is None:
			state = sess.run(self.initial_state)
		else:
			state = init_state
		epoch_size = data.shape[0]
		epoch_outputs = []

		for epoch_i in range(epoch_size):
			if is_training:
				learning_rate_decay = LEARNING_RATE_DECAY ** max(epoch_i - LEARNING_RATE_DECAY_START_EPOCH_I, 0)
				self.assign_learning_rate(sess, LEARNING_RATE * learning_rate_decay)

			feed_dict = {self.initial_state: state, self.epoch_data: data[epoch_i]}

			if targets is not None:
				feed_dict[self.targets] = targets[epoch_i]

			vals = sess.run(fetches, feed_dict)
			state = vals["final_state"]
			total_loss += vals["loss"]
			iters += NUM_STEPS

			if verbose:
				if targets is not None and not is_training:
					av_perc =np.mean(np.mean(np.mean(np.fabs(np.divide(vals["diff"], targets[epoch_i])), axis=2), axis=1))
				else:
					av_perc = -1
				print("Iteration: %d, Speed: %.3f, Loss: %.3f, Av Perc: %f" % (iters, iters*BATCH_SIZE/(time.time()-start_time), vals["loss"], av_perc))

				writer.writerow([iters, vals["loss"]])

			if not is_training and vals["loss"] > 0.005:
				print("Something is wrong. Loss: ", vals["loss"])

			if record_outputs:
				epoch_outputs.append(vals["outputs"])

		res_file.close()

		return epoch_outputs, state

def MakeHandlerClassFromArgv(model, session):
	class MyTCPHandler(socketserver.StreamRequestHandler, object):
		def __init__(self, *args, **kwargs):
			self.model = model
			self.session = session
			self.prev_state = None
			super(MyTCPHandler, self).__init__(*args, **kwargs)

		"""
		The request handler class for our server.

		It is instantiated once per connection to the server, and must
		override the handle() method to implement communication to the
		client.
		"""

		def handle(self):
			# self.request is the TCP socket connected to the client
			while True:
				self.data = self.rfile.readline().strip()
				if not self.data:
					break
				#print("{} wrote:".format(self.client_address[0]))
				#print(self.data)

				arrData = np.array(json.loads(self.data.decode("utf-8")))

				model_data = np.expand_dims(np.expand_dims(np.expand_dims(arrData, axis=0), axis=0), axis=0)

				epoch_data, self.prev_state = self.model.run(session, model_data, np.array([[[[0, 0, 0]]]]), False, verbose=True, record_outputs=True, init_state=self.prev_state)
				# just send back the same data, but upper-cased
				self.request.sendall(bytes(json.dumps(epoch_data[0][0][0].tolist())+"\r\n", "utf-8"))

	return MyTCPHandler

def main(_):
	if not ONLINE_MODE:
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

		training_data, training_targets = data[data.shape[0]//90:, :, :, :], targets[targets.shape[0]//90:, :, :, :]
		valid_data, valid_targets = data[:data.shape[0]//90, :, :, :], targets[:targets.shape[0]//90, :, :, :]

	with tf.Graph().as_default():
		# Build model

		initializer = tf.truncated_normal_initializer(stddev=INITIALIZER_STD_DEV)

		with tf.name_scope("Train"):
			with tf.variable_scope("Model", reuse=None, initializer=initializer):
				train_model = Model(BATCH_SIZE, NUM_STEPS, is_training=True)
		print("Done initializing training model!")

		with tf.name_scope("Valid"):
			with tf.variable_scope("Model", reuse=True, initializer=initializer):
				valid_model = Model(BATCH_SIZE, NUM_STEPS, is_training=False)
		print("Done initializing valid model!")

		with tf.name_scope("Online"):
			with tf.variable_scope("Model", reuse=True, initializer=initializer):
				online_model = Model(1, 1, is_training=False)
		print("Done initializing online model!")

		init_op = tf.global_variables_initializer()

		sv = tf.train.Saver()
		with tf.Session() as sess:
			if not ONLINE_MODE:
				sess.run(init_op)

				train_model.run(sess, training_data, training_targets, True, verbose=True)
				valid_model.run(sess, valid_data, valid_targets, False, verbose=True)

				print("Saving Model...")
				sv.save(sess, LOG_DIR, global_step=0)
				print("Saved!")
			else:
				ckpt = tf.train.get_checkpoint_state(LOG_DIR[:LOG_DIR.rfind("/")])
				if ckpt and ckpt.model_checkpoint_path:
					print(ckpt.model_checkpoint_path)
					sv.restore(sess, ckpt.model_checkpoint_path)
				else:
					raise Exception("No checkpoint found!")
				_, state = valid_model.run(sess, valid_data, valid_targets, False, verbose=True)
				socketserver.TCPServer.allow_reuse_address = True
				server = socketserver.TCPServer((HOST, PORT), MakeHandlerClassFromArgv(online_model, sess))
				print("Running server!")
				server.serve_forever()

if __name__ == "__main__":
	tf.app.run()