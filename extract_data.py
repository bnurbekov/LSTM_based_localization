import scipy.io
import numpy as np
import os, os.path

# Processes all .mat files in the current folder
def get_data_from_folder(data_folder):
	data = []
	targets = []

	assert os.path.exists(data_folder)
	assert os.path.isdir(data_folder)

	for file in os.listdir(data_folder):
		if os.path.splitext(file)[1] != ".mat":
			continue
		mat = scipy.io.loadmat(os.path.join(data_folder, file), struct_as_record=False, squeeze_me=True)
		init_record_i = mat["summary"].iRecIBasemapLoad + 1
		print("Initial Record Ind: %d" % init_record_i)

		pose = mat["summary"].pose

		target = []
		target.append(pose.xS_m[init_record_i:])
		target.append(pose.yS_m[init_record_i:])
		target.append(pose.hS_m[init_record_i:])
		target_t = np.array(target, dtype=np.float32).transpose()
		targets.append(np.subtract(target_t, target_t[0]))

		data.append(mat["summary"].scan.data[init_record_i:])

	assert len(data) != 0
	assert len(targets) != 0

	return np.array(data, dtype=np.float32), np.array(targets, dtype=np.float32)

# Processes all .mat files in the current folder
def get_data_from_file(filename):
	assert os.path.exists(filename)
	assert os.path.splitext(filename)[1] == ".mat"

	mat = scipy.io.loadmat(filename, struct_as_record=False, squeeze_me=True)
	init_record_i = mat["summary"].iRecIBasemapLoad + 1
	print("Initial Record Ind: %d" % init_record_i)

	pose = mat["summary"].pose

	target = []
	target.append(pose.xS_m[init_record_i:])
	target.append(pose.yS_m[init_record_i:])
	target.append(pose.hS_m[init_record_i:])
	target_t = np.array(target, dtype=np.float32).transpose()

	data = mat["summary"].scan.data[init_record_i:]

	return np.array(data, dtype=np.float32), np.array(target_t, dtype=np.float32)

def slice_data(data, targets, batch_size, num_steps):
	assert targets.shape[0] == data.shape[0]
	data_length = data.shape[0]

	batch_partition_len = data_length // batch_size

	data_x = np.zeros([batch_size, batch_partition_len] + (list(data.shape[1:]) if len(data.shape) > 1 else []), dtype=np.float32)
	data_y = np.zeros([batch_size, batch_partition_len] + (list(targets.shape[1:]) if len(targets.shape) > 1 else []), dtype=np.float32)

	for i in range(batch_size):
		data_x[i] = data[i*batch_partition_len:(i+1)*batch_partition_len, :]
		data_y[i] = targets[i*batch_partition_len:(i+1)*batch_partition_len, :]

	epoch_size = batch_partition_len // num_steps

	final_x = np.zeros([epoch_size, batch_size, num_steps] + (list(data.shape[1:]) if len(data.shape) > 1 else []), dtype=np.float32)
	final_y = np.zeros([epoch_size, batch_size, num_steps] + (list(targets.shape[1:]) if len(targets.shape) > 1 else []), dtype=np.float32)
	for i in range(epoch_size):
		final_x[i] = data_x[:,i*num_steps:(i+1)*num_steps,:]
		final_y[i] = data_y[:,i*num_steps:(i+1)*num_steps,:]

	return final_x, final_y