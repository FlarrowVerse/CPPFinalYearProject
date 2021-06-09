import pickle
import os
import numpy as np
from global_vars import *

def get_magnitudes(freq_data, total_samples):
	mags = np.zeros(total_samples)
	print(mags.shape)
	for i in range(len(freq_data)):
		for j in range(len(freq_data[0])):
			mags[i * j] = np.abs(freq_data[i, j])
	return mags

def calculate_ideal_affinity(data):
	total_samples = len(data[0]) * len(data[0][0])
	# print(total_samples)
	Y = np.ndarray((total_samples, 2))
	# print(Y.shape)

	speaker_1 = get_magnitudes(data[0], total_samples)
	speaker_2 = get_magnitudes(data[1], total_samples)

	for i in range(total_samples):
		if speaker_1[i] > speaker_2[i]:
			Y[i, 0] = 1
			Y[i, 1] = 0
		else:
			Y[i, 0] = 0
			Y[i, 1] = 1

	return Y

'''
Load the speaker files and compute YY^T
'''
def generate_yy_t():
	speaker_files = {}
	files = 0
	for dirs in os.listdir(os.path.join(pickle_data_dir)):
		if dirs.startswith('speaker'):
			speaker_files[dirs] = []
			for filename in os.listdir(os.path.join(pickle_data_dir, dirs)):
				speaker_files[dirs].append(filename)
				files += 1
	
	files = int(files / len(speaker_files))

	print(speaker_files, files)

	for i in range(files):
		speaker_data = []
		for speaker in speaker_files:
			speaker_data.append(pickle.load(open(
				os.path.join(
					pickle_data_dir, 
					speaker, 
					speaker_files[speaker][i]
					), 'rb')))
			print(os.path.join(pickle_data_dir, speaker, speaker_files[speaker][i]))
			
		Y = calculate_ideal_affinity(speaker_data)
		print(Y.shape)
		
		for i in range(0, len(Y), 300):
			Y_batch = Y[max(0, i-300):i, :]
			Yt_batch = np.transpose(Y_batch)
			# print(Y_batch.shape, 'x', Yt_batch.shape)

			YYt_batch = np.dot(Y_batch, Yt_batch)
			# print(YYt_batch.shape)



if __name__ == '__main__':

	generate_yy_t()
	