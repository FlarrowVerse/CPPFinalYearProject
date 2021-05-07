import pickle
import os
import numpy as np
from global_vars import *

def calculate_ideal_affinity(data):
	Y = np.ndarray((N_STFT_SAMPLES, len(data)))
	print(Y)
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
			
		Y = calculate_ideal_affinity(speaker_data)



if __name__ == '__main__':

	generate_yy_t()
	# # arr = np.array([[1,2],[3,4]])
	# # pickle.dump(arr, open('sample.pkl', 'wb'))

	# ob = pickle.load(open(os.path.join(pickle_data_dir, 'mix', '25.pkl'), 'rb'))
	# print(ob.shape)
	# print(ob[0][0])

	# # print(pickle.load(open('sample.pkl', 'rb')))