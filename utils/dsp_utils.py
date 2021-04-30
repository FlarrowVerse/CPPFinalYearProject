import numpy as np
import librosa
import pickle
import os
import matplotlib.pyplot as plt

from global_vars import *

def stft(filename):
	audio, sr = librosa.load(filename)

	# print('Sampling Rate:',sr)
	# print('Audio shape:',audio.shape, audio.dtype)

	freq_domain_data = librosa.stft(audio, hop_length=HOP_SIZE, n_fft=FRAME_SIZE)

	# print('STFT shape:',freq_domain_data.shape, freq_domain_data.dtype)
	
	# plot_stft(freq_domain_data, sr)
	return freq_domain_data

def plot_stft(freq_domain_data, sr):
	print('Frequencies: ', librosa.fft_frequencies(sr=sr, n_fft=FRAME_SIZE).shape)
	for freq in range(librosa.fft_frequencies(sr=sr, n_fft=FRAME_SIZE).shape[0]):
		plt.plot(librosa.fft_frequencies(sr=sr, n_fft=FRAME_SIZE), np.abs(freq_domain_data[:, freq]))
		plt.show()

		choice = input('Continue?(Y/N):')
		if choice != 'y' and choice != 'Y':
			break

def pickle_dump(data, dirname):
	index = len(os.listdir(os.path.join(pickle_data_dir, dirname))) + 1	
	pickle_file = open(os.path.join(pickle_data_dir, dirname, str(index)+'.pkl'), 'wb')
	pickle.dump(data, pickle_file)
	pickle_file.close()

if __name__ == '__main__':
	stft(os.path.join(input('Base dir:'), input('Filename:')))