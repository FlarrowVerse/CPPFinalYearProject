import librosa
import soundfile as sf
import numpy as np
import os
import matplotlib.pyplot as plt

import random

from global_vars import *

import scipy.signal as signal
from dsp_utils import *


def set_sample_specs(speaker_dict):
	SAMPLE_LEN = 0
	SAMPLE_RATE = 0

	for speaker in speaker_dict:
		for file in speaker_dict[speaker]:
			speaker_file = os.path.join(raw_data_dir, speaker, file)
			speaker_data, sr = librosa.load(speaker_file)
			if SAMPLE_LEN == 0:
				SAMPLE_LEN = len(speaker_data)
			else:
				SAMPLE_LEN = min(SAMPLE_LEN, len(speaker_data))

			SAMPLE_RATE = sr

	return SAMPLE_LEN, SAMPLE_RATE


def mix(speaker_dict, file_nums, SAMPLE_LEN, SAMPLE_RATE):
	print('Mixture Data for: ')
	mix_data = np.ndarray((0,)) # mixed data

	for i in range(len(file_nums)):

		# loading actual audio data
		speaker_file = os.path.join(
			raw_data_dir, 
			'speaker_'+str(i+1), 
			speaker_dict['speaker_'+str(i+1)][file_nums[i]]
			)
		speaker_data, sr = librosa.load(speaker_file)
		print(speaker_file)

		# performing stft on speaker file
		stft_data = stft(speaker_file, SAMPLE_LEN)
		pickle_dump(stft_data, 'speaker_'+str(i+1))

		# mixing data
		if len(mix_data) == 0:
			mix_data = speaker_data[:SAMPLE_LEN]
		else:
			mix_data += speaker_data[:SAMPLE_LEN]

		# removing this added file from list of unmixed files
		speaker_dict['speaker_'+str(i+1)].pop(file_nums[i])

	
	index = len(os.listdir(os.path.join(raw_data_dir, 'mix'))) + 1
	print(index)

	mix_file = os.path.join(raw_data_dir, 'mix', 'mix_'+str(index)+'.wav')
	
	sf.write(mix_file, mix_data, SAMPLE_RATE)
	pickle_dump(stft(mix_file, SAMPLE_LEN), 'mix')


#util to return a list of files for a speaker
def get_speaker_files(speaker_name):
	return os.listdir(os.path.join(raw_data_dir, speaker_name))

# util to return list of file names for each speaker as a dictionary
def get_speaker_dict():
	speaker_dict = {} # stores names and audio files of speaker dirs

	for folder in os.listdir(os.path.join(raw_data_dir)):
		if folder.startswith('speaker'):
			speaker_dict[folder] = get_speaker_files(folder)

	return speaker_dict

# number of files in a speaker folder
def get_num_files(speaker_dict):
	for speaker in speaker_dict.keys():
		return len(speaker_dict[speaker])

# get a list of n random number in the range(0, max_range)
def get_random_nums(n, max_range):
	rand_list = []
	for _ in range(n):
		rand_list.append(random.randint(0, max_range-1))
	return rand_list

# create mixture of speaker files
def create_speaker_mix(SAMPLE_LEN, SAMPLE_RATE):
	speaker_dict = get_speaker_dict() # list of all audio files of all speakers
	speakers = len(speaker_dict) # number of speakers
	num_files = get_num_files(speaker_dict) # number of times mix must be done


	while num_files > 0:
		file_nums = get_random_nums(speakers, num_files) # list of n random numbers, n = speakers

		mix(speaker_dict, file_nums, SAMPLE_LEN, SAMPLE_RATE) # mix n files
		num_files -= 1
		
	

if __name__ == '__main__':
	
	SAMPLE_LEN, SAMPLE_RATE = set_sample_specs(get_speaker_dict())
	create_speaker_mix(SAMPLE_LEN, SAMPLE_RATE)