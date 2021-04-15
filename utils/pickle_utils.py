import pickle
import os
import numpy as np
from global_vars import *

if __name__ == '__main__':

	# arr = np.array([[1,2],[3,4]])
	# pickle.dump(arr, open('sample.pkl', 'wb'))

	ob = pickle.load(open(os.path.join(pickle_data_dir, 'speaker_1', '15.pkl'), 'rb'))
	print(ob.shape)
	print(ob)

	# print(pickle.load(open('sample.pkl', 'rb')))