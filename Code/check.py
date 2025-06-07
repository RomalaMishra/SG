import numpy as np
from keras.utils import np_utils
import pickle

with open("train_labels", "rb") as f:
		train_labels = np.array(pickle.load(f), dtype=np.int32)
print(max(train_labels))
# train_labels = np_utils.to_categorical(train_labels)
# print(train_labels)