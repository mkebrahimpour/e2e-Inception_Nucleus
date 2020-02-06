"""
Implementation of "End-to-End Auditory Object Recognition via Inception Nucleus"
Mohammad K. Ebrahimpour

"""

import pickle
from glob import glob
from random import choice
from time import time
from keras.utils.np_utils import to_categorical
import librosa
import numpy as np

from constants import *




class DataReader:
    def __init__(self):
        self.train_files = glob(os.path.join(OUTPUT_DIR_TRAIN, '**.pkl'))
        print('training files =', len(self.train_files))
        self.test_files = glob(os.path.join(OUTPUT_DIR_TEST, '**.pkl'))
        print('testing files =', len(self.test_files))

    def next_batch_train(self, batch_size):
        return DataReader._next_batch(batch_size, self.train_files)

    def next_batch_test(self, batch_size):
        return DataReader._next_batch(batch_size, self.test_files)

    def train_files_count(self):
        return len(self.train_files)

    def test_files_count(self):
        return len(self.test_files)

    def get_all_training_data(self):
        return DataReader._get_data(self.train_files)

    def get_all_testing_data(self):
        return DataReader._get_data(self.test_files)

    @staticmethod
    def _get_data(file_list, progress_bar=False):
        def load_into(_filename, _x, _y):
            with open(_filename, 'rb') as f:
                audio_element = pickle.load(f)
                _x.append(audio_element['audio'])
                _y.append(int(audio_element['class_id']))

        x, y = [], []
        for filename in file_list:
            load_into(filename, x, y)
        return np.array(x), np.array(y)

    @staticmethod
    def _next_batch(batch_size, file_list):
        return DataReader._get_data([choice(file_list) for _ in range(batch_size)])
    



def read_audio_from_filename(filename, target_sr):
    audio, _ = librosa.load(filename, sr=target_sr, mono=True)
    audio = audio.reshape(-1, 1)
    return audio


def next_batch_blank(batch_size):
    return np.zeros(shape=(batch_size, AUDIO_LENGTH, 1), dtype=np.float32), np.ones(shape=batch_size)

def mixup(data, one_hot_labels, alpha=1, debug=False):
    np.random.seed(42)

    batch_size = data.shape[0]
    weights = np.random.beta(alpha, alpha, batch_size)
    index = np.random.permutation(batch_size)
    x1, x2 = data, data[index]
    x = np.array([x1[i] * weights [i] + x2[i] * (1 - weights[i]) for i in range(len(weights))])
    y1 = np.array(one_hot_labels).astype(np.float)
    y2 = np.array(np.array(one_hot_labels)[index]).astype(np.float)
    y = np.array([y1[i] * weights[i] + y2[i] * (1 - weights[i]) for i in range(len(weights))])
    if debug:
        print('Mixup weights', weights)
    return x, y


if __name__ == '__main__':
#    read_audio_from_filename('samples/15564-2-0-0.wav', target_sr=TARGET_SR)
# =============================================================================
#     data_reader = DataReader()
#     a = time()
#     data_reader.next_batch_train(128)
#     print(time() - a, 'sec')
#     data_reader.next_batch_test(32)
# =============================================================================
    data_reader = DataReader()
    x_tr, y_tr = data_reader.get_all_training_data()
    y_tr = to_categorical(y_tr, num_classes=10)
    x,y = mixup(x_tr,y_tr)

