"""
Implementation of "End-to-End Auditory Object Recognition via Inception Nucleus"
Mohammad K. Ebrahimpour

"""


import pickle
from glob import iglob
from shutil import rmtree
import os
import numpy as np

from constants import *
from model_data import read_audio_from_filename


def add_noise(data):
    noise = np.random.randn(len(data))
    noise = np.reshape(noise,[data.shape[0],data.shape[1]])
    data_noise = data + 0.005 * noise
    return data_noise

def shift(data):
    return np.roll(data, 800)

def mkdir_p(path):
    import errno
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def del_folder(path):
    try:
     tree(path)
    except:
        pass


del_folder(OUTPUT_DIR_TRAIN)
del_folder(OUTPUT_DIR_TEST)
mkdir_p(OUTPUT_DIR_TRAIN)
mkdir_p(OUTPUT_DIR_TEST)


def extract_class_id(wav_filename):
    """
    The name of the audio file. The name takes the following format: [fsID]-[classID]-[occurrenceID]-[sliceID].wav, where:
    [fsID] = the Freesound ID of the recording from which this excerpt (slice) is taken
    [classID] = a numeric identifier of the sound class (see description of classID below for further details)
    [occurrenceID] = a numeric identifier to distinguish different occurrences of the sound within the original recording
    [sliceID] = a numeric identifier to distinguish different slices taken from the same occurrence
    """
    return wav_filename.split('-')[1]


def convert_data():
    for i, wav_filename in enumerate(iglob(os.path.join(DATA_AUDIO_DIR, '**/**.wav'), recursive=True)):
        class_id = extract_class_id(wav_filename)
        audio_buf = read_audio_from_filename(wav_filename, target_sr=TARGET_SR)
        # normalize mean 0, variance 1
        audio_buf = (audio_buf - np.mean(audio_buf)) / np.std(audio_buf)
        original_length = len(audio_buf)
        print(i, wav_filename, original_length, np.round(np.mean(audio_buf), 4), np.std(audio_buf))
        if original_length < AUDIO_LENGTH:
            audio_buf = np.concatenate((audio_buf, np.zeros(shape=(AUDIO_LENGTH - original_length, 1))))
            print('PAD New length =', len(audio_buf))
        elif original_length > AUDIO_LENGTH:
            audio_buf = audio_buf[0:AUDIO_LENGTH]
            print('CUT New length =', len(audio_buf))

        output_folder = OUTPUT_DIR_TRAIN
        if 'fold10' in wav_filename:
            output_folder = OUTPUT_DIR_TEST
        output_filename = os.path.join(output_folder, str(i) + '.pkl')
        
        
        noisy_data = add_noise(audio_buf)
        shifted_data = shift(audio_buf)
        
        if 'fold10' in wav_filename:
            out = {'class_id': class_id,
               'audio': audio_buf,
               'sr': TARGET_SR}
            with open(output_filename, 'wb') as w:
                pickle.dump(out, w)
        else:            
            out = {'class_id': class_id,
               'audio': audio_buf,
               'sr': TARGET_SR}
            out_noisy = {'class_id': class_id,
               'audio': noisy_data,
               'sr': TARGET_SR}
            out_shift = {'class_id': class_id,
               'audio': shifted_data,
               'sr': TARGET_SR}
        output_filename = os.path.join(output_folder, str(i)+'.pkl')
        with open(output_filename, 'wb') as w:
            pickle.dump(out, w)
        output_filename = os.path.join(output_folder, str(i)+'-2' + '.pkl')
        with open(output_filename, 'wb') as w:
            pickle.dump(out_noisy, w)
        output_filename = os.path.join(output_folder, str(i)+'-3' + '.pkl')
        with open(output_filename, 'wb') as w:
            pickle.dump(out_shift, w)


if __name__ == '__main__':
    convert_data()
print ('Done!')