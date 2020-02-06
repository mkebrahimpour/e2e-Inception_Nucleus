"""
Implementation of "End-to-End Auditory Object Recognition via Inception Nucleus"
Mohammad K. Ebrahimpour

"""



import sys
import os
from keras.callbacks import Callback
from keras.callbacks import ReduceLROnPlateau
from keras.utils.np_utils import to_categorical
import tensorflow as tf
from file_logger import FileLogger
from model_data import DataReader
from model_data import *
from models import *
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint


class MetricsHistory(Callback):
    def on_epoch_end(self, epoch, logs={}):
        file_logger.write([str(epoch),
                           str(logs['loss']),
                           str(logs['val_loss']),
                           str(logs['acc']),
                           str(logs['val_acc'])])


if __name__ == '__main__':
    model_name = 'inception'
    args = sys.argv
    if len(args) == 2:
        model_name = args[1].lower()
    print('Model selected:', model_name)
    file_logger = FileLogger('out_{}.tsv'.format(model_name), ['step', 'tr_loss', 'te_loss',
                                                               'tr_acc', 'te_acc'])
    model = None
    num_classes = 10
    
    if model_name == 'inception':
        model = inception_sound(num_classes=num_classes)
    elif model_name == 'inception_full':
        model = fully_inception(num_classes=num_classes)

        

    if model is None:
        exit('Please choose a valid model: [inception]')

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    print(model.summary())

    data_reader = DataReader()
    x_tr, y_tr = data_reader.get_all_training_data()
    y_original = y_tr

    x_te, y_te = data_reader.get_all_testing_data()

    
    y_tr = to_categorical(y_tr, num_classes=num_classes)
    y_te = to_categorical(y_te, num_classes=num_classes)
    


    reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=10, min_lr=0.0001, verbose=1)
    metrics_history = MetricsHistory()
    

    checkpointer = ModelCheckpoint(filepath="best_weights.hdf5", 
                               monitor = 'val_acc',
                               verbose=1, 
                               save_best_only=True)      
    batch_size = 16
    history = model.fit(x=x_tr,
              y=y_tr,
              batch_size=batch_size,
              epochs=300,
              verbose=1,
              shuffle=True,
              validation_split = 0.2,
              callbacks=[metrics_history, reduce_lr, checkpointer])
    
    
    
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()
    
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()
    file_logger.close()
