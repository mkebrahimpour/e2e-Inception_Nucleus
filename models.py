"""
Implementation of "End-to-End Auditory Object Recognition via Inception Nucleus"
Mohammad K. Ebrahimpour

"""

import keras.backend as K
import keras
from keras import regularizers
from keras.layers import Lambda
from keras.layers.convolutional import Conv1D, MaxPooling1D,Conv2D, MaxPooling2D
from keras.layers import Conv2D, MaxPooling2D,GlobalAveragePooling2D, Flatten, LSTM,Conv1D,MaxPooling1D,Permute,Reshape,Concatenate
from keras.layers import Dropout, Dense, TimeDistributed, Input, concatenate
from keras.layers.core import Activation, Dense
from keras.layers.normalization import BatchNormalization
from keras.models import *#Sequential
from keras import backend as K
from keras.regularizers import l2

from constants import *

kernel_init = keras.initializers.he_uniform(seed=1369)
bias_init = keras.initializers.Constant(value=0.2)

def inception_module(x,
                     filters_1x4,
                     filters_1x8_reduce,
                     filters_1x8,
                     filters_1x16_reduce,
                     filters_1x16,
                     kernel_initializer,
                     bias_initializer,
                  #   filters_pool_proj,
                     name=None):
    
    conv_1x4 = Conv1D(filters_1x4, (4), strides = 4, padding='same', activation='relu', kernel_initializer= kernel_init,bias_initializer=bias_init, kernel_regularizer=l2(0.0002))(x)
    
    conv_1x8 = Conv1D(filters_1x8_reduce, (8), strides = 4, padding='same', activation='relu',kernel_initializer= kernel_init,bias_initializer=bias_init, kernel_regularizer=l2(0.0002))(x)
    conv_1x8 = Conv1D(filters_1x8, (8), strides = 4 , padding='same', activation='relu',kernel_initializer= kernel_init,bias_initializer=bias_init, kernel_regularizer=l2(0.0002))(conv_1x8)

    conv_1x16 = Conv1D(filters_1x16_reduce, (16), padding='same', activation='relu',kernel_initializer= kernel_init,bias_initializer=bias_init, kernel_regularizer=l2(0.0002))(x)
    conv_1x16 = Conv1D(filters_1x16, (16), padding='same', activation='relu',kernel_initializer= kernel_init,bias_initializer=bias_init, kernel_regularizer=l2(0.0002))(conv_1x16)

  #  pool_proj = MaxPool2D((3, 3), strides=(1, 1), padding='same')(x)
  #  pool_proj = Conv2D(filters_pool_proj, (1, 1), padding='same', activation='relu', kernel_initializer=kernel_init, bias_initializer=bias_init)(pool_proj)

    output = concatenate([conv_1x4, conv_1x8, conv_1x16], axis= 1, name=name)
    
    return output

def inception_module_cmp(x,
                     filters_1x4,
                     kernel1,
                     filters_1x8_reduce,
                     filters_1x8,
                     kernel2,
                     filters_1x16_reduce,
                     filters_1x16,
                     kernel3,
                     kernel_initializer,
                     bias_initializer,
                  #   filters_pool_proj,
                     name=None):
    
    conv_1x4 = Conv1D(filters_1x4, (kernel1), strides = 4, padding='same', activation='relu', kernel_initializer= kernel_init,bias_initializer=bias_init, kernel_regularizer=l2(0.0002))(x)
    
    conv_1x8 = Conv1D(filters_1x8_reduce, (kernel2), strides = 4, padding='same', activation='relu',kernel_initializer= kernel_init,bias_initializer=bias_init, kernel_regularizer=l2(0.0002))(x)
    conv_1x8 = Conv1D(filters_1x8, (kernel2), strides = 4 , padding='same', activation='relu',kernel_initializer= kernel_init,bias_initializer=bias_init, kernel_regularizer=l2(0.0002))(conv_1x8)

    conv_1x16 = Conv1D(filters_1x16_reduce, (kernel3), padding='same', activation='relu',kernel_initializer= kernel_init,bias_initializer=bias_init, kernel_regularizer=l2(0.0002))(x)
    conv_1x16 = Conv1D(filters_1x16, (kernel3), padding='same', activation='relu',kernel_initializer= kernel_init,bias_initializer=bias_init, kernel_regularizer=l2(0.0002))(conv_1x16)

  #  pool_proj = MaxPool2D((3, 3), strides=(1, 1), padding='same')(x)
  #  pool_proj = Conv2D(filters_pool_proj, (1, 1), padding='same', activation='relu', kernel_initializer=kernel_init, bias_initializer=bias_init)(pool_proj)

    output = concatenate([conv_1x4, conv_1x8, conv_1x16], axis= 1, name=name)
    
    return output

def inception_module_bn(x,
                     filters_1x4,
                     kernel1,
                     filters_1x8_reduce,
                     filters_1x8,
                     kernel2,
                     filters_1x16_reduce,
                     filters_1x16,
                     kernel3,
                     kernel_initializer,
                     bias_initializer,
                  #   filters_pool_proj,
                     name=None):
    
    conv_1x4 = Conv1D(filters_1x4, (kernel1), strides = 4, padding='same', kernel_initializer= kernel_init,bias_initializer=bias_init, kernel_regularizer=l2(0.0002))(x)
    conv_1x4 = BatchNormalization()(conv_1x4)
    conv_1x4 = Activation('relu')(conv_1x4)
    
    
    conv_1x8 = Conv1D(filters_1x8_reduce, (kernel2), strides = 4, padding='same',kernel_initializer= kernel_init,bias_initializer=bias_init, kernel_regularizer=l2(0.0002))(x)
    conv_1x8 = BatchNormalization()(conv_1x8)
    conv_1x8 = Activation('relu')(conv_1x8)
    
    conv_1x8 = Conv1D(filters_1x8, (kernel2), strides = 4 , padding='same',kernel_initializer= kernel_init,bias_initializer=bias_init, kernel_regularizer=l2(0.0002))(conv_1x8)
    conv_1x8 = BatchNormalization()(conv_1x8)
    conv_1x8 = Activation('relu')(conv_1x8)
    
    conv_1x16 = Conv1D(filters_1x16_reduce, (kernel3), padding='same',kernel_initializer= kernel_init,bias_initializer=bias_init, kernel_regularizer=l2(0.0002))(x)
    conv_1x16 = BatchNormalization()(conv_1x16)
    conv_1x16 = Activation('relu')(conv_1x16)
    
    conv_1x16 = Conv1D(filters_1x16, (kernel3), padding='same',kernel_initializer= kernel_init,bias_initializer=bias_init, kernel_regularizer=l2(0.0002))(conv_1x16)
    conv_1x16 = BatchNormalization()(conv_1x16)
    conv_1x16 = Activation('relu')(conv_1x16)
  #  pool_proj = MaxPool2D((3, 3), strides=(1, 1), padding='same')(x)
  #  pool_proj = Conv2D(filters_pool_proj, (1, 1), padding='same', activation='relu', kernel_initializer=kernel_init, bias_initializer=bias_init)(pool_proj)

    output = concatenate([conv_1x4, conv_1x8, conv_1x16], axis= 1, name=name)
    
    return output





def inception_sound(num_classes=10):
    input_layer = Input([AUDIO_LENGTH, 1])
    n_classes = num_classes
    x = Conv1D (kernel_size = (80), filters = 32, strides=(4), activation='relu', name='conv_1x2', kernel_initializer=kernel_init, bias_initializer=bias_init, kernel_regularizer=l2(0.0002))(input_layer)

    x = inception_module(x,
                     filters_1x4=64,
                     filters_1x8_reduce=64,
                     filters_1x8=64,
                     filters_1x16_reduce=64,
                     filters_1x16=64,
                     kernel_initializer= kernel_init,
                     bias_initializer=bias_init,
                   #  filters_pool_proj=32,
                     name='inception_3a')

    x = MaxPooling1D(pool_size = (10), strides=1)(x)
    x = Reshape([64,-1,1])(x)

    x = Conv2D (kernel_size = (3,3), filters = 32,padding='same', activation='relu',kernel_initializer=kernel_init, bias_initializer=bias_init, kernel_regularizer=l2(0.0002))(x)
    x = MaxPooling2D(pool_size = (2,2), strides=(2,2))(x)

    x = Conv2D (kernel_size = (3,3), filters = 64,padding='same', activation='relu',kernel_initializer=kernel_init, bias_initializer=bias_init, kernel_regularizer=l2(0.0002))(x)
    x = Conv2D (kernel_size = (3,3), filters = 64, padding='same',activation='relu',kernel_initializer=kernel_init, bias_initializer=bias_init, kernel_regularizer=l2(0.0002))(x)
    x = MaxPooling2D(pool_size = (2,2), strides=(2,2))(x)

    x = Conv2D (kernel_size = (3,3), filters = 128, padding='same',activation='relu')(x)
    x = MaxPooling2D(pool_size = (2,2), strides=(2,2))(x)
    
    
    x = Conv2D (kernel_size = (1,1), filters = 10,padding='same', activation='relu',kernel_initializer=kernel_init, bias_initializer=bias_init, kernel_regularizer=l2(0.0002))(x)
    x = MaxPooling2D(pool_size = (2,2), strides=(2,2))(x)
    x = GlobalAveragePooling2D(data_format='channels_last')(x)


    x = Activation('softmax')(x)

    model = Model(input_layer, [x], name='inception_v1')
    
    return model

def inception_sound_bn(num_classes=10):
    input_layer = Input([AUDIO_LENGTH, 1])
    n_classes = num_classes
    x = Conv1D (kernel_size = (80), filters = 32, strides=(4), activation='relu', name='conv_1x2', kernel_initializer=kernel_init, bias_initializer=bias_init, kernel_regularizer=l2(0.0002))(input_layer)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = inception_module_bn(x,
                     filters_1x4=64,
                     kernel1 = 4,
                     filters_1x8_reduce=64,
                     filters_1x8=64,
                     kernel2 = 8,
                     filters_1x16_reduce=64,
                     filters_1x16=64,
                     kernel3 = 16,
                     kernel_initializer= kernel_init,
                     bias_initializer=bias_init,
                   #  filters_pool_proj=32,
                     name='inception')

    x = MaxPooling1D(pool_size = (10), strides=1)(x)
    x = Reshape([64,-1,1])(x)

    x = Conv2D (kernel_size = (3,3), filters = 32,padding='same',kernel_initializer=kernel_init, bias_initializer=bias_init, kernel_regularizer=l2(0.0002))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size = (2,2), strides=(2,2))(x)

    x = Conv2D (kernel_size = (3,3), filters = 64,padding='same',kernel_initializer=kernel_init, bias_initializer=bias_init, kernel_regularizer=l2(0.0002))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x) 
    x = Conv2D (kernel_size = (3,3), filters = 64, padding='same',kernel_initializer=kernel_init, bias_initializer=bias_init, kernel_regularizer=l2(0.0002))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x) 
    x = MaxPooling2D(pool_size = (2,2), strides=(2,2))(x)

    x = Conv2D (kernel_size = (3,3), filters = 128, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size = (2,2), strides=(2,2))(x)
    

    
    x = Conv2D (kernel_size = (1,1), filters = 10,padding='same',kernel_initializer=kernel_init, bias_initializer=bias_init, kernel_regularizer=l2(0.0002))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size = (2,2), strides=(2,2))(x)
    x = GlobalAveragePooling2D(data_format='channels_last')(x)


    x = Activation('softmax')(x)

    model = Model(input_layer, [x], name='inception_v1')
    
    return model
   

def inception_sound_filter_sensitivity(num_classes=10):
    input_layer = Input([AUDIO_LENGTH, 1])
    n_classes = num_classes
    x = Conv1D (kernel_size = (80), filters = 32, strides=(4), name='conv_1x2', kernel_initializer=kernel_init, bias_initializer=bias_init, kernel_regularizer=l2(0.0002))(input_layer)

    x = inception_module_cmp(x,
                     filters_1x4=64,
                     kernel1 = 20,
                     filters_1x8_reduce=64,
                     filters_1x8=64,
                     kernel2 = 40,
                     filters_1x16_reduce=64,
                     filters_1x16=64,
                     kernel3 = 60,
                     kernel_initializer= kernel_init,
                     bias_initializer=bias_init,
                   #  filters_pool_proj=32,
                     name='inception_3a')

    x = MaxPooling1D(pool_size = (10), strides=1)(x)
    x = Reshape([64,-1,1])(x)

    x = Conv2D (kernel_size = (3,3), filters = 32,padding='same', activation='relu',kernel_initializer=kernel_init, bias_initializer=bias_init, kernel_regularizer=l2(0.0002))(x)
    x = MaxPooling2D(pool_size = (2,2), strides=(2,2))(x)

    x = Conv2D (kernel_size = (3,3), filters = 64,padding='same', activation='relu',kernel_initializer=kernel_init, bias_initializer=bias_init, kernel_regularizer=l2(0.0002))(x)
    x = Conv2D (kernel_size = (3,3), filters = 64, padding='same',activation='relu',kernel_initializer=kernel_init, bias_initializer=bias_init, kernel_regularizer=l2(0.0002))(x)
    x = MaxPooling2D(pool_size = (2,2), strides=(2,2))(x)

    x = Conv2D (kernel_size = (3,3), filters = 128, padding='same',activation='relu')(x)
    x = MaxPooling2D(pool_size = (2,2), strides=(2,2))(x)
    

    
    x = Conv2D (kernel_size = (1,1), filters = 10,padding='same', activation='relu',kernel_initializer=kernel_init, bias_initializer=bias_init, kernel_regularizer=l2(0.0002))(x)
    x = MaxPooling2D(pool_size = (2,2), strides=(2,2))(x)
    x = GlobalAveragePooling2D(data_format='channels_last')(x)


    x = Activation('softmax')(x)

    model = Model(input_layer, [x], name='inception_v1')
    
    return model

def fully_inception(num_classes=10):
    input_layer = Input([AUDIO_LENGTH, 1])
    n_classes = num_classes
    x = inception_module_cmp(input_layer,
                     filters_1x4=32,
                     kernel1 = 60,
                     filters_1x8_reduce=32,
                     filters_1x8=32,
                     kernel2 = 80,
                     filters_1x16_reduce=32,
                     filters_1x16=32,
                     kernel3 = 100,
                     kernel_initializer= kernel_init,
                     bias_initializer=bias_init,
                   #  filters_pool_proj=32,
                     name='inception_1')


    x = MaxPooling1D(pool_size = (10), strides=1)(x)
    x = Reshape([32,-1,1])(x)

    x = Conv2D (kernel_size = (3,3), filters = 32,padding='same', activation='relu',kernel_initializer=kernel_init, bias_initializer=bias_init, kernel_regularizer=l2(0.0002))(x)
    x = MaxPooling2D(pool_size = (2,2), strides=(2,2))(x)

    x = Conv2D (kernel_size = (3,3), filters = 64,padding='same', activation='relu',kernel_initializer=kernel_init, bias_initializer=bias_init, kernel_regularizer=l2(0.0002))(x)
    x = Conv2D (kernel_size = (3,3), filters = 64, padding='same',activation='relu',kernel_initializer=kernel_init, bias_initializer=bias_init, kernel_regularizer=l2(0.0002))(x)
    x = MaxPooling2D(pool_size = (2,2), strides=(2,2))(x)

    x = Conv2D (kernel_size = (3,3), filters = 128, padding='same',activation='relu')(x)
    x = MaxPooling2D(pool_size = (2,2), strides=(2,2))(x)
    

    
    x = Conv2D (kernel_size = (1,1), filters = 10,padding='same', activation='relu',kernel_initializer=kernel_init, bias_initializer=bias_init, kernel_regularizer=l2(0.0002))(x)
    x = MaxPooling2D(pool_size = (2,2), strides=(2,2))(x)
    x = GlobalAveragePooling2D(data_format='channels_last')(x)


    x = Activation('softmax')(x)

    model = Model(input_layer, [x], name='inception_v1')
    
    return model
