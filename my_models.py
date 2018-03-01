"""
Author : Md. Mahedi Hasan
Project: spatio_temporal_network
File : my_models.py
Description: this file contains training model
"""

# python packages
from keras.models import Sequential, model_from_json
from keras.models import load_model
from keras.layers import (Flatten, Input, Dense, Dropout, AveragePooling2D)

# project modules
from . import config
from . import c3d


# path variables and constant
nb_classes = config.nb_classes


# finetuning pre-train c3d model
def model_c3d():
    print("\nconstructing c3d model... ")
    model = c3d.get_c3d(config.c3d_weights_path)


    # this model contains total 20 layer
    my_model = c3d.get_int_c3d(model, "pool5")

    # freezing upto 15 layer
    for layer in my_model.layers[:14]:
        layer.trainable = False

    # adding my own layers for gait recognition problem
    my_model.add(Flatten())
    
    # FC layers group
    my_model.add(Dense(4096, activation='relu', name='fc6'))
    my_model.add(Dropout(.5))
    
    my_model.add(Dense(4096, activation='relu', name='fc7'))
    my_model.add(Dropout(.5))
    
    my_model.add(Dense(nb_classes, activation='softmax', name='softmax_layer'))
    return my_model
    


def model_conv_for_gallery():
    pass




    
if __name__ == "__main__":
    model_c3d().summary()


















    
