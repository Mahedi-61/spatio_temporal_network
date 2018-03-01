"""
Author     : Md. Mahedi Hasan
Project    : spatio_temporal_network
File : model_utils.py
Description: this file contains code for handling model
"""


# python packages
import numpy as np
import os
import matplotlib.pyplot as plt
from keras.models import model_from_json

from keras.callbacks import (EarlyStopping,
                             Callback,
                             ModelCheckpoint,
                             ReduceLROnPlateau)


# project modules
from ... import root_dir
from . import config



# path variable and constant
"""
# ** For convenience of repeated experiment following rule is followed. 
# ** trianed model is saved output directory 
# ** But trained model is read from weights directory.
# ** please copy and paste the model files in weights directory before reading
"""

output_dir = os.path.join(root_dir.stn_path(), "output")
weights_dir = os.path.join(root_dir.stn_path(), "weights")

train_model_name = "my_train_conv_model.json"
train_model_weight = "my_train_conv_weight.h5"



# utilites function
def save_model(model):
    print("saving model.....")
    train_model_dir = os.path.join(output_dir, train_model_name)
    
    json_string = model.to_json()
    open(train_model_dir, 'w').write(json_string)




def read_model():
    print("reading stored model architecture and weight")
    train_model_dir = os.path.join(weights_dir, train_model_name)
    train_model_weight_dir = os.path.join(weights_dir, train_model_weight)
    
    json_string = open(train_model_dir).read()

    model = model_from_json(json_string)
    model.load_weights(train_model_weight_dir)

    return model




class LossHistory(Callback):
    def on_train_begin(self, batch, logs = {}):
        self.losses = []
        self.val_losses = []
        

    def on_epoch_end(self, batch, logs = {}):
        self.losses.append(logs.get("loss"))
        self.val_losses.append(logs.get("val_loss"))




def set_early_stopping():
    return EarlyStopping(monitor = "val_loss",
                               patience = config.early_stopping_patience,
                               mode = "auto",
                               verbose = 2)





def set_model_checkpoint():
    train_model_weight_dir = os.path.join(output_dir, train_model_weight)
    
    return ModelCheckpoint(train_model_weight_dir,
                monitor = 'val_loss',
                verbose = 2,
                save_best_only = True,
                save_weights_only = True,
                mode = 'auto',
                period = 5)





def set_reduce_lr():
    return ReduceLROnPlateau(monitor='val_loss',
                             factor = config.lr_reduce_factor,
                             patience = config.lr_reduce_patience,
                             min_lr = 5e-5)





def show_loss_function(loss, val_loss, nb_epochs):
    plt.xlabel("Epochs ------>")
    plt.ylabel("Loss -------->")
    plt.title("Loss function")
    plt.plot(loss, "blue", label = "Training Loss")
    plt.plot(val_loss, "green", label = "Validation Loss")
    plt.xticks(range(0, nb_epochs)[0::2])
    plt.legend()
    plt.show()



    
