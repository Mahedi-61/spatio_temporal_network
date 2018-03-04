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


# all models are palced and stored in model directory
# model checkpoint are saved in checkpoint directory
# so final model should be replaced from checkpoint to model dir.


# saving function
def save_conv_model(model):
    print("saving model.....")
    
    json_string = model.to_json()
    open(config.train_conv_model_path, 'w').write(json_string)



def save_conv_model_gallery(model):
    print("saving model.....")
    
    json_string = model.to_json()
    open(config.train_conv_model_gallery_path, 'w').write(json_string)




def set_conv_model_checkpoint():

    train_conv_model_weight_path = os.path.join(config.checkpoint_dir,
                                        config.train_conv_model_weight)
    
    return ModelCheckpoint(train_conv_model_weight_path,
                monitor = 'val_loss',
                verbose = 2,
                save_best_only = True,
                save_weights_only = True,
                mode = 'auto',
                period = 5)



def set_conv_model_gallery_checkpoint():

    train_conv_model_gallery_weight_path = os.path.join(config.checkpoint_dir,
                            config.train_conv_model_gallery_weight)
    
    return ModelCheckpoint(train_conv_model_gallery_weight_path,
                monitor = 'val_loss',
                verbose = 2,
                save_best_only = True,
                save_weights_only = True,
                mode = 'auto',
                period = 0)





# reading function
def read_conv_model():
    print("reading stored conv_model architecture and weight ...")
    
    json_string = open(config.train_conv_model_path).read()

    model = model_from_json(json_string)
    model.load_weights(config.train_conv_model_weight_path)

    return model




def read_conv_model_gallery():
    print("reading stored conv_model_gallery architecture and weight ...")
    
    json_string = open(config.train_conv_model_gallery_path).read()

    model = model_from_json(json_string)
    model.load_weights(config.train_conv_model_gallery_weight_path)

    return model



# utilities funciton
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



