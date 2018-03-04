"""
Author : Md. Mahedi Hasan
Project: spatio_temporal_network
File: config.py
Description: this file contains configuration info
"""

# python packages
import os


# project modules
from ... import root_dir



# path vairables and constant
model_dir = os.path.join(root_dir.stn_path(), "model") 
checkpoint_dir = os.path.join(root_dir.stn_path(), "checkpoint")
output_dir =  os.path.join(root_dir.stn_path(), "output")

# mean cube file over casia 124 subjects
casia_mean_cube_file_path = os.path.join(output_dir, "casia_mean_cube.npy")

# video clip 
clip_size = 16
train_fpc = 8   # fpc: frame_per_clip



# image and input shape to the model
img_size = 112
img_channel = 3
img_shape = (img_size, img_size, img_channel)

input_shape = (clip_size, img_size, img_size, img_channel)
nb_classes = 62



# train and validation sequence
ls_train_seq = ["bg01", "bg02", "cl01", "cl02" , "nm01", "nm02", "nm03", "nm04"]
ls_valid_seq = ["nm05", "nm06"]

ls_gallery_train_seq =  ["nm01", "nm02", "nm03", "nm04", "nm05", "nm06"]
ls_gallery_valid_seq =  ["bg01", "bg02"]


# test video clip
test_fpc = 4


# network training parameter
learning_rate = 1e-3
training_batch_size = 12
training_epochs = 40


# model utilites
lr_reduce_factor = 0.5
lr_reduce_patience = 15
early_stopping_patience = 30


# model testing configuration


# model and their weights name
c3d_weights_file = "c3d_weights_tf.h5"
c3d_weights_path = os.path.join(model_dir, c3d_weights_file)


train_conv_model = "train_conv_model.json"
train_conv_model_path = os.path.join(model_dir, train_conv_model)

train_conv_model_weight = "train_conv_model_weight.h5"
train_conv_model_weight_path = os.path.join(model_dir, train_conv_model_weight)

train_conv_model_gallery = "train_conv_model_gallery.json"
train_conv_model_gallery_path = os.path.join(model_dir, train_conv_model_gallery)

train_conv_model_gallery_weight = "train_conv_model_gallery_weight.h5"
train_conv_model_gallery_weight_path = os.path.join(model_dir, train_conv_model_gallery_weight)























