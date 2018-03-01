"""
Author : Md. Mahedi Hasan
Project: spatio_temporal_network
File: config.py
Description: this file contains configuration info
"""

# video clip 
clip_size = 16
train_fpc = 8   # fpc: frame_per_clip



# image and input shape to the model
img_size = 112
img_channel = 3
img_shape = (img_size, img_size, img_channel)

input_shape = (clip_size, img_size, img_size, img_channel)
nb_classes = 50



# train and validation sequence
ls_train_seq = ["bg01", "bg02", "cl01", "cl02" , "nm01", "nm02", "nm03", "nm04"]
ls_valid_seq = ["nm05", "nm06"]

ls_gallery_train_seq =  ["nm01", "nm02", "nm03", "nm04"]
ls_gallery_valid_seq =  ["bg01", "bg02", "cl01", "cl02"]


# test video clip
test_fpc = 4


# network training parameter
learning_rate = 1e-3
training_batch_size = 12
training_epochs = 40


# model utilites
lr_reduce_factor = 0.5
lr_reduce_patience = 20
early_stopping_patience = 30


# model testing configuration
