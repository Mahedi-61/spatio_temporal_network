"""configuration info for spatio_temporal_network"""

# python packages
import os


# project modules
from ... import root_dir



# path vairables and constant
model_dir = os.path.join(root_dir.stn_path(), "model") 
checkpoint_dir = os.path.join(root_dir.stn_path(), "checkpoint")
output_dir =  os.path.join(root_dir.stn_path(), "output")
crop_img_dir = os.path.join(root_dir.data_path(), "crop_img")

# mean cube file over casia 124 subjects
casia_mean_cube_file_path = os.path.join(output_dir, "casia_mean_cube.npy")


# video clip 
clip_size = 16
train_fpc = 8   # fpc: frame_per_clip
angle_list = ["angle_000", "angle_018", "angle_036", "angle_054",
              "angle_072", "angle_090", "angle_108", "angle_126",
              "angle_144", "angle_162", "angle_180"]


# image and input shape to the model
img_size = 112
img_channel = 3
img_shape = (img_size, img_size, img_channel)

input_shape = (clip_size, img_size, img_size, img_channel)
nb_classes = 11



# train and validation sequence
ls_train_seq = ["nm01", "nm02", "nm03", "nm04"]
ls_valid_seq = ["nm05", "nm06"]


#ls_gallery_train_seq =  ["nm01", "nm02", "nm03", "nm04"]
#ls_gallery_valid_seq =  ["bg01", "bg02"]


# test video clip
test_fpc = 8
testing_batch_size = 12


# network training parameter
learning_rate = 1e-3
training_batch_size = 12
training_epochs = 40


# model utilites
lr_reduce_factor = 0.5
lr_reduce_patience = 5
early_stopping_patience = 10


# model and their weights name
c3d_weights_file = "c3d_weights_tf.h5"
c3d_weights_path = os.path.join(model_dir, c3d_weights_file)


conv_model = "conv_model.json"
conv_model_path = os.path.join(model_dir, conv_model)

conv_model_weight = "conv_model_weight.h5"
conv_model_weight_path = os.path.join(model_dir, conv_model_weight)

#conv_model_gallery = "conv_model_gallery.json"
#conv_model_gallery_path = os.path.join(model_dir, conv_model_gallery)

#conv_model_gallery_weight = "conv_model_gallery_weight.h5"
#conv_model_gallery_weight_path = os.path.join(model_dir, conv_model_gallery_weight)























