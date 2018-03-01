"""
Author : Md. Mahedi Hasan
Project: spatio_temporal_network
File: test_preprocessing
Description: this file is for test image preprocessing
"""


# python packages
import numpy as np
import PIL
import os
import matplotlib.pyplot as plt
from PIL import Image


# project modules
from ... import root_dir
from . import config



# path variables and constant
input_dir = os.path.join(root_dir.data_path(), "crop_img")
output_dir =  os.path.join(root_dir.stn_path(), "output")
casia_mean_cube_file_dir = os.path.join(output_dir, "casia_mean_cube.npy")

img_size = config.img_size
img_channel = config.img_channel

img_shape = config.img_shape
input_shape = config.input_shape

clip_size = config.clip_size
test_fpc = config.test_fpc
divisor = (clip_size // test_fpc)




# methods for preprocessing
def resize_image(img, size):
    #Pillow return images size as (w, h)
    width, height = img.size 

    if(width > height):
        new_width = size
        new_height = int(size * (height / width) + 0.5)

    else:
        new_height = size
        new_width = int(size * (width / height) + 0.5)

    # resize for keeping aspect ratio
    img_res = img.resize((new_width, new_height), resample = PIL.Image.BICUBIC)

    # pad the borders to create a square image
    img_pad = Image.new("RGB", (size, size), (128, 128, 128))
    ulc = ((size - new_width) // 2, (size - new_height) // 2)
    img_pad.paste(img_res, ulc)

    return img_pad







# methods for data preparation
def get_data_from_images(images):
    data = np.ndarray((len(images), clip_size, img_size, img_size, img_channel),
                      dtype = np.float32)

    #Pillow returns numpy array of (width, height,channel(RGB))
    for clip_no, img_clips in enumerate(images):

        data_clip = np.ndarray((clip_size, img_size, img_size, img_channel),
                      dtype = np.float32)

        for i, img_file in enumerate(img_clips):
            img = Image.open(img_file)
            img = resize_image(img, img_size)

            #convert PIL image to numpy array
            img_px = np.array(img)
            data_clip[i] = img_px

        data[clip_no] = data_clip
        del data_clip

    print("processing {0} test image clips".format(clip_no + 1))
    return data





    

# process images for gait recognition
def process_test_images(subject_id,
                        probe_angle,
                        probe_seq):


    #images and labels
    X_images = []
    y_labels = []

    # considering each subject    
    subject_dir = os.path.join(input_dir, subject_id)

    total_clip_for_each_sub = 0
    subject_angle_dir = os.path.join(subject_dir, probe_angle)
    
    # considering each gait sequence
    for seq in probe_seq:
        seq_dir = os.path.join(subject_angle_dir, seq)
        
        input_img_list = sorted(os.listdir(seq_dir), key = lambda x: int(x.split(".")[0]))
        input_img_dir = [os.path.join(seq_dir, input_img) for input_img in input_img_list]

        # dividing each gait sequence according to clip_size
        clip_no = (len(input_img_list) // clip_size) * divisor
        
        if(clip_no > 0):
            total_clip_for_each_sub += (clip_no - (divisor -1))

            for i in range(0, clip_no - (divisor -1)):
                b_id = i * test_fpc
                e_id = clip_size + (i * test_fpc)
                X_images.append(input_img_dir[b_id : e_id])

                # label start from 0
                y_labels.append(int(subject_id[1:]) - 1)


    print("test data of:: subject: %s, angel: %s, sequence: %s" %
                              (subject_id, probe_angle, probe_seq))
    
    print("total video clip to predict: " , total_clip_for_each_sub)

    return X_images, y_labels







def load_test_data(subject_id_list,  probe_seq, probe_angle):
    print("\nstart preprocessing test data ...")

    # getting all training images and labels
    X_images, y_labels = process_test_images(subject_id_list,
                                             probe_seq,
                                             probe_angle)

    # converting raw images to numpy array
    X_test= get_data_from_images(X_images)

    # calculating and subtracting clip mean
    mean_cube = np.load(casia_mean_cube_file_dir)
    X_test -= mean_cube
    
    return X_test, y_labels 





if __name__ == '__main__':
    load_test_data("p002", "angle_000", ["nm05", "nm06"])



















