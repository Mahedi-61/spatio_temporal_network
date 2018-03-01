"""
Author : Md. Mahedi Hasan
Project: spatio_temporal_network
File: img_preprocessing
Description: this file is for image preprocessing
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
train_fpc = config.train_fpc
divisor = (clip_size // train_fpc)




    
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

    print("\n")
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

        if ((clip_no + 1) % 1000 == 0):
            print("processing {0} image clips".format(clip_no + 1))
            
    print("processing {0} image clips".format(clip_no + 1))
    return data



    

# process images for gait recognition
def process_images(subject_id_list, walking_seq, angle_list = []):

    #images and labels
    X_images = []
    y_labels = []

    # considering each subject
    for subject_id in subject_id_list:     
        subject_dir = os.path.join(input_dir, subject_id)

        total_clip_for_each_sub = 0

        # getting angle
        ls_angle = sorted(os.listdir(subject_dir), key = lambda x: int(x[-3:]))

        #check angle_list is provided or not
        if(len(angle_list) == 0):
            angle_list = ls_angle

        num_angle =  len(angle_list)
        print("\n\n%s subject have: %d angle gait vidoes" % (subject_id, num_angle))
        
        # considering each angle
        for angle in angle_list:
            subject_angle_dir = os.path.join(subject_dir, angle)

            # considering each gait sequence
            for seq in walking_seq:
                seq_dir = os.path.join(subject_angle_dir, seq)
                input_img_list = sorted(os.listdir(seq_dir), key = lambda x: int(x.split(".")[0]))
                input_img_dir = [os.path.join(seq_dir, input_img) for input_img in input_img_list]

                # dividing each gait sequence according to clip_size
                clip_no = (len(input_img_list) // clip_size) * divisor
                
                if(clip_no > 0):
                    total_clip_for_each_sub += (clip_no - (divisor -1))

                    for i in range(0, clip_no - (divisor -1)):
                        b_id = i * train_fpc
                        e_id = clip_size + (i * train_fpc)
                        X_images.append(input_img_dir[b_id : e_id])

                        # label start from 0
                        y_labels.append(int(subject_id[1:]) - 1)

        print("%s subject has total %d clip for %s" %(subject_id,
                                                      total_clip_for_each_sub, walking_seq))

    return X_images, y_labels






# methods for train set data loading
def load_train_data():
    print("\nstart preprocessing train data")

    # calculating total number of person having gait videos
    num_subject = len(os.listdir(input_dir))
    print("total number subjects for training: ", num_subject)

    subject_id_list = sorted(os.listdir(input_dir), key = lambda x: int(x[1:]))
    print(subject_id_list)

    # getting all training images and labels
    X_images, y_labels = process_images(subject_id_list, config.ls_train_seq)

    # converting raw images to numpy array
    X_train = get_data_from_images(X_images)

    # calculating and subtracting clip mean
    print("\nsubtracting mean_cube ...")
    
    mean_cube = np.mean(X_train, axis = 0)
    X_train -= mean_cube
    np.save(casia_mean_cube_file_dir, mean_cube)
    
    return X_train, y_labels  





def load_validation_data():
    print("\nstart preprocessing validation data")

    # calculating total number of person having gait videos
    num_subject = len(os.listdir(input_dir))
    print("total number subjects for validation: ", num_subject)

    subject_id_list = sorted(os.listdir(input_dir), key = lambda x: int(x[1:]))
    print(subject_id_list)

    # getting all training images and labels
    X_images, y_labels = process_images(subject_id_list, config.ls_valid_seq)

    # converting raw images to numpy array
    X_valid = get_data_from_images(X_images)

    # calculating and subtracting clip mean
    print("\nsubtracting mean_cube ...")
    mean_cube = np.load(casia_mean_cube_file_dir)
    X_valid -= mean_cube
    
    return X_valid, y_labels 







# methods for gallery set data loading
def load_gallery_data(data_type):
    print("\nstart preprocessing %s data" % data_type)

    # calculating total number of person having gait videos
    num_subject = len(os.listdir(input_dir))
    print("total number subjects for %s %s: " % (data_type, num_subject))

    subject_id_list = sorted(os.listdir(input_dir), key = lambda x: int(x[1:]))
    print(subject_id_list)

    # getting all images and labels
    if(data_type == "train"):
        X_images, y_labels = process_images(subject_id_list,
                                            config.ls_gallery_train_seq)

    elif(data_type == "valid"):
        X_images, y_labels = process_images(subject_id_list,
                                            config.ls_gallery_valid_seq)
        
    # converting raw images to numpy array
    X_data = get_data_from_images(X_images)

    # calculating and subtracting clip mean
    print("\nsubtracting mean_cube ...")
    
    mean_cube = np.load(casia_mean_cube_file_dir)
    X_data -= mean_cube
    
    return X_data, y_labels  





# developing casia_dataset_mean
def make_casia_dataset_mean():
    
    # considering each subject
    for subject_id in subject_id_list:     
        subject_dir = os.path.join(input_dir, subject_id)

        # getting angle
        angle_list = sorted(os.listdir(subject_dir), key = lambda x: int(x[-3:]))
        num_angle =  len(angle_list)
        print("\n\n%s subject have: %d angle gait vidoes" % (subject_id, num_angle))

       
        # considering each angle
        for angle in angle_list:
            subject_angle_dir = os.path.join(subject_dir, angle)

            # getting sequence
            seq_list = sorted(os.listdir(subject_angle_dir))
            num_seq = len(seq_list)
            print("%s angle have %d gait sequence" % (angle, num_seq))

            # considering each gait sequence
            for seq in seq_list:
                seq_dir = os.path.join(subject_angle_dir, seq)
                
                input_img_list = sorted(os.listdir(seq_dir), key = lambda x: int(x.split(".")[0]))
                input_img_dir = [os.path.join(seq_dir, input_img) for input_img in input_img_list]

                 # dividing each gait sequence according to clip_size
                clip_size = 16 # no overlapping 
                clip_no = (len(input_img_list) // clip_size)
                
                if(clip_no > 0):
                    total_clip_for_each_sub +=  clip_no

                    for i in range(0, clip_no):
                        b_id = i * clip_size
                        e_id = clip_size + (i * clip_size)
                        X_images.append(input_img_dir[b_id : e_id])




if __name__ == '__main__':
    load_train_data()




















