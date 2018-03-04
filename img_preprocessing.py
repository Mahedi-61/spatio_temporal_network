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
def process_images(subject_id_list,
                   walking_seq,
                   angle_list = [],
                   start_id = 1):

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
                        y_labels.append(int(subject_id[1:]) - start_id)

        print("%s subject has total %d clip for %s" %(subject_id,
                                                      total_clip_for_each_sub, walking_seq))

    return X_images, y_labels






# methods for train set data loading
def load_train_data(data_type):
    print("\nstart preprocessing %s data" % data_type)

    # calculating total number of person having gait videos
    num_subject = len(os.listdir(input_dir))
    print("total number subjects: ", num_subject)

    total_id_list = sorted(os.listdir(input_dir), key = lambda x: int(x[1:]))

    print("train subject id list: 1 to 62")
    subject_id_list = total_id_list[:62]


    # getting all training images and labels
    if(data_type == "train"):
        X_images, y_labels = process_images(subject_id_list,
                                            config.ls_train_seq,
                                            start_id = 1)

    # getting all training images and labels
    elif(data_type == "valid"):
        X_images, y_labels = process_images(subject_id_list,
                                            config.ls_valid_seq,
                                            start_id = 1)


    # converting raw images to numpy array
    X_data = get_data_from_images(X_images)


    # calculating and subtracting clip mean
    print("\nsubtracting mean_cube ...")
    mean_cube = np.load(config.casia_mean_cube_file_path)
    X_data -= mean_cube

    return X_data, y_labels 







# methods for gallery set data loading
def load_gallery_data(data_type):
    print("\nstart preprocessing %s data" % data_type)

    # calculating total number of person having gait videos
    num_subject = len(os.listdir(input_dir))
    print("total number subjects for %s %s: " % (data_type, num_subject))

    total_id_list = sorted(os.listdir(input_dir), key = lambda x: int(x[1:]))

    print("gallery subject id list: 63 to 124")
    gallery_subject_id_list = total_id_list[62:124]
    print(gallery_subject_id_list)


    # getting all images and labels
    if(data_type == "train"):
        X_images, y_labels = process_images(gallery_subject_id_list,
                                            config.ls_gallery_train_seq,
                                            start_id = 63)

    elif(data_type == "valid"):
        X_images, y_labels = process_images(gallery_subject_id_list,
                                            config.ls_gallery_valid_seq,
                                            start_id = 63)

        
    # converting raw images to numpy array
    X_data = get_data_from_images(X_images)

    # calculating and subtracting clip mean
    print("\nsubtracting mean_cube ...")
    
    mean_cube = np.load(config.casia_mean_cube_file_path)
    X_data -= mean_cube
    
    return X_data, y_labels  






# developing casia_dataset_mean for 124 subjects
def make_casia_dataset_mean():
    print("\nconstructing mean cube over casiaB dataset")
    X_images = []

    # calculating total number of person having gait videos
    num_subject = len(os.listdir(input_dir))
    print("total number subjects for %s: " % (num_subject))

    total_id_list = sorted(os.listdir(input_dir), key = lambda x: int(x[1:]))

    # considering each subject
    for subject_id in total_id_list:
        subject_dir = os.path.join(input_dir, subject_id)

        total_clip_for_each_sub = 0

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



    print("preparing data ...")
    X_data = get_data_from_images(X_images)

    print("calculating mean ...")
    mean_cube = np.mean(X_data, axis = 0)

    print("saving mean in output directory")
    np.save(config.casia_mean_cube_file_path, mean_cube)
        





if __name__ == '__main__':
    make_casia_dataset_mean()












