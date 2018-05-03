""" test my conv_model_gallery for experimental format on Caisa-B Dataset"""

# python packages
import numpy as np
import os
from collections import Counter
from keras.optimizers import SGD
from keras.utils import to_categorical
from sklearn.metrics import accuracy_score
from prettytable import PrettyTable


# project modules
from . import my_models
from . import model_utils
from . import test_preprocessing
from . import config
from ... import root_dir


# path variable and constant 
input_dir = config.crop_img_dir


# display options
table = PrettyTable(["angle", "accuracy"])


def show_prediction_label(predictions):
    
    clip_result = []
    for pred in predictions:
        clip_result.append(np.argmax(pred))


    print("\npredicted subject id through all video clips")  
    print(clip_result)
    data = Counter(clip_result)

    print(data.most_common())
    print("\npredicted subject id: ", data.most_common(1)[0][0] + 1)

        

    
def predict(model, subject_id_list, probe_angle, probe_type):

    if(probe_type == "normal"): probe_seq = ["nm05", "nm06"]
    elif(probe_type == "bag"): probe_seq = ["bg01", "bg02"]
    elif(probe_type == "coat"): probe_seq = ["cl01", "cl02"]
    
    # looping for all probe view angle
    for angle_nb, p_angle in enumerate(probe_angle):
        
        print("\n\npredicting probe set type:", probe_type, ",  angle :", p_angle, " ...")
        row = [p_angle]
        y_pred = []
        y_true = []

        for subject_id in subject_id_list:
            # get data
            X_test, y_test = test_preprocessing.load_test_data(subject_id,
                                                               p_angle,
                                                               probe_seq)

            # for person identification
            #y_true.append(int(subject_id[1:]) - 1)
            # for angle classification
            y_true.append(angle_nb)
            y_test = to_categorical(y_test, config.nb_classes)

            
            # predicting two videos each...
            print("predicting ...")
            predictions = model.predict(X_test,
                                        config.testing_batch_size,
                                        verbose = 2)


            # getting total probabilty score for each probe set
            total_prob_score = np.sum(predictions, axis = 0)
            pred_angle_id = np.argmax(total_prob_score)

            y_pred.append(pred_angle_id)


        print("\ntrue label: ", y_true)
        print("precited label ", y_pred)

        acc_score = accuracy_score(y_true, y_pred)
        print("accuracy: ", acc_score * 100)

        row.append(acc_score * 100)
        table.add_row(row)
	





################# main work here #################
print("\n#### test result of my gait recognition algorithm on CASIA Dataset-B ####")

# test configuration
print("\nstart preprocessing test data")

# calculating total number of person having gait videos
num_subject = len(os.listdir(input_dir))
print("total number subjects:", num_subject)
total_id_list = sorted(os.listdir(input_dir), key = lambda x: int(x[1:]))


print("gallery subject id list: 63 to 124")
subject_id_list = total_id_list[62:124]
print(subject_id_list)


probe_type = "normal"
probe_angle = config.angle_list



# loading trained model
model = model_utils.read_conv_model()

# predicting
predict(model, subject_id_list, probe_angle, probe_type)


print("\n\n############## Summary of my gait recognition algorithm ############## ")
print("Probe set type:", probe_type)
print(table)





