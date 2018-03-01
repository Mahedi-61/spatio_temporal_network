"""
Author : Md. Mahedi Hasan
Project: spatio_temporal_features
File: train.py
Description: train my c3d model for extracting spatio-temporal features
"""

# python packages
import numpy as np
from keras.optimizers import SGD
from keras.utils import to_categorical


# project files
from . import my_models
from . import model_utils
from . import config
from . import img_preprocessing


# path variable and constant 
nb_epochs = config.training_epochs
batch_size = config.training_batch_size
lr = config.learning_rate



# preprocessing
X_train, y_train = img_preprocessing.load_train_data()
y_train = to_categorical(y_train, config.nb_classes)
print("\ntrian data shape: ", X_train.shape)
print("train label shape: ", len(y_train))



X_valid, y_valid = img_preprocessing.load_validation_data()
y_valid = to_categorical(y_valid, config.nb_classes)
print("\nvalid data shape: ", X_valid.shape)
print("valid label shape: ", len(y_valid))



# constructing model
model = model_utils.read_conv_model()


optimizers = SGD(lr = lr,
                  momentum = 0.9,
                  nesterov = True)



objective = "mean_squared_error"
model.compile(optimizer = optimizers,
              loss =     objective,
              metrics = ['accuracy'])



# training and evaluating 
history = model_utils.LossHistory()
early_stopping = model_utils.set_early_stopping()
model_cp = model_utils.set_conv_model_gallery_checkpoint()
reduce_lr = model_utils.set_reduce_lr()


"""
# saving model json file
model_utils.save_conv_model_gallery(model)

model.fit(X_train,
          y_train,
          batch_size = batch_size,
          shuffle = True,
          epochs = nb_epochs,
          callbacks = [early_stopping, model_cp, reduce_lr],
          verbose = 2,
          validation_data = (X_valid, y_valid))



# drawing historical loss function
#loss = history.losses
#val_loss = history.val_losses
# model_utils.show_loss_function(loss, val_loss, num_epochs)
"""


























