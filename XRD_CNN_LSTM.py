# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 11:42:02 2019

@author: Arun.Sagotra
"""

#-*- coding: utf-8 -*-  
#!usr/bin/env python  

"""
SPACE GROUP CNN + LSTM 

"""
################################################################# 
#Libraries and dependencies
#################################################################

# Loads series of functions for preprocessing and data augmentation
from XRD_normalize import * 
# Loads CAMs visualizations for a-CNN
from XRD_vis import * 

import numpy as np  
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn import metrics
from sklearn.model_selection import KFold

# Neural networks uses Keras with TF background
import keras as K
from keras.models import Model
from keras.models import Sequential
from keras.layers import Dropout
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import EarlyStopping
from keras.layers import  MaxPooling1D
import tensorflow as tf
from keras.callbacks import TensorBoard
# Clear Keras and TF session, if run previously
K.backend.clear_session()
tf.reset_default_graph()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
# Training Parameters

BATCH_SIZE=128

# Network Parameters
n_input = 1200 # Total angles in XRD pattern
n_classes = 7 # Number of space-group classes
filter_size = 2
kernel_size = 10

################################################################
# Load data and preprocess
################################################################

# Load simulated and anonimized dataset
import os
dirname = os.path.dirname(__file__)
theor = pd.read_csv(os.path.join(dirname, 'Datasets/theor.csv'), index_col=0)
theor = theor.iloc[1:,]
theor_arr=theor.values
#print(theor)
# Normalize data for training
ntheor = normdata(theor_arr)

# Load labels for simulated data
label_theo = pd.read_csv(os.path.join(dirname, 'Datasets/label_theo.csv'), header=None, index_col=0)
label_theo = label_theo[1].tolist()

# Load experimental data as dataframe
exp_arr_new = pd.read_csv(os.path.join(dirname, 'Datasets/exp.csv'), index_col=0)
exp_arr_new = exp_arr_new.values

# Load experimental class labels
label_exp= pd.read_csv(os.path.join(dirname, 'Datasets/label_exp.csv'), index_col=0).values
label_exp = label_exp.reshape([len(label_exp),])

# Load class enconding
space_group_enc = pd.read_csv(os.path.join(dirname, 'Datasets/encoding.csv'), index_col=0)
space_group_enc = list(space_group_enc['0'])

# Normalize experimental data
nexp = normdata(exp_arr_new)

# Define spectral range for data augmentation
exp_min = 0
exp_max = 1200 
theor_min = 125

#window size for experimental data extraction
window = 20
theor_max = theor_min+exp_max-exp_min

# Preprocess experimental data
post_exp = normdatasingle(exp_data_processing (nexp, exp_min, exp_max, window))

################################################################
# Perform data augmentation
################################################################

# Specify how many data points we augmented
th_num = 2000

# Augment data, this may take a bit
augd,pard,crop_augd = augdata(ntheor, th_num, label_theo, theor_min, theor_max)    

# Enconde theoretical labels
label_t=np.zeros([len(pard),])
for i in range(len(pard)):
    label_t[i]=space_group_enc.index(pard[i])

# Input the num of experimetal data points       
exp_num =88

# Prepare experimental arrays for training and testing
X_exp = np.transpose(post_exp[:,0:exp_num])
y_exp = label_exp[0:exp_num]

# Prepare simulated arrays for training and testing
X_th = np.transpose(crop_augd )
y_th = label_t
# Create auxiliary arrays
accuracy_test=[]
accuracy_train=[]
logs=[]
ground_truth=[]
predictions_ord=[]
trains=[]
tests=[]
trains_combine=[]
trains_y=[]
matrix=[]
X_train=[]
X_test=[]
y_train=[]
y_test=[]
temp_x = X_exp
temp_y = y_exp
enc = OneHotEncoder(sparse=False)
exp_train_x,exp_train_y = exp_augdata(temp_x.T,5000,temp_y)
# Combine theoretical and experimenal dataset for training
train_combine = np.concatenate((X_th,exp_train_x.T))
trains_combine.append(train_combine)
train_combine = train_combine.reshape(train_combine.shape[0],1200,1)
train_y = np.concatenate((y_th,exp_train_y))
train_y = enc.fit_transform(train_y.reshape(-1,1))
X_train, X_test, y_train, y_test = train_test_split(train_combine, train_y, test_size=0.2, random_state=42)
# Define network structure
model = Sequential()

model.add(K.layers.Conv1D(32, 8,strides=8, padding='same',input_shape=(1200,1), activation='relu'))
model.add(Dropout(0.2))
model.add(K.layers.Conv1D(32, 5,strides=5, padding='same', activation='relu'))
model.add(Dropout(0.2))
model.add(K.layers.Conv1D(32, 3,strides=3, padding='same', activation='relu'))
model.add(K.layers.MaxPooling1D(pool_size=3, stride=1))
#model.add(K.layers.GlobalAveragePooling1D())
model.add(K.layers.LSTM(32,input_shape=X_train.shape,
               return_sequences=False))
model.add(K.layers.Dense(n_classes, activation='softmax'))
        
#Define optimizer        
optimizer = K.optimizers.Adam()
tensorboard = TensorBoard(log_dir='./logs', histogram_freq=0,
                          write_graph=True, write_images=False)   
# Compile model
model.compile(loss='categorical_crossentropy',
                      optimizer=optimizer,
                      metrics=['categorical_accuracy'])
# Fit model
hist = model.fit(X_train, y_train, batch_size=BATCH_SIZE, nb_epoch=100,
                         verbose=1, validation_data=(X_test, y_test),callbacks=[tensorboard])
prediction=model.predict(X_test)
 
#Go from one-hot to ordinal...
prediction_ord=[np.argmax(element) for element in prediction]
predictions_ord.append(prediction_ord)
matrix = metrics.confusion_matrix(y_test.argmax(axis=1), prediction.argmax(axis=1))

print(matrix)
#Compute loss and accuracy for each k validation
accuracy_test=model.evaluate(X_test, y_test, verbose=1)
accuracy_train=model.evaluate(X_train, y_train, verbose=1)
        
#Save logs
log = pd.DataFrame(hist.history)
logs.append(log)
print(accuracy_train[1], accuracy_test[1]) 
plt.plot(logs[0]['categorical_accuracy'], 'C2', label = 'train')
plt.plot(logs[0]['val_categorical_accuracy'], 'C3', label = 'test')
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend( loc='upper left')
plt.show()
