import os
import json
import nltk
import numpy as np
import sys
from keras.models import Sequential
from keras import layers
from sklearn.model_selection import train_test_split, GridSearchCV
from keras.utils import normalize
from sklearn.utils import class_weight
from stylemeasures import get_complexity_measures
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from keras.callbacks import EarlyStopping
import random
import gensim
from nltk.tokenize import word_tokenize
from sklearn import svm
import fasttext.util
from sklearn.preprocessing import OneHotEncoder
from sklearn import preprocessing
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from imblearn.pipeline import make_pipeline
from sklearn.cluster import KMeans
from keras.models import load_model
import warnings
warnings.filterwarnings("ignore")


def lstm_task_3(train_x, train_y, val_x, val_y, test_x):
        """
        Build RNN model
        :param train_x: train x data
        :param train_y: train y data
        :param val_x: validation xdata
        :param val_y: validation y data
        :param test_x: test x data
        :return:
        predictions for dataset
        """
        train_x = np.array(train_x)
        train_y = np.array(train_y)
        test_x = np.array(test_x)
        val_x = np.array(val_x)
        val_y = np.array(val_y)
        shape = train_x.shape[1:]
        number_labels = train_x.shape[1]

        #unique_labels = ?
        flattened_labels = train_y.flatten()
        #class_weights = class_weight.compute_sample_weight('balanced', unique_labels, flattened_labels)
        #sample_weights = np.array([class_weights[0] if x == 0 else class_weights[1] for x in flattened_labels])

        # shape can be one dimensional iff target is multi-author
        #sample_weights = sample_weights.reshape((train_x.shape[1], train_x.shape[0])).transpose()

        batch = 5

        es = EarlyStopping(monitor='val_loss', patience=3, verbose=1, mode="min", restore_best_weights=True)
        model = Sequential()
        model.add(layers.Masking(mask_value=0, input_shape=shape))
        model.add(layers.Bidirectional(layers.LSTM(128, return_sequences=True, return_state=False)))  # 128 internal units
        #model.add(layers.Dropout(0.5))
        model.add(layers.Bidirectional(layers.LSTM(128, return_sequences=True, return_state=False)))
        #model.add(layers.TimeDistributed(layers.Dense(number_labels, activation='softmax')))
        model.add(layers.Dense(number_labels, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam')  # , sample_weight_mode='temporal')
        model.summary()

        model.fit(train_x, train_y, validation_data=(val_x, val_y), epochs=1000, batch_size=batch, callbacks=[es],
                  verbose=2)  # ,
        # sample_weight=sample_weights)

        model_name = 'model_task3.h5'
        model.save(model_name)


def pipeline_task_3(train_x, train_y, val_x, val_y, test_x):
    train_x = np.array(train_x)
    train_y = np.array(train_y)
    test_x = np.array(test_x)
    val_x = np.array(val_x)
    val_y = np.array(val_y)
    shape = train_x.shape[1:]
    number_labels = train_x.shape[1]
    log_reg = LogisticRegression(multi_class='multinomial')
    flattened_train_x = train_x.flatten()
    flattened_train_y = train_y.flatten()
    flattened_test_x = test_x.flatten()
    flattened_val_x = val_x.flatten()
    flattened_val_x = val_y.flatten()
    model = Sequential()
    model.add(layers.Masking(mask_value=0, input_shape=shape))
    #model.add(layers.Flatten())
    #log_reg.fit(flattened_train_x, flattened_train_y)
    kmeans = KMeans()
    pipeline = make_pipeline(
        ('masking', layers.Masking(mask_value=0, input_shape=shape)),
        ('kmeans', kmeans),
        ('log_reg', log_reg),
        #('lstm', layers.Bidirectional(layers.LSTM(128, return_sequences=True, return_state=False))),
        ('dense', layers.Dense(number_labels, activation='softmax')),
    )
    model.add(pipeline)
    param_grid = dict(kmeans__n_clusters=range(2,100))
    grid_clf = GridSearchCV(model, param_grid, cv=2, verbose=1)
    grid_clf.fit(flattened_train_x, flattened_train_y)
    model_name = 'kmeans_pipeline_task3.h5'
    model.save(model_name)



def conv3d_model_task_3(train_x, train_y, val_x, val_y, test_x):
    train_x = np.array(train_x)
    train_y = np.array(train_y)
    test_x = np.array(test_x)
    val_x = np.array(val_x)
    val_y = np.array(val_y)
    shape = train_x.shape[1:]
    number_labels = train_x.shape[1]
    batch = 5

    es = EarlyStopping(monitor='val_loss', patience=3, verbose=1, mode="min", restore_best_weights=True)
    model = Sequential()
    model.add(layers.Masking(mask_value=0, input_shape=shape))
    model.add(layers.Conv3D(2, 3, input_shape=shape))  # 128 internal units
    model.add(layers.Dropout(0.5))
    model.add(layers.Conv3D(2, 3, input_shape=shape))
    # model.add(layers.TimeDistributed(layers.Dense(number_labels, activation='softmax')))
    model.add(layers.Dense(number_labels, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam')  # , sample_weight_mode='temporal')
    model.summary()

    model.fit(train_x, train_y, validation_data=(val_x, val_y), epochs=10, batch_size=batch, callbacks=[es],
              verbose=2)  # ,
    # sample_weight=sample_weights)

    model_name = 'conv3d_model_task3.h5'
    model.save(model_name)

def RNN_model(train_x, train_y, val_x, val_y, test_x):
    train_x = np.array(train_x)
    train_y = np.array(train_y)
    test_x = np.array(test_x)
    val_x = np.array(val_x)
    val_y = np.array(val_y)
    shape = train_x.shape[1:]
    time_steps = train_x.shape[1]
    es = EarlyStopping(monitor='val_loss', patience=3, verbose=1, mode="min", restore_best_weights=True)
    batch = 5
    model = Sequential([
        layers.Masking(mask_value=0, input_shape=shape),
        layers.SimpleRNN(1, input_shape=[None,1], return_sequences=True),
        layers.SimpleRNN(1, return_sequences=True, input_shape=[None, 1]),
        layers.TimeDistributed(layers.Dense(time_steps, activation='softmax'))
    ])
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    model.fit(train_x, train_y, validation_data=(val_x, val_y), epochs=10, batch_size=batch, callbacks=[es],
              verbose=2)
    model_name = 'rnn_model_task3.h5'
    model.save(model_name)

