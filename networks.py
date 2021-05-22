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


def model_task_1(x_train, y_train, padded_val_x, padded_val_y, x_test):
    pass


def lstm_task_2(train_x, train_y, val_x, val_y, test_x):
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

    unique_labels = np.array([0, 1])
    flattened_labels = train_y.flatten()
    class_weights = class_weight.compute_sample_weight('balanced', unique_labels, flattened_labels)
    sample_weights = np.array([class_weights[0] if x == 0 else class_weights[1] for x in flattened_labels])

    # shape can be one dimensional iff target is multi-author
    sample_weights = sample_weights.reshape((train_x.shape[1], train_x.shape[0])).transpose()

    batch = 1

    es = EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode="min", restore_best_weights=True)
    model = Sequential()
    model.add(layers.Masking(mask_value=0, input_shape=shape))
    model.add(layers.Bidirectional(layers.LSTM(128, return_sequences=True, return_state=False)))  # 128 internal units
    model.add(layers.TimeDistributed(layers.Dense(1, activation='sigmoid')))
    model.compile(loss='binary_crossentropy', optimizer='adam')  # , sample_weight_mode='temporal')

    model.fit(train_x, train_y, validation_data=(val_x, val_y), epochs=1000, batch_size=batch, callbacks=[es],
              verbose=2)  # ,
    # sample_weight=sample_weights)
    return model


def model_task3(preds, train_x, model):
    """
    :param preds: set of predictions
    :param train_x: train set
    :param model: trained model
    :return:
    predicted authors and trained model
    """
    time_steps = train_x.shape[1]
    padding_size = train_x.shape[2]
    c = np.zeros(padding_size, dtype=int).tolist()
    authors_texts = []
    for z, text in enumerate(preds):
        authors = []
        for i, paragraph in enumerate(text):
            if i == 0: authors.append(1)
            elif paragraph[0] == 0: authors.append(authors[i-1])
            else:
                j = 0
                data = []
                text = []
                text.append(train_x[z][j].tolist())
                text.append(train_x[z][i].tolist())
                text = text + (time_steps - 2) * [c]
                data.append(text)
                while(model.predict(data)[0][1] >= 0.5) and i < j:
                    j += 1
                    data = []
                    text = []
                    text.append(train_x[z][j].tolist())
                    text.append(train_x[z][i].tolist())
                    text = text + (time_steps - 2) * [c]
                    data.append(text)
                if i == j:
                    new_author = len(list(set(authors)))
                    authors.append([new_author])
                else: authors.append(authors[j])
        authors_texts.append(authors)
    return authors_texts, model


def get_predictions(model, task, test_data):
    """
    :param model: trained model
    :param task: string that indicates task
    :param test_data:
    :return:
    predictions
    """
    predictions_probs = model.predict(test_data)
    predictions = []
    if task == 'task-1':
        return
    elif task == 'task-2' or task == 'task-3':
        for pred in predictions_probs:
            pred_lst = [[1] if x >= 0.5 else [0] for x in pred]
            predictions.append(pred_lst)
    return predictions


def evaluate(predictions, test_y, scores, multi):
    """
    :param multi: Bool whether multi class task or not
    :param predictions: array of predictions
    :param test_y : labels data
    :param scores: list of scores to evaluate (f1, accuracy...)
    :return:
    eval_scores: precision, recall, f_1 and accuracy score
    """
    eval_scores = []
    setting = 'macro' if multi else 'binary'

    for score in scores:
        if score != accuracy_score:
            eval_scores.append(score(np.array(predictions).flatten(), np.array(test_y).flatten(), average=setting))
        else: eval_scores.append(score(np.array(predictions).flatten(), np.array(test_y).flatten()))
    return eval_scores


def get_evaluation(x, val_x, val_y, labels, scores, task, model=None, random_state=None):
    """
    Split data, train classifier, get predictions & calculate evaluation scores
    :param model:
    :param random_state:
    :param x: padded x data
    :param val_x: padded validation x data
    :param val_y: padded validation y data
    :param labels: padded labels
    :param task: string to indicate task 1, 2 or 3
    :param scores: list of scores for evaluation
    :return:
    evluation scores
    """
    x_train, x_test, y_train, y_test = train_test_split(x, labels, test_size=0.2, random_state=random_state)
    if task == 'task-1':
        model = model_task_1(x_train, y_train, val_x, val_y, x_test)
        multiple_classes = False
    elif task == 'task-2':
        model = lstm_task_2(x_train, y_train, val_x, val_y, x_test)
        multiple_classes = False
    elif task == 'task-3':
        multiple_classes = True
    predictions = get_predictions(model, task, x_test)
    if task == 'task-3':
        predictions, model = model_task3(predictions, x_train, model)
    evaluations = evaluate(predictions, y_test, scores, multiple_classes)
    return evaluations, model
