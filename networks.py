import os
import json
import nltk
import numpy as np
import sys
from keras.models import Sequential
from keras import layers
from sklearn.model_selection import train_test_split
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
import warnings
warnings.filterwarnings("ignore")


def majority_baseline(train_labels, val_labels):
    """
    :param train_labels: train labels
    :param val_labels: test labels
    :return:
    accuracy score for majority baseline
    """
    majority_class = np.argmax(np.bincount(np.array(train_labels).flatten()))
    counts = np.bincount(np.array(val_labels).flatten())
    majority_score = counts[majority_class]
    score = majority_score / (counts[0] + counts[1])
    return score


def random_baseline(val_labels):
    """
    :param val_labels: test labels
    :return:
    accuracy for random baseline
    """
    val_labels = np.array(val_labels)
    random_values = [random.randint(0, 1) for x in range(len(val_labels.flatten()))]
    counts = np.bincount(val_labels.flatten())
    counts_preds = np.bincount(random_values)
    tp = (counts[0] if counts[0] <= counts_preds[0] else counts_preds[0])
    tn = (counts[1] if counts[1] <= counts_preds[1] else counts_preds[1])
    fp = (counts_preds[0] - counts[0] if counts_preds[0] > counts[0] else 0)
    fn = (counts_preds[1] - counts[1] if counts_preds[1] > counts[1] else 0)
    accuracy_random = (tp + tn) / (tp + tn + fp + fn)
    return accuracy_random


def train_classifier(train_x, train_y, val_x, val_y, test_x):
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
    model.summary()

    model.fit(train_x, train_y, validation_data=(val_x, val_y), epochs=1000, batch_size=batch, callbacks=[es],
              verbose=2)  # ,
    # sample_weight=sample_weights)
    predictions_probs = model.predict(test_x)
    predictions = []
    # print(f"batch size {batch}")
    for pred in predictions_probs:
        pred_lst = [[1] if x >= 0.5 else [0] for x in pred]
        predictions.append(pred_lst)
    return predictions


def train_svm(train_x, train_y, val_x, val_y, test_x):
    train_x = np.array(train_x)
    train_y = np.array(train_y)
    test_x = np.array(test_x)
    val_x = np.array(val_x)
    val_y = np.array(val_y)
    shape = train_x.shape[1:]

    unique_labels = np.array([0, 1])
    flattened_labels = train_y.flatten()
    batch = 1
    svm_classifier = svm.SVC(decision_function_shape='ovo', kernel='sigmoid')

    es = EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode="min", restore_best_weights=True)
    #model = Sequential()
    #model.add(layers.Masking(mask_value=0, input_shape=shape))
    #model.add()

    svm_classifier.fit(train_x, train_y)
    predictions = svm_classifier.predict(test_x)
    return predictions

def evaluate(predictions, test_y, scores):
    """
    :param predictions: array of predictions
    :param test_y : labels data
    :param scores: list of scores to evaluate (f1, accuracy...)
    :return:
    eval_scores: precision, recall, f_1 and accuracy score
    """
    eval_scores = []
    for score in scores:
        eval_scores.append(score(np.array(predictions).flatten(), np.array(test_y).flatten()))
    return eval_scores


def get_evaluation(padded_x, padded_val_x, padded_val_y, padded_labels_style_change, scores, task, random_state=0):
    """
    Split data, train classifier, get predictions & calculate evaluation scores
    :param task:
    :param padded_x: padded x data
    :param padded_val_x: padded validation x data
    :param padded_val_y: padded validation y data
    :param padded_labels_style_change: padded labels
    :param scores: list of scores for evaluation
    :return:
    evluation scores
    """
    x_train, x_test, y_train, y_test = train_test_split(padded_x, padded_labels_style_change, test_size=0.2,
                                                        random_state=random_state)

    if task == 'task-2':
        predictions = train_classifier(x_train, y_train, padded_val_x, padded_val_y, x_test)
    elif task == 'task-3':
        predictions = train_svm(x_train, y_train, padded_val_x, padded_val_y, x_test)

    evaluations = evaluate(predictions, y_test, scores)
    return evaluations


def baselines(padded_labels_style_change):
    y_train, y_test = train_test_split(padded_labels_style_change, test_size=0.2, random_state=42)
    majority_baseline_scores = majority_baseline(y_train, y_test)
    random_baseline_acc = random_baseline(y_test)
    return majority_baseline_scores, random_baseline_acc