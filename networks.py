import numpy as np
from keras.models import Sequential
from keras import layers
from sklearn.model_selection import train_test_split, GridSearchCV
from keras.utils import normalize
from sklearn.utils import class_weight
from sklearn.utils.multiclass import unique_labels
from tensorflow.python.keras.layers.core import Masking
from tensorflow.python.keras.layers.recurrent_v2 import LSTM
from stylemeasures import get_complexity_measures
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from keras.callbacks import EarlyStopping
from nltk.tokenize import word_tokenize
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from imblearn.pipeline import make_pipeline
from sklearn.cluster import KMeans
from keras.models import load_model
import tensorflow as tf
from tensorflow.keras import Model
import warnings
warnings.filterwarnings("ignore")


# class MultiAuthorModel(Model):

#   def __init__(self):
#     super(MultiAuthorModel, self).__init__()
#     self.masking = Masking()
#     self.rnn1 = LSTM(128, activation='tanh')
#     self.flatten = Flatten()
#     self.d1 = Dense(128, activation='relu')
#     self.d2 = Dense(10)

#   def call(self, x):
#     x = self.conv1(x)
#     x = self.flatten(x)
#     x = self.d1(x)
#     return self.d2(x)

# Create an instance of the model

def calc_class_weights(x, y):
    class_weights = class_weight.compute_class_weight('balanced', np.unique(y), y)

    sample_weights = np.array([class_weights[0] if label == 0 else class_weights[1] for label in y])

    # shape can be one dimensional iff target is multi-author
    sample_weights = sample_weights.reshape((x.shape[1], x.shape[0])).transpose()
    return sample_weights


def model_task_1(train_x, train_y, padded_val_x, padded_val_y, test_x):
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
    train_ds = tf.data.Dataset.from_tensor_slices((train_x, train_y))
    val_ds = tf.data.Dataset.from_tensor_slices((padded_val_x, padded_val_y))

    for i in train_ds:
        print(i)

    train_x.shape

    train_x = np.array(train_x, dtype=float)
    train_y = np.array(train_y)
    padded_val_x = np.array(padded_val_x)
    padded_val_y = np.array(padded_val_y)
    test_x = np.array(test_x)

    # flattened_labels = train_y.flatten()
    # print(train_y)
    # print(flattened_labels)
    # class_weights = calc_class_weights(train_x, train_y)
    shape = (8, 5)
    print(train_x.shape)
    print(train_ds._flat_shapes)


    es = EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode="min", restore_best_weights=True)
    model = Sequential()

    model.add(layers.Dense(350, input_shape=train_x.shape, activation='relu'))
    model.add(layers.Dense(50, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
   
    # # model.add(layers.Masking(mask_value=0, input_shape=train_x.shape))
    # model.add(layers.LSTM(128, return_sequences=True, return_state=False, input_shape=shape))  # 128 internal units
    # model.add(layers.Dense(1, activation='sigmoid'))
    
    model.compile(loss='binary_crossentropy', optimizer='adam')  # , sample_weight_mode='temporal')
    model.summary()

    model.fit(train_x, train_y, validation_data=(padded_val_x, padded_val_y), epochs=1000, callbacks=[es], verbose=2)

    predictions = model.predict(test_x)
    print(predictions)

    model_name = 'model_task1.h5'
    model.save(model_name)





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
    model.summary()

    model.fit(train_x, train_y, validation_data=(val_x, val_y), epochs=1000, batch_size=batch, callbacks=[es],
              verbose=2)  # ,
    # sample_weight=sample_weights)
    model_name = 'model_task2.h5'
    model.save(model_name)


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
        model.add(layers.Dropout(0.5))
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


def get_predictions(task, test_data):
    model_name = 'model_task1.h5' if task=='task-1' else 'model_task2.h5' if task=='task-2' else 'rnn_model_task3.h5'
    #loaded_model = pickle.load(open(model_name, 'rb'))
    loaded_model = load_model(model_name)
    predictions_probs = loaded_model.predict(test_data)
    predictions = []
    if task == 'task-1':
        return
    elif task == 'task-2':
        for pred in predictions_probs:
            pred_lst = [[1] if x >= 0.5 else [0] for x in pred]
            predictions.append(pred_lst)
    elif task == 'task-3':
        for probs in predictions_probs:
            authors_text = []
            for par in probs:
                most_prob_author_par = np.argmax(par) + 1
                authors_text.append(most_prob_author_par)
            predictions.append(authors_text)
    return predictions


def evaluate(predictions, test_y, scores, multi):
    """
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
    if task == 'task-1':
        model_task_1(x_train, y_train, padded_val_x, padded_val_y, x_test)
        multiple_classes = False
    elif task == 'task-2':
        lstm_task_2(x_train, y_train, padded_val_x, padded_val_y, x_test)
        multiple_classes = False
    elif task == 'task-3':
        #y_train_encoded = one_hot_encoding(y_train, encoder)
        #padded_val_encoded = one_hot_encoding(padded_val_y, encoder)
        y_train_encoded = manual_encoding(y_train)
        padded_val_encoded = manual_encoding(padded_val_y)
        #lstm_task_3(x_train, y_train_encoded, padded_val_x, padded_val_encoded, x_test)
        # pipeline_task_3(x_train, y_train_encoded, padded_val_x, padded_val_encoded, x_test)
        # RNN_model(x_train, y_train_encoded, padded_val_x, padded_val_encoded, x_test)
        RNN_model(x_train, y_train_encoded, padded_val_x, padded_val_encoded, x_test)
        multiple_classes = True

    predictions = get_predictions(task, x_test)
    evaluations = evaluate(predictions, y_test, scores, multiple_classes)
    return evaluations


def manual_encoding(data):
    encoded_data = []
    max_len = len(data[0])
    for arrs in data:
        encoded_text = []
        for label in arrs:
            num_authors = int(label[0])
            encoded_par = np.zeros(max_len)
            if not num_authors == 0:
                encoded_par[num_authors] = 1
            encoded_text.append(encoded_par)
        encoded_data.append(encoded_text)
    return encoded_data
