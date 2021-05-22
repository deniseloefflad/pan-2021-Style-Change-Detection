import os
import json
import nltk
import numpy as np
import sys
from keras.utils import normalize
from stylemeasures import get_complexity_measures
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import gensim
from networks import get_evaluation#, lstm_task_2
from nltk.tokenize import word_tokenize
from keras.models import load_model
#import fasttext.util
import warnings
warnings.filterwarnings("ignore")
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

np.set_printoptions(threshold=np.inf)

scores = (precision_score, recall_score, f1_score, accuracy_score)
# global parameters
embedding_model = None
compl_measures_all = text_ids = labels = embedding_all_docs = None
val_compl_measures = val_text_ids = val_labels = val_embedding_all_docs = None


def get_labels(filename, return_labels = None):
    """
    :param filename:
    :return:
    """
    number_of_authors, structure, changes = list(), list(), list()
    labels = {}
    with open(filename) as json_file:
        data = json.load(json_file)
        # process data and give correct format
        changes = [1]  # set that 1 paragraph is always treated like a switch
        changes = changes + data['changes']
        changes = [[x] for x in changes]

        author_par = data['paragraph-authors']
        author_par = [[x] for x in author_par]

        number_of_authors.append(data['authors'])
        data['authors'] = number_of_authors
        # return all data or requested data
        if return_labels == None:
            data['changes'] = changes
            data['paragraph-authors'] = author_par
            return data
        else:      
            for label in return_labels: 
                labels[label] = data[label]
            return labels


def get_word_embeddings(line, embedding_model):
    """
    :param embedding_model:
    :param line: string
    :return:
    embedding for the string
    """
    line = line.replace("/", " / ")
    paragraph = word_tokenize(line)  # tokenize paragraph
    words = [word for word in paragraph if
             word in embedding_model.index_to_key]  # remove out-of-vocabulary words from pretrained embeddings
    c = np.zeros(300, dtype=float)
    if words:
        return np.array(np.mean(embedding_model[words], axis=0), dtype=float)
    else:
        return c


def read_data(folder, embedding_model, target):
    """get complexity measures for every paragraph & labels
    :return:
    complexity_measures_all_docs: array with complexity measures for all texts, for all paragraphs
    all_text_ids
    labels
    """
    all_text_ids = []
    i = 0
    complexity_measures_all_docs = []
    embedding_all_docs = []
    files = os.listdir(folder)
    labels = []
    for filename in files:
        if ('json' not in filename) and ('txt' in filename):
            text_filename = filename
            file = []
            filename = os.path.join(folder, filename)
            with open(filename, 'r', encoding='utf-8') as f:
                complexity_measures_text = []
                text = f.readlines()
                text_ids = []

                if target == 'multi-author':
                    f.seek(0)
                    embedding_all_docs.append(get_word_embeddings(f.read(), embedding_model))

                for line in text:
                    if line.strip():
                        complexity_measures_par = get_complexity_measures(line)
                        complexity_measures_text.append(complexity_measures_par)
                        text_ids.append(i)
                        if embedding_model != None and target != 'multi-author':
                            if line:
                                par_embedding = get_word_embeddings(line, embedding_model)
                            file.append(par_embedding)
                
                if embedding_model != None and target != 'multi-author':
                    embedding_all_docs.append(file)

                if complexity_measures_text:
                    complexity_measures_all_docs.append(complexity_measures_text)
                    all_text_ids.append(text_ids)
                    i += 1
                filename_json = os.path.join(folder, 'truth-' + text_filename.replace('txt', 'json'))
                labels_json = get_labels(filename_json)
                labels.append(labels_json)
    return complexity_measures_all_docs, all_text_ids, labels, embedding_all_docs


# anonymous function to pad a given input and labels
def _pad(x, labels, padding_x, padding_y, pad_len, target):
    padded_x = []
    padded_y = []
    elem_y = []
    for i, elem in enumerate(x):
        start_size = len(elem)
        elem += (pad_len - start_size) * [padding_x]
        if target == 'multi-author':
            elem_y = labels[i] #* start_size + (pad_len - start_size) * [padding_y]
        elif (target == 'changes') or (target == 'paragraph-authors'):
            elem_y = labels[i] + (pad_len - len(labels[i])) * [padding_y]
        padded_x.append(elem)
        padded_y.append(elem_y)
    return padded_x, padded_y


def pad_embeddings(files, validations_x, validation_labels, target):
    """
    :param target:
    :param files:
    :param validations_x:
    :param validation_labels:
    :return:
    """
    max_x = len(max(files, key=len))
    max_val = len(max(validations_x, key=len))
    longest_file = max(max_x, max_val)
    c = np.zeros(300, dtype=int)

    for file in files:
        pad_num = longest_file - len(file)
        for i in range(pad_num):
            file.append(c)

    padded_val_x, padded_val_y = _pad(validations_x, validation_labels, c, [0], longest_file, target)

    return np.array(files), padded_val_x, padded_val_y


def padding(x, labels, val_x, val_labels, target):
    """
    padding data
    :param x: x data
    :param labels: labels
    :return:
    padded x & y data
    """
    pad_compl_measures = [0, 0, 0, 0, 0]
    pad_style_change = [0]

    max_x = len(max(x, key=len))
    max_val = len(max(val_x, key=len))
    max_len = max(max_x, max_val)

    padded_x, padded_y = _pad(x, labels, pad_compl_measures, pad_style_change, max_len, target)
    padded_val_x, padded_val_y = _pad(val_x, val_labels, pad_compl_measures, pad_style_change, max_len, target)

    return padded_x, padded_y, padded_val_x, padded_val_y


def initialize_global_params(data_folder, validation_folder, embeddings_dict):
    global embedding_model

    if embedding_model == None:
        embedding_model = gensim.models.KeyedVectors.load_word2vec_format(embeddings_dict, binary=False, limit=200000)

    global compl_measures_all, text_ids, labels, embedding_all_docs

    if (compl_measures_all and text_ids and labels and embedding_all_docs) == None:
        compl_measures_all, text_ids, labels, embedding_all_docs = read_data(data_folder, embedding_model)

    global val_compl_measures, val_text_ids, val_labels, val_embedding_all_docs

    if (val_compl_measures and val_text_ids and val_labels and val_embedding_all_docs) == None:
        val_compl_measures, val_text_ids, val_labels, val_embedding_all_docs = read_data(validation_folder,
                                                                                         embedding_model)


def task_1(folder, validation_folder, embeddings_dict):
    task = 'task-1'
    predictions = []

    global embedding_model
    global compl_measures_all, text_ids, labels, embedding_all_docs
    global val_compl_measures, val_text_ids, val_labels, val_embedding_all_docs

    initialize_global_params(folder, validation_folder, embeddings_dict)

    target = 'multi-author'

    print("--------downloaded---------")
    _train_compl_measures, _text_ids, train_y, train_x_emb = read_data(folder, embedding_model, target)
    print("-------read folder -------------")
    _val_compl_measures, _val_text_ids, val_y, val_x_emb = read_data(validation_folder, embedding_model, target)

    val_y = [item[target] for item in val_y]
    train_y = [item[target] for item in train_y]  # sc = style change

    train_x_emb = normalize(train_x_emb, axis=1, order=2)
    val_x_emb = normalize(val_x_emb, axis=1, order=2)

    embedding_scores, predictions, model = get_evaluation(train_x_emb, val_x_emb, val_y, train_y, scores, task)

    model.save('model_task1.h5')

    #----------------- print results
    print("results embeddings precision, recall, f1, accuracy: " + str(embedding_scores))

    return predictions


def task_2(folder, validation_folder, embeddings_dict):

    task = 'task-2'

    global embedding_model
    global compl_measures_all, text_ids, labels, embedding_all_docs
    global val_compl_measures, val_text_ids, val_labels, val_embedding_all_docs

    initialize_global_params(folder, validation_folder, embeddings_dict)

    target = 'changes'
    val_labels = [item[target] for item in val_labels]
    labels_style_change = [item[target] for item in labels]  # sc = style change

    #----------------- complexity measures
    padded_compl_measures, padded_labels, padded_val_x, padded_val_y = padding(compl_measures_all,
                                                                                            labels_style_change,
                                                                                            val_compl_measures,
                                                                                            val_labels, target)
    normalized_compl = normalize(padded_compl_measures, axis=2, order=2)
    normalied_validation = normalize(padded_val_x, axis=2, order=2)


    #----------------- embeddings
    padded_embeddings, padded_val_x_embeddings, padded_val_y_embeddings = pad_embeddings(embedding_all_docs,
                                                                                         val_embedding_all_docs,
                                                                                         val_labels,
                                                                                         target)

    normalized_embeddings = normalize(padded_embeddings, axis=2, order=2)
    normalized_val_emb = normalize(padded_val_x_embeddings, axis=2, order=2)


    #----------------- combined complexity feats & embeddings
    combined_x = np.concatenate((np.array(normalized_embeddings), np.array(normalized_compl)), axis=2)
    combined_val = np.concatenate((np.array(normalized_val_emb), np.array(normalied_validation)), axis=2)

    eval_scores_compl, predictions, model_compl = get_evaluation(normalized_compl, normalied_validation, padded_val_y,
                                                                 padded_labels, scores, task)
    eval_scores_embeddings, predictions, model_emb = get_evaluation(normalized_embeddings, normalized_val_emb,
                                                                    padded_val_y_embeddings,padded_labels, scores, task)
    combined_scores, predictions, model_combined = get_evaluation(combined_x, combined_val, padded_val_y, padded_labels,
                                                                  scores, task)

    model_compl.save('task2-complexity-model.h5')
    model_emb.save('task2-embeddings-model.h5')
    model_combined.save('task2-combined-model.h5')
    #----------------- print results
    print("results complexity measures precision, recall, f1, accuracy: " + str(eval_scores_compl))
    print("results embeddings precision, recall, f1, accuracy: " + str(eval_scores_embeddings))
    print("results combined precision, recall, f1, accuracy: " + str(combined_scores))
    return predictions


def task_3(folder, validation_folder, embeddings_dict):
    task = 'task-3'

    global embedding_model
    global compl_measures_all, text_ids, labels, embedding_all_docs
    global val_compl_measures, val_text_ids, val_labels, val_embedding_all_docs

    initialize_global_params(folder, validation_folder, embeddings_dict)

    target = 'paragraph-authors'
    val_labels = [item[target] for item in val_labels]
    labels_authors = [item[target] for item in labels]

    model_compl = load_model('task2-complexity-model.h5')
    model_emb = load_model('task2-embeddings-model.h5')
    model_combined = load_model('task2-combined-model.h5')

    #pad complexity measures, pad embeddings and pad validation data
    padded_measures, padded_labels, padded_val_x, padded_val_y = padding(compl_measures_all, labels_authors,
                                                                                            val_compl_measures,
                                                                                            val_labels, target)
    normalized_compl = normalize(padded_measures, axis=2, order=2)
    normalied_validation = normalize(padded_val_x, axis=2, order=2)

    padded_embeddings, padded_val_x_embeddings, padded_val_y_embeddings = pad_embeddings(embedding_all_docs,
                                                                                         val_embedding_all_docs,
                                                                                         val_labels,
                                                                                         target)

    normalized_embeddings = normalize(padded_embeddings, axis=2, order=2)
    normalized_val_emb = normalize(padded_val_x_embeddings, axis=2, order=2)

    combined_x = np.concatenate((np.array(normalized_embeddings), np.array(normalized_compl)), axis=2)
    combined_val = np.concatenate((np.array(normalized_val_emb), np.array(normalied_validation)), axis=2)

    complexity_scores, predictions, model = get_evaluation(normalized_compl, normalied_validation, padded_val_y,
                                                           padded_labels, scores, task, model_compl)
    embedding_scores, predictions, model = get_evaluation(normalized_embeddings, normalized_val_emb, padded_val_y,
                                                          padded_labels, scores, task, model_emb)
    combined_scores, predictions, model = get_evaluation(combined_x, combined_val, padded_val_y, padded_labels, scores,
                                                         task, model_combined)
    print('---------LSTM Task 3--------------')
    print("results complexity precision, recall, f1, accuracy: " + str(complexity_scores))
    print("results embeddings precision, recall, f1, accuracy: " + str(embedding_scores))
    print("results combined precision, recall, f1, accuracy: " + str(combined_scores))
    return predictions


def save_to_output_format(predicts_task_1, predicts_task_2, predicts_task_3, txt_ids):
    """
    :param predicts_task_1:
    :param predicts_task_2:
    :param predicts_task_3:
    :param txt_ids:
    :return:
    """
    filename_base = 'solution-problem-'
    file_ending = '.json'
    for num_id, spec_id in enumerate(txt_ids):
        filename = filename_base + str(spec_id) + file_ending
        print(filename)
        txt_solutions = {
            "multi-author": predicts_task_1[num_id],
            "changes": predicts_task_2[num_id],
            "paragraph-authors": predicts_task_3[num_id]
        }
        print(txt_solutions)

        with open(filename, 'w') as json_file:
            json.dump(txt_solutions, json_file)

# ---------------------------------------------MAIN--------------------------------------------------------------------


if __name__ == "__main__":
    if len(sys.argv) < 4:
        raise TypeError("Please enter the path to a dataset, validation & the embedding dict as input argument!")
    folder = sys.argv[1]
    validation_folder = sys.argv[2]
    embeddings_dict = sys.argv[3]
    predictions_task_1 = task_1()
    predictions_task_2 = task_2(folder, validation_folder, embeddings_dict)
    predictions_task_3 = task_3(folder, validation_folder, embeddings_dict)
    save_to_output_format(predictions_task_1, predictions_task_2, predictions_task_3, text_ids)
