import os
import json
import nltk
import numpy as np
import sys
import networks
from sklearn.model_selection import train_test_split
from keras.utils import normalize
from stylemeasures import get_complexity_measures
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import gensim
from sklearn.model_selection import train_test_split
from nltk.tokenize import word_tokenize

#import fasttext.util
import warnings
warnings.filterwarnings("ignore")
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')


def get_labels(filename):
    """
    :param filename:
    :return:
    """
    number_of_authors, structure, changes = list(), list(), list()
    with open(filename) as json_file:
        data = json.load(json_file)
        number_of_authors.append(data['authors'])
        structure = data['structure']
        changes = [1]  # set that 1 paragraph is always treated like a switch
        changes = changes + data['changes']
        changes = [[x] for x in changes]
        authors_structure = data['paragraph-authors']
        labels = number_of_authors, structure, changes, authors_structure

    return labels


def get_word_embeddings(line):
    """
    :param line: string
    :return:
    embedding for the string
    """
    line = line.replace("/", " / ")
    paragraph = word_tokenize(line)  # tokenize paragraph
    words = [word for word in paragraph if
             word in embedding_model.vocab]  # remove out-of-vocabulary words from pretrained embeddings
    c = np.zeros(300, dtype=int)
    if words:
        return np.mean(embedding_model[words], axis=0)
    else:
        return c


def read_data(folder):
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
            file = [] #store paragraph embeddings
            filename = os.path.join(folder, filename)
            with open(filename, 'r', encoding='utf-8') as f:
                complexity_measures_text = []
                text = f.readlines()
                text_ids = []
                for line in text:
                    if line.strip():
                        complexity_measures_par = get_complexity_measures(line)
                        complexity_measures_text.append(complexity_measures_par)
                        text_ids.append(i)
                        if line:
                            par_embedding = get_word_embeddings(line)
                        file.append(par_embedding)
                embedding_all_docs.append(file)

                if complexity_measures_text:
                    complexity_measures_all_docs.append(complexity_measures_text)
                    all_text_ids.append(text_ids)
                    i += 1
                filename_json = os.path.join(folder, 'truth-' + text_filename.replace('txt', 'json'))
                labels_json = get_labels(filename_json)
                labels.append(labels_json)
    return complexity_measures_all_docs, all_text_ids, labels, embedding_all_docs


def pad_embeddings(files, validations_x, validation_labels):
    """
    :param files:
    :param validations_x:
    :param validation_labels:
    :return:
    """
    padded_val_x = []
    padded_val_y = []
    max_x = len(max(files, key=len))
    max_val = len(max(validations_x, key=len))
    longest_file = max(max_x, max_val)
    c = np.zeros(300, dtype=int)

    for file in files:
        pad_num = longest_file - len(file)
        for i in range(pad_num):
            file.append(c)

    for i, elem in enumerate(validations_x):
        elem += (longest_file - len(elem)) * [c]
        elem_y = validation_labels[i] + (longest_file - len(validation_labels[i])) * [[0]]
        padded_val_x.append(elem)
        padded_val_y.append(elem_y)

    return np.array(files), padded_val_x, padded_val_y


def padding(x, labels, val_x, val_labels):
    """
    padding data
    :param x: x data
    :param labels: labels
    :return:
    padded x & y data
    """
    pad_compl_measures = [0, 0, 0, 0, 0]
    pad_style_change = [0]
    padded_x = []
    padded_y = []
    padded_val_x = []
    padded_val_y = []

    max_x = len(max(x, key=len))
    max_val = len(max(val_x, key=len))
    max_len = max(max_x, max_val)

    for i, elem in enumerate(x):
        elem += (max_len - len(elem)) * [pad_compl_measures]
        elem_y = labels[i] + (max_len - len(labels[i])) * [pad_style_change]
        padded_x.append(elem)
        padded_y.append(elem_y)

    for i, elem in enumerate(val_x):
        elem += (max_len - len(elem)) * [pad_compl_measures]
        elem_y = val_labels[i] + (max_len - len(val_labels[i])) * [pad_style_change]
        padded_val_x.append(elem)
        padded_val_y.append(elem_y)
    return padded_x, padded_y, padded_val_x, padded_val_y


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











if __name__ == "__main__":
    scores = (precision_score, recall_score, f1_score, accuracy_score)
    validation = True


    #fasttext.util.download_model('en', if_exists='ignore')  # English
    #ft = fasttext.load_model('cc.en.300.bin')
    #print(ft)

    if len(sys.argv) < 4:
        raise TypeError("Please enter the path to a dataset, validation & the embedding dict as input argument!")


    


    folder = sys.argv[1]
    validation_folder = sys.argv[2]
    embeddings_dict = sys.argv[3]

    np.set_printoptions(threshold=np.inf)
    embedding_model = gensim.models.KeyedVectors.load_word2vec_format(embeddings_dict, binary=False, limit=200000)
    print("--------downloaded---------")
    compl_measures_all, text_ids, labels, embedding_all_docs = read_data(folder)
    print("-------read folder -------------")
    val_compl_measures, val_text_ids, val_labels, val_embedding_all_docs = read_data(validation_folder)
    print("-------read validation -------------")
    val_labels_authors = [item[3] for item in val_labels]
    labels_authors = [item[3] for item in labels]

    val_labels = [item[2] for item in val_labels]
    labels_style_change = [item[2] for item in labels]  # sc = style change


    #----------------- complexity measures


    def get_results_compl(compl_measures, labs, val_measures, val_labs, scrs):
        """

        :return:
        """
        padded_compl_measures, padded_labels, padded_val_x, padded_val_y = padding(compl_measures, labs,
                                                                                            val_measures, val_labs)
        normalized_compl = normalize(padded_compl_measures, axis=2, order=2)
        normalied_validation = normalize(padded_val_x, axis=2, order=2)
        eval_scores_compl = networks.get_evaluation(normalized_compl, normalied_validation, padded_val_y, padded_labels,
                                       scrs)
        return eval_scores_compl, padded_labels, padded_val_y, normalized_compl, normalied_validation

    #eval_compl_style_change = get_results_compl(compl_measures_all, labels_style_change, val_compl_measures, val_labels, scores)
    eval_compl_authors, padded_labels, padded_val_y, normalized_compl, normalied_validation = get_results_compl(compl_measures_all, labels_authors, val_compl_measures, val_labels_authors,
                                                scores)

    #----------------- embeddings
    def get_results_embeddings(embedding_docs, val_embs, val_labs, padded_labels, scrs):
        padded_embeddings, padded_val_x_embeddings, padded_val_y_embeddings = pad_embeddings(embedding_docs, val_embs,
                                                                                             val_labs)

        normalized_embeddings = normalize(padded_embeddings, axis=2, order=2)
        normalized_val_emb = normalize(padded_val_x_embeddings, axis=2, order=2)
        eval_scores_embeddings = networks.get_evaluation(normalized_embeddings, normalized_val_emb, padded_val_y_embeddings,
                                                scrs, padded_labels)
        return eval_scores_embeddings, normalized_embeddings, normalized_val_emb

    #eval_scores_embeddings = get_results_embeddings(embedding_all_docs, val_embedding_all_docs, val_labels, scores)
    eval_embeddings_authors, normalized_embeddings, normalized_val_emb = get_results_embeddings(embedding_all_docs, val_embedding_all_docs, val_labels_authors,
                                                     scores, padded_labels)

    #----------------- combined complexity feats & embeddings
    def get_results_combined(normalized_embeddings,normalized_val_emb,normalized_compl,normalied_validation,padded_labels, padded_val_y, scrs):
        combined_x = np.concatenate((np.array(normalized_embeddings), np.array(normalized_compl)), axis=2)
        combined_val = np.concatenate((np.array(normalized_val_emb), np.array(normalied_validation)), axis=2)
        combined_scores = networks.get_evaluation(combined_x, combined_val, padded_val_y, padded_labels, scrs)
        return combined_scores

    #eval_combined_style_change = get_results_embeddings(padded_val_y, padded_labels, scores)
    eval_combined_authors = get_results_combined(normalized_embeddings, normalized_val_emb, normalized_compl, normalied_validation,
                                                 padded_labels, padded_val_y, scores)


    #----------------- baselines
    y_train, y_test = train_test_split(padded_labels, test_size=0.2, random_state=42)
    majority_baseline_scores = networks.majority_baseline(y_train, y_test)
    random_baseline_acc = networks.random_baseline(y_test)

    #----------------- print results
    print("results complexity measures precision, recall, f1, accuracy: " + str(eval_compl_authors))
    print("results embeddings precision, recall, f1, accuracy: " + str(eval_embeddings_authors))
    print("results combined precision, recall, f1, accuracy: " + str(eval_combined_authors))
    print("majority baseline accuracy: " + str(majority_baseline_scores))
    print("random baseline accuracy: " + str(random_baseline_acc))
