from nltk.corpus import stopwords
from nltk import word_tokenize, Counter, pos_tag, sent_tokenize
from math import sqrt
import numpy as np
import re


stop_words = stopwords.words('english')
vowels = ['a', 'e', 'i', 'o', 'u', 'y']


def calc_cttr(text):  # removing sw here seems to harm the variability
    """
    :param text: plain text
    :return:
    corrected cttr for text
    """
    text = re.sub(r'[^\w]', ' ', text).lower()
    tokens = word_tokenize(text)
    tags = pos_tag(tokens)
    types = Counter(tag for word, tag in tags)
    if not ((len(tokens) == 0) or (sqrt(2 * len(tokens)) == 0)):
        corrected_ttr = (len(types) / (sqrt(2 * len(tokens))))
    else:
        corrected_ttr = 0
    return corrected_ttr


def function_word_frequency(text):
    """
    :param text: plain text
    :return:
    function word frequency
    """
    tokens = word_tokenize(text)
    function_words = [x for x in tokens if x in stop_words]
    num_function_words = len(function_words)
    num_all_tokens = len(tokens)
    return num_function_words / num_all_tokens


def count_syllables(token):
    """
    :param token: word token
    :return:
    int, number of syllables
    """
    syllable_count = 0
    j = 0
    for i, letter in enumerate(token):
        letter = letter.lower()
        if any(v in vowels for v in letter):
            if j == 0:
                #1st letter being a vowel
                syllable_count += 1
            if j == len(token)-1 and letter == 'e':
                #silent e
                syllable_count += 0
            elif j > 0 and not any(v in vowels for v in token[j-1].lower()):
                #no diphtong
                syllable_count += 1
        j += 1
    return syllable_count


def linsear_write_formula(text):
    """
    :param text: plain text
    :return:
    linsear write formula
    """
    limit_words = 100
    tokens = word_tokenize(text)
    num_sentences = len(sent_tokenize(text))
    number_syllables = []
    points = 0
    if len(tokens) > limit_words:
        first_hundred_tokens = tokens[:100]
    else:
        first_hundred_tokens = tokens
    for token in first_hundred_tokens:
        number_syllables.append(count_syllables(token))
    for syllable in number_syllables:
        if syllable <= 2:
            points += 1
        else:
            points += 3
    linsear_formula = points / num_sentences
    return linsear_formula


def calc_mean_sent_length(text):
    """
    :param text: plain text
    :return:
    mean sentence length for text
    """
    sentences = sent_tokenize(text)
    tokens_sentences = []
    for sent in sentences:  # keep stopwords as it might contain relevant information regarding to sentence length
        sent = re.sub(r'[^\w]', ' ', sent).lower()
        tokens = word_tokenize(sent)
        tokens_sentences.append(tokens)
    mean_sent_length = sum(len(token) for token in tokens_sentences)/len(sentences)
    return mean_sent_length


def calc_mean_word_length(text):
    """
    :param text: plain text
    :return:
    mean word length for text
    """
    text = re.sub(r'[^\w]', ' ', text).lower()
    tokens = word_tokenize(text)
    tokens_without_sw = [token for token in tokens if not token in stop_words]  # remove sw because mwl always the same
    if not len(tokens_without_sw) == 0:
        mean_word_length = sum(len(token) for token in tokens_without_sw)/len(tokens_without_sw)
    else:
        mean_word_length = 0
    return mean_word_length


def get_complexity_measures(line):
    """
    get complexity measures for every line
    :param line: txt to analyze
    :return:
    measures: list of 3 measures: cttr, mean & word length
    """
    cttr = calc_cttr(line)
    mean_sent_length = calc_mean_sent_length(line)
    mean_word_length = calc_mean_word_length(line)
    function_word_freq = function_word_frequency(line)
    linsear_formula = linsear_write_formula(line)
    measures = cttr, mean_sent_length, mean_word_length, function_word_freq, linsear_formula
    return np.array(measures)
