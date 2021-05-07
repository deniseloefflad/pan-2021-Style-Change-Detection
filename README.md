# pan-2021-Style-Change-Detection
Participation in PAN 2021 Task: Style Change Detection

call: 
```
python lex-measures.py <path-to-folder> <path-to-validation> <path-to-embedding-dictionary>
```
e.g.
```
python lex-measures.py dataset-wide validation_dataset embedding-dict
```
# Dependencies
```
import os
import json
import nltk
import numpy as np
import sys
from keras.models import Sequential
from keras import layers
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from keras.utils import normalize
from sklearn.preprocessing import MinMaxScaler
from complexitymeasures import get_complexity_measures
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from keras.callbacks import EarlyStopping
import random
import gensim
import re
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec
from gensim.models import FastText
from gensim.models.keyedvectors import KeyedVectors
from gensim.models.wrappers import FastText
```

