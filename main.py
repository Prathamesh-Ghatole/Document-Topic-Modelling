import numpy as np
import pandas as pd
import pickle
import colorama
import gensim
import nltk
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import CountVectorizer

# Download essential components for nltk
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('omw-1.4')

# Importing custom library functions
from lib.functions import *