import pandas as pd
import numpy as np
import colorama
import pickle
import gensim
from sklearn.feature_extraction.text import CountVectorizer

# Importing custom library functions
from lib.functions import *

###################
# PART 1: Load Data from pickle, 

# Load the list of documents
with open('data/newsgroups', 'rb') as f:
    newsgroup_data = pickle.load(f)

# Use CountVectorizor to find three letter tokens, remove stop_words, 
# remove tokens that don't appear in at least 20 documents,
# remove tokens that appear in more than 20% of the documents
vect = CountVectorizer(min_df=20, max_df=0.2, stop_words='english', 
                       token_pattern='(?u)\\b\\w\\w\\w+\\b')
# Fit and transform
X = vect.fit_transform(newsgroup_data)

# Convert sparse matrix to gensim corpus.
corpus = gensim.matutils.Sparse2Corpus(X, documents_columns=False)

# Mapping from word IDs to words (To be used in LdaModel's id2word parameter)
id_map = dict((v, k) for k, v in vect.vocabulary_.items())

##################
# PART 2

# Use the gensim.models.ldamodel.LdaModel constructor to estimate 
# LDA model parameters on the corpus, and save to the variable `ldamodel`

# Your code here:
ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=10, id2word=id_map, passes=25, random_state=34)

#################

def lda_topics():
    
    return list(ldamodel.show_topics(num_topics=10, num_words=10))

lda_topics()

#################

new_doc = ["\n\nIt's my understanding that the freezing will start to occur because \
of the\ngrowing distance of Pluto and Charon from the Sun, due to it's\nelliptical orbit. \
It is not due to shadowing effects. \n\n\nPluto can shadow Charon, and vice-versa.\n\nGeorge \
Krumins\n-- "]

################

def topic_distribution():
    
    sparse_doc = vect.transform(new_doc)
    gen_corpus = gensim.matutils.Sparse2Corpus(sparse_doc, documents_columns=False)
    return list(ldamodel[gen_corpus])[0] # It's a list of lists! You just want the first one.
    #return list(ldamodel.show_topics(num_topics=10, num_words=10)) # For topic_names

topic_distribution()

#################

# Manually assign labels based on most important features and most important words in them:
# Assigning labels manually for the first step in supervised learning.
def topic_names():
    
    return ['Education','Science','Computers & IT','Religion','Automobiles','Sports','Science','Religion','Computers & IT','Science']

#################