import gensim
from sklearn.feature_extraction.text import CountVectorizer


# Use CountVectorizor to find three letter tokens, remove stop_words, 
# remove tokens that don't appear in at least 20 documents,
# remove tokens that appear in more than 20% of the documents
vect = CountVectorizer(min_df=20, max_df=0.2, stop_words='english', 
                       token_pattern='(?u)\\b\\w\\w\\w+\\b')

# ^ Initializes the CountVectorizer

# Fit and transform
def pre_process(ls_of_string):
    """Takes in a list of strings and preprocesses them for the gensim LDA model."""
    
    X = vect.fit_transform(ls_of_string)
    
    # Convert sparse matrix to gensim corpus.
    corpus = gensim.matutils.Sparse2Corpus(X, documents_columns=False)
    
    # Mapping from word IDs to words (To be used in LdaModel's id2word parameter)
    id_map = dict((v, k) for k, v in vect.vocabulary_.items())

    return (corpus, id_map)

def train(corpus, num_topics, id2word, passes, random_state):
    trained_model = gensim.models.ldamodel.LdaModel(
        corpus, 
        num_topics=num_topics, 
        id2word = id2word, 
        passes=passes, 
        random_state=random_state)

    training_log = ('''
    Number of topics: {},
    passes: {},
    random state: {}\n'''.format(num_topics, passes, random_state))

    return trained_model, training_log

def autoname_topics(model):

    #Fetch the 2nd item (string of probability) of every tuple in a list of tuples.
    probabilities = [item[1] for item in model.show_topics(num_topics=10, num_words=10)]
    
    #Manually assign keys the topics based on keywors and their probabilities.
    keys = ['Education','Science','Computers & IT','Religion','Automobiles','Sports','Science','Religion','Computers & IT','Science']

    #Initialize a dict.
    topics = {}
    #Update the dict with key-value pairs
    topics.update(zip(keys, probabilities))

    return topics

def gen_topic_map(num_of_topics):
    k = [i for i in range(num_of_topics)]
    v = ['Education','Science','Computers & IT','Religion','Automobiles','Sports','Science','Religion','Computers & IT','Science']

    return dict(zip(k,v))

def test_topic_distribution(doc_to_test, trained_model, topic_map):
    
    doc_to_test = [doc_to_test]

    sparse_doc = vect.transform(doc_to_test)
    gen_corpus = gensim.matutils.Sparse2Corpus(sparse_doc, documents_columns=False)
    
    topics = sorted(list(trained_model[gen_corpus])[0], key=lambda x: -x[1]) # It's a list of lists! We just want the first one.

    return [(topic_map[x[0]], x[1]) for x in topics][:3]