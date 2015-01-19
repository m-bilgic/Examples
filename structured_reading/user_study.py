import utilities.experimentutils as exputil
import utilities.datautils as datautil
import numpy as np
import nltk
from sklearn import metrics
import matplotlib.pyplot as plt
import matplotlib as mpl
import sys
import os

# Loading Data
vct = exputil.get_vectorizer({'vectorizer':"tfidf", 'limit':None, 'min_size':None})
# Sentence tokenizers
sent_tk = nltk.data.load('tokenizers/punkt/english.pickle')

def load_data(dataname, path):
    import pickle

    DATA_PKL = path + '/data.pkl'

    if os.path.isfile(DATA_PKL):
        vct, data = pickle.load(open(DATA_PKL, 'rb'))
    else:
        vct = exputil.get_vectorizer({'vectorizer':"tfidf", 'limit':None, 'min_size':None})
        data = datautil.load_dataset(dataname, path, categories=None, rnd=5463, shuffle=True)
        data.train.data = np.array(data.train.data, dtype=object)
        data.test.data = np.array(data.test.data, dtype=object)
        data.train.bow = vct.fit_transform(data.train.data)
        data.test.bow = vct.transform(data.test.data)
        pickle.dump((vct, data), open(DATA_PKL, 'wb'))

    return data, vct

# Get the sentences for testing
def _sentences(docs, doc_labels, sent_tk):
    data = []
    true_labels = []
    sent = sent_tk.tokenize_sents(docs)
    for sentences, doc_label in zip(sent, doc_labels):
        data.extend(sentences)
        true_labels.extend([doc_label] * len(sentences))
    return data, np.array(true_labels)

def load_study(filename):
    f = open(filename)
    with f:
        lines = f.readlines() 
    data =  [l.strip().split("\t") for l in lines[1:]] #discard first line
    data = np.array(data, dtype=object)
    return data

IMDB_DATA = "C:\\Users\\mbilgic\\Desktop\\aclIMDB"

# Load the dataset
print "Loading the data"
imdb, vct = load_data('imdb', IMDB_DATA)


# Load the dictionary from dist

with open(IMDB_DATA+"\\dictionary2.csv") as f:
    lines = f.readlines()
    vocab = [l.strip() for l in lines]

from sklearn.feature_extraction.text import  TfidfVectorizer
vct = TfidfVectorizer(encoding='ISO-8859-1', min_df=5, max_df=1.0, binary=False, vocabulary=vocab)
vct.fit(imdb.train.data)

# Get the snippets from the saved file
docs_study = load_study('rnd_imdb.txt')
lbl = [float(d[0]) for d in docs_study]
snippets = [d[1] for d in docs_study]
snip_bow = vct.transform(snippets)

## Get the expert data and snippets
#exp_sent, exp_lbl = _sentences(imdb.test.data, imdb.test.target, sent_tk)
#exp_sent_bow = vct.transform(exp_sent)
#print "Sentences:", len(exp_sent), len(exp_lbl)

from sklearn.linear_model import LogisticRegression
expert = LogisticRegression('l1', C=5)

#from sklearn.tree import DecisionTreeClassifier
#expert = DecisionTreeClassifier()

#from sklearn.naive_bayes import MultinomialNB
#expert = MultinomialNB()

#expert.fit(imdb.test.bow, imdb.test.target)
expert.fit(vct.transform(imdb.test.data), imdb.test.target)
pred = expert.predict(snip_bow)
prob = expert.predict_proba(snip_bow)

print "num nonzero %d" %len(np.nonzero(expert.coef_[0])[0])

#exit(0)

def print_top_terms(model, terms, n=20):
    print '\nTop Coefficients'
    coef = model.coef_[0]
    srted = np.argsort(coef)
    topi = srted[::-1][:n]
    boti = srted[:n]
    #"\n".join('%s\t%.2f' % (names[j], coef[j]) for j in np.argsort(coef)[::-1] if coef[j] != 0)
    print 'Class-0 Terms:\n' + '\n'.join('%s (%.2f)' % (n, c) for n, c in zip(terms[topi], coef[topi]) if c != 0)
    print '\nClass-1 Terms:\n' + '\n'.join('%s (%.2f)' % (n, c) for n, c in zip(terms[boti], coef[boti]) if c != 0)
    print '\nintercept=%g' % model.intercept_
    
#terms = np.array(vct.get_feature_names())



#print "printing terms to a file"

#with open(IMDB_DATA+"\\vocab.csv", "w") as f:
#    for i in range(len(terms)):
#        f.write(str(i)+","+terms[i].encode("utf-8")+","+str(expert.coef_[0][i])+"\n")

#exit(0)

for t, p in zip(pred,prob):
    l = t if (1- p.max()) < .46 else 2 
    print l
