import utilities.experimentutils as exputil
import utilities.datautils as datautil
import numpy as np
import nltk
from sklearn import metrics
import matplotlib.pyplot as plt
import matplotlib as mpl
import sys
import os
from sklearn.feature_extraction.text import  TfidfVectorizer
import pickle
from sklearn.linear_model import LogisticRegression


with open("imdb-vocab-annotated.txt") as f:
    lines = f.readlines()
    vocab = [l.strip() for l in lines]

print "Dictionary size: %d" %len(vocab)
#print vocab

# Loading Data
print "Loading the data"
vct = TfidfVectorizer(encoding='ISO-8859-1', min_df=5, max_df=1.0, binary=False, vocabulary=vocab)

if os.path.isfile("imdb-data.pkl"):
    data = pickle.load(open("imdb-data.pkl", 'rb'))
else:
    data = datautil.load_dataset("imdb", "C:\\Users\\mbilgic\\Desktop\\aclIMDB", categories=None, rnd=5463, shuffle=True)
    data.train.data = np.array(data.train.data, dtype=object)
    data.test.data = np.array(data.test.data, dtype=object)
    pickle.dump(data, open("imdb-data.pkl", 'wb'))

print "Fitting the vectorizer"
data.test.bow = vct.fit_transform(data.test.data)

# Fit the expert
print "Training the expert"
expert = LogisticRegression('l2', C=1)
expert.fit(data.test.bow, data.test.target)

terms = np.array(vct.get_feature_names())

coefs = expert.coef_[0]

for t in range(len(terms)):
    print "%s\t%0.4f" %(terms[t], coefs[t])

exit(0)

data.train.bow = vct.transform(data.train.data)

y_pred = expert.predict(data.train.bow)

from sklearn.metrics import accuracy_score

print "Accuracy %0.4f" %accuracy_score(data.train.target, y_pred)

# full documents

for doc in range(data.train.bow.shape[0]):    
    if len(data.train.bow[doc].indices) < 1:         
        print
        print data.train.data[doc]  
        raw_input()
exit(0)


# Sentence tokenizers
sent_tk = nltk.data.load('tokenizers/punkt/english.pickle')

n = 1000
rnd = np.random.RandomState()
rnd_docs = rnd.choice(len(data.train.target), size = n, replace = False)
sentences = sent_tk.tokenize_sents(data.train.data[rnd_docs])

# individual sentences
for doc in sentences:
    for sent in doc:
        x = vct.transform([sent])
        max_prob = max(expert.predict_proba(x)[0])
        if max_prob < 0.55:            
            print
            print sent        
            raw_input()