'''
Created on Dec 18, 2014

@author: mbilgic
'''

import utilities.experimentutils as exputil
import learner
import utilities.datautils as datautil
import numpy as np
import experiment.base as exp
import nltk

if __name__ == '__main__':
    # Load the data
    print "Loading the data..."
    imdb_path = 'C:/Users/mbilgic/Desktop/aclImdb/'    
    data = datautil.load_dataset('imdb', imdb_path, categories=None, rnd=5463, shuffle=True)
    sent_tk = nltk.data.load('tokenizers/punkt/english.pickle')
    
    # Vectorize the data
    print "Vectorizing the data..."
    vct = exputil.get_vectorizer({'vectorizer':"tfidf", 'limit':None, 'min_size':None})    
    data.train.bow = vct.fit_transform(data.train.data)
    data.test.bow = vct.transform(data.test.data)
    data.train.data = np.array(data.train.data, dtype=object)
    data.test.data = np.array(data.test.data, dtype=object)
    print "Train size: (%d, %d)" %data.train.bow.shape
    print "Test size: (%d, %d)" %data.test.bow.shape
    
    # The expert
    print "Training the expert..."
    expert = exputil.get_classifier('lrl2',parameter=1)
    expert.fit(data.train.bow, data.train.target)
    
    ## Select N random documents from the test
    n = 1000
    rnd = np.random.RandomState(2345)
    rnd_docs = rnd.choice(len(data.test.target), size = n, replace = False)
    
    ## Get the sentences per document
    sentences = sent_tk.tokenize_sents(data.test.data[rnd_docs])
    
    num_trials = 5
    
    for t in range(num_trials):
        print "\n\nTrial: %d" %t
        rnd = np.random.RandomState(t)
    
        # The student
        student = learner.strategy.StructuredLearner(exputil.get_classifier('lrl2',parameter=1))        
        student.set_sent_tokenizer(sent_tk)
        student.set_vct(vct)
        student.set_snippet_utility('sr')
        
        # Get a bootstrap
        bootstrap_size = 200
        bootstrap = rnd.choice(len(data.train.target), size = bootstrap_size, replace = False)
        
        # Fit the student to bootstrap
        print "Training the student..."
        student.fit(data.train.bow[bootstrap], data.train.target[bootstrap], doc_text=data.train.data[bootstrap])
        
        student_cm = np.zeros(shape=(2,2))
        expert_cm = np.zeros(shape=(2,2))
        
        for i, idx in enumerate(rnd_docs):        
            true_target = data.test.target[idx]                
            bow = vct.transform(sentences[i])
            
            expert_pred = expert.predict(bow)
            for p in expert_pred:
                expert_cm[true_target, p] += 1
            
            student_pred = student.snippet_model.predict(bow)
            for p in student_pred:
                student_cm[true_target, p] += 1
            
            
        
        print "\n\nNum test documents: %d" %len(sentences)
        num_sentences = 0
        for sent in sentences:
            num_sentences += len(sent)
        print "Num test sentences: %d" %num_sentences
        
        print "True class distribution: %s" % expert_cm.sum(1)
        
        print "Expert predictions: %s" % expert_cm.sum(0)
        print "Expert accuracy: %0.4f" % ((expert_cm[0,0]+expert_cm[1,1])/float(num_sentences))
        
        print "Student predictions: %s" % student_cm.sum(0)
        print "Student accuracy: %0.4f" % ((student_cm[0,0]+student_cm[1,1])/float(num_sentences))
        
        calibration = ['_no_calibrate', 'zscores_pred', 'zscores_rank']
        
        for cal in calibration:
            student.set_calibration_method(cal)
                
            _, chosen_sentences = student._compute_snippet(data.test.data[rnd_docs])
            
            student_cm = np.zeros(shape=(2,2))
            expert_cm = np.zeros(shape=(2,2))
            
            for i, idx in enumerate(rnd_docs):        
                true_target = data.test.target[idx]                
                bow = vct.transform([chosen_sentences[i]])
                
                expert_pred = expert.predict(bow)
                for p in expert_pred:
                    expert_cm[true_target, p] += 1
                
                student_pred = student.snippet_model.predict(bow)
                for p in student_pred:
                    student_cm[true_target, p] += 1
            
            print "\n\nCalibration: %s" %cal
            
            num_sentences = len(chosen_sentences)
            
            print "Num test sentences: %d" %num_sentences
            
            print "True class distribution: %s" % expert_cm.sum(1)
            
            print "Expert predictions: %s" % expert_cm.sum(0)
            print "Expert accuracy: %0.4f" % ((expert_cm[0,0]+expert_cm[1,1])/float(num_sentences))
            
            print "Student predictions: %s" % student_cm.sum(0)
            print "Student accuracy: %0.4f" % ((student_cm[0,0]+student_cm[1,1])/float(num_sentences))        