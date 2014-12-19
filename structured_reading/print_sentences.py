'''
Created on Dec 16, 2014

@author: mbilgic
'''

import utilities.experimentutils as exputil
import learner
import utilities.datautils as datautil
import numpy as np
import experiment.base as exp
import nltk

if __name__ == '__main__':
        
    ## Get the data ready
    imdb_path = 'C:/Users/mbilgic/Desktop/aclImdb/'
    rnd = np.random.RandomState(2345)
    clf = exputil.get_classifier('lrl2',parameter=1)
    expert = exputil.get_classifier('lrl2',parameter=1)
    vct = exputil.get_vectorizer({'vectorizer':"tfidf", 'limit':None, 'min_size':None})
    data = datautil.load_dataset('imdb', imdb_path, categories=None, rnd=5463, shuffle=True)
    data.train.bow = vct.fit_transform(data.train.data)
    expert = exputil.get_classifier('lrl2',parameter=1)
    
    ## Set the learner options and expert
    sent_tk = nltk.data.load('tokenizers/punkt/english.pickle')
    student = learner.strategy.StructuredLearner(clf)
    student.set_sent_tokenizer(sent_tk)
    student.set_vct(vct)
    student.set_snippet_utility('sr')
    student.set_calibration(True)
    expert.fit(data.train.bow, data.train.target)
    
    ## Get the boostrap and train
    data.train.remaining = rnd.permutation(len(data.train.target))
    ## balanced bootstrap
    # bt_obj = learner.strategy.BootstrapFromEach(None, seed=5463)
    # initial = bt_obj.bootstrap(data.train, step=200, shuffle=True)
    ## random bootstrap 
    data.train.data = np.array(data.train.data, dtype=object)
    initial = rnd.choice(data.train.remaining, 200, replace=False)
    student.fit(data.train.bow[initial], data.train.target[initial], doc_text=data.train.data[initial])
    
    ## Select N random documents
    n = 100
    rnd_docs = list(set(data.train.remaining) - set(initial))
    np.random.shuffle(rnd_docs)
    print len(rnd_docs), rnd_docs[:10]
    
    # Process
    ## Get the sentences per document
    sent = sent_tk.tokenize_sents(data.train.data[rnd_docs[:n]])
    
    ## compute the max scoring snippet (with calibration)
    sent_scores, snippet_text = student._compute_snippet(data.train.data[rnd_docs[:n]])
    
    def print_document_senteces(data, student, indices, n=100):
        results = []
        for i, idx in enumerate(indices[:n]):
            doc_i = []
            bow = vct.transform(sent[i])
            pred = student.snippet_model.predict(bow)
            exp_pred = expert.predict(bow)
            for j, s in enumerate(zip(sent[i], pred, exp_pred)):
                selected = False
                if snippet_text[i] == s[0]:
                    selected = True
                doc_i.append([i, data.target[idx], j, s[1], s[2],student._snippet_max(bow[j]), s[0], selected])
    #         doc_i.extend([sent_scores[i], snippet_text[i]])
            results.extend(doc_i)
        return results
    
    # r =  print_document_senteces(data.train, student, rnd_docs)
    # "\n".join(["{}\t".format(str(x)) for x in r])
    
    # #print scores per sentence
    ## DOCID, TARGET(True label), SENTENCE ID, PRED (student), SCORE(maxprob), Text
    d = 0
    for i, idx in enumerate(rnd_docs[:n]):
        
        print "-"*40
        print "Doc:", i
        print "Target:", data.train.target[idx]
        
        bow = vct.transform(sent[i])
        pred = student.snippet_model.predict(bow)
        exp_pred = expert.predict(bow)
        print "ID\tPred\tExpertPred\tScore\ttext"
        for j, s in enumerate(zip(sent[i], pred, exp_pred)):
            if snippet_text[i] == s[0]:  # if selected sentence, mark it
                #print "*%s\t%s\t%s\t%.3f\t%s" % (j, s[1], s[2], student._snippet_max(bow[j]), s[0])
                pass
            else:
                #print "%s\t%s\t%s\t%.3f\t%s" % (j, s[1], s[2], student._snippet_max(bow[j]), s[0])
                pass
            d +=1
        #print sent_scores[i], snippet_text[i]
    
    print d
    
    # Doc, Target, ID,    Pred,    ExpertPred,    Score,    text

    r =  print_document_senteces(data.train, student, rnd_docs)
    # no_m = [x for x in r if ]
    print "Total:", len(r)
    print "Student Pred 0:",len([x for x in r if x[3] == 0])
    print "True 0:", len([x for x in r if x[1] == 0])
    print "Expert Pred 0:",len([x for x in r if x[4] == 0])
    print "Boostrap: ", data.train.target[initial].sum()
    
    from sklearn.metrics import accuracy_score

    data.test.bow = vct.transform(data.test.data)
    
    from sklearn.metrics import confusion_matrix

    # test = student.model.predict(data.test.bow)
    # print "student accuracy on test",  accuracy_score(data.test.target, test)
    # print "expert accuracy on test",  accuracy_score(data.test.target, expert.predict(data.test.bow))
    # print "Total test instances", len(data.test.target)
    print
    print "Expert Accu on rnd docs: %.3f (%s / %s)" % (accuracy_score([x[1] for x in r], [x[4] for x in r]), len([x for x in r if x[4] == x[1]]), len(r))
    print "Student Accu on rnd docs:  %.3f (%s / %s)" % (accuracy_score([x[1] for x in r], [x[3] for x in r]) ,len([x for x in r if x[3] == x[1]]), len(r))
    
    print "CM Student"
    print confusion_matrix([x[1] for x in r], [x[3] for x in r])
    # print confusion_matrix(data.test.target, test)
    print "CM Expert"
    print confusion_matrix([x[1] for x in r], [x[4] for x in r])
    # print confusion_matrix(data.test.target, expert.predict(data.test.bow))
    # Doc, Target, ID,    Pred,    ExpertPred,    Score,    text., selected
    
    f = [s for s in r if s[7] == True]
    for s in f:
        #print s[0], s[2], s[3], s[4], s[5], s[6], s[7]
        pass
    
    print "Correct labeled (expert):", len([s for s in f if s[1] == s[4]]), "out of ", len(f)
    print "Correct predicted (student):", len([s for s in f if s[1] == s[3]]), "out of ", len(f)
    print 
    print "CM of Expert on selected sentences"
    print "Accuracy: %.3f" % accuracy_score([x[1] for x in f], [x[4] for x in f])
    print confusion_matrix([x[1] for x in f], [x[4] for x in f])
    
    print "CM of Student on selected sentences"
    print "Accuracy: %.3f" % accuracy_score([x[1] for x in f], [x[3] for x in f])
    print confusion_matrix([x[1] for x in f], [x[3] for x in f])