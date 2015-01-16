'''
A Factor graph consists of

Nodes
and
Features
'''

import numpy as np

class Node(object):
    def __init__(self, node_id, domain_size):
        self.node_id = node_id
        self.domain_size = domain_size

class Feature(object):
    def __init__(self, nodeA, nodeB, feature_function):
        self.nodeA = nodeA
        self.nodeB = nodeB
        self.feature_function = feature_function
    
    def feature_value(self, nodeA_value, nodeB_value):
        return self.feature_function(nodeA_value, nodeB_value)

class TiedFeatures(object):
    def __init__(self, features):
        self.features = features

if __name__ == '__main__':
    a = Node(0, 2)
    b = Node(1, 2)
    
    def identity_feature_function(nodeA_value, nodeB_value, nodeA_check, nodeB_check):
        if nodeB_value is None:
            return int(nodeA_value == nodeA_check)
        else:
            return int(nodeA_value == nodeA_check and nodeB_value == nodeB_check)
    
    features = []
    
    #for i in range(2):
    #    for j in range(2):
    #        print i, j
    #        features.append(Feature(a, b, lambda x, y: identity_feature_function(x, y, i, j)))
    
    features.append(Feature(a, b, lambda x, y: identity_feature_function(x, y, 0, 0)))
    features.append(Feature(a, b, lambda x, y: identity_feature_function(x, y, 0, 1)))
    features.append(Feature(a, b, lambda x, y: identity_feature_function(x, y, 1, 0)))
    features.append(Feature(a, b, lambda x, y: identity_feature_function(x, y, 1, 1)))
    
    weights = np.array([-1., -2., -1., -2])
    
    # print the probs
    z = 0
    for i in range(2):
        for j in range(2):
            fs = 0
            for k in range(len(features)):
                fs += -weights[k]*features[k].feature_value(i, j)
            
            z += np.exp(fs)
    
    print "z ", z
    
    for i in range(2):
        for j in range(2):
            fs = 0
            for k in range(len(features)):
                fs += -weights[k]*features[k].feature_value(i, j)
            print i, j, np.exp(fs)/z
    
    
    
        