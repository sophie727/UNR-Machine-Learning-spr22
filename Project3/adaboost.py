import errno
import numpy as np
import math
import random
from regex import P
from sklearn import tree

def adaboost_train(X, Y, max_iter):
    f = []
    alpha = np.zeros(max_iter)

    weight = len(Y) # this is the total number of samples you need in the dataset with duplicate samples (representing the weighted dataset)


    weighted_data = X.copy()
    weighted_labels = Y.copy()

    weights = np.ones(len(X)) # initializing this to 1 instead of 1/N because it's not like I'll actually be using the normalized weights, anyway

    for i in range(max_iter):
        stump = tree.DecisionTreeClassifier(max_depth = 1)
        stump = stump.fit(weighted_data, weighted_labels)

        f.append(stump)

        error = 0

        for j in range(len(weighted_data)):
            prediction = stump.predict([weighted_data[j]])
            if prediction != weighted_labels[j]:
                error += 1
        
        error /= weight

        if error == 0:
            # this is a perfect tree, basically. That basically means we can stop the for loop. alpha can't really be calculated since 1/0 is undefined, so I'll just let it be something large.
            alpha[i] = 10000
            break
        elif error == 1:
            # this won't actually happen because the decision tree stump won't result in an error of greater than 50% (just switch the question labels around and you'll get an improved error).
            pass
        else:
            a = 1/2 * math.log(1/error - 1)
        alpha[i] = a

        weight = 0

        for j in range(len(weights)):
            prediction = stump.predict([weighted_data[j]])
            if prediction != weighted_labels[j]:        
                weights[j] *= 10 * math.e ** (a)
            else:
                weights[j] *= 10 * math.e ** (-a)

            weights[j] = int(weights[j])

            weight += weights[j]

        weight = int(weight)

        weighted_data = np.zeros([weight, len(X[0])])
        weighted_labels = np.zeros(weight)


        index = 0
        total_weight = weights[0]
        for j in range(weight):
            if (j > total_weight):
                index += 1
                total_weight += weights[index]
                
            weighted_data[j] = X[index]
            weighted_labels[j] = Y[index]
    
    return f, alpha


def adaboost_test(X, Y, f, alpha):
    correct_count = 0

    for i in range(len(X)):
        prediction = 0
        for j in range(len(f)):
            prediction += alpha[j] * f[j].predict([X[i]])

        if (prediction * Y[i]) > 0:
            correct_count += 1
            
    return (correct_count / len(X))

    
        