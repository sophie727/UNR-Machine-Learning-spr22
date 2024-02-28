import numpy as np
from scipy.spatial import distance


def KNN_predict(X_train,Y_train,X, K): #predict what label some point X should have
    distances = np.zeros([len(Y_train), 2])

    for i in range(len(Y_train)):
        distances[i][0] = distance.euclidean(X_train[i], X)
        distances[i][1] = Y_train[i]
    
    distances = distances[np.argsort(distances[:, 0])]

    label_count = 0

    for i in range(K):
        if distances[i][1] == 1:
            label_count += 1
    
    if 2 * label_count > K:
        return 1
    else:
        return -1


def KNN_test(X_train,Y_train,X_test,Y_test,K):
    correct_count = 0

    for i in range(len(Y_test)):
        prediction = KNN_predict(X_train, Y_train, X_test[i], K)
        if prediction == Y_test[i]:
            correct_count += 1
            # print("correct", X_train[i], i)

    return correct_count / len(Y_test)


# test
# Xtrain = [[0, 0], [1, 1], [1, 2], [1, 3], [2, 2], [3, 2]]
# Ytrain = [1, 1, -1, 1, 1, -1]

# print(KNN_predict(Xtrain, Ytrain, [3, 1], 3))

# print(KNN_test(Xtrain, Ytrain, Xtrain, Ytrain, 1))

# Xtest = [[1, 1], [2, 1], [3, 1], [3, 3]]
# Ytest = [1, -1, 1, 1]
# print(KNN_test(Xtrain, Ytrain, Xtest, Ytest, 3))