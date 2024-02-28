import numpy as np

def update_rule(weight, bias, X, Y): # X is the feature vector, Y is the label. This function returns 0 or 1, based on whether or not to update.
    activation = bias
    for i in range(len(weight)):
        activation += X[i] * weight[i]
    
    updater = Y * activation
    
    if updater > 0:
        # print (">0", updater)
        return 0
    else:
        # print ("<=0", updater)
        return 1

def perceptron_train(X,Y):
    if hasattr(X[0], "__len__"):
        init_weight = np.ones(len(X[0]))
    else:
        init_weight = np.ones(1)
    init_bias = 1

    if hasattr(X[0], "__len__"):
        final_weight = np.zeros(len(X[0]))
    else:
        final_weight = np.zeros(1)
    final_bias = 0

    while (np.array_equal(init_weight, final_weight) == False) or (init_bias != final_bias):
        init_weight = final_weight.copy()
        init_bias = final_bias

        
        for i in range(len(Y)):
            updater = update_rule(final_weight, final_bias, X[i], Y[i])
            # print(updater)
            if updater == 1:
                for j in range(len(final_weight)):
                    final_weight[j] += Y[i]*X[i][j]
                final_bias += Y[i]

                # print("updater", X[i], Y[i])
                # print(final_weight, final_bias)

    return final_weight, final_bias

def perceptron_test(X_test, Y_test, w, b):
    correct_count = 0
    for i in range(len(Y_test)):
        if (update_rule(w, b, X_test[i], Y_test[i]) == 0):
            correct_count += 1

    return correct_count / len(Y_test)



# test
# Xtrain = [[3, 3], [-2, -2], [2, -1], [2, 2], [-3, -1], [-3, 1], [1, 1], [-2, -1], [3, 1], [-1, -1]]
# Ytrain = [1, -1, 1, 1, -1, -1, 1, -1, 1, -1]
# print(perceptron_train(Xtrain, Ytrain))