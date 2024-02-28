import math
import numpy as np
from data_storage import build_nparray

# decision tree implementation

def calculate_probability_true(Y):
    probability_true = float(0)
    for label in Y:
        if label == 1:
            probability_true += 1
    if len(Y) != 0:
        probability_true /= float(len(Y))
    return probability_true


def calculate_entropy(Y): 
    # entropy only deals with the labels, not the features
    # H = - p(no)log_2(p(no)) - p(yes)log_2(p(yes))
    entropy = float(0)
    probability_true = calculate_probability_true(Y)
    if ((probability_true == 0) or (probability_true == 1)):
        entropy = 0
    else:
        entropy -= probability_true * math.log(probability_true, 2) + (1 - probability_true) * math.log(1 - probability_true, 2)
    
    return entropy


def feature_true_labels(X, Y, i, true_false): # this will return a list of the labels corresponding to the samples with feature i being true/false (0 or 1).
    label_list = []
    for sample in range(len(X)):
        if X[sample][i] == true_false:
            label_list.append(Y[sample])
    return label_list


def choose_split(X, Y):
    # step 1: calcuate entropy of the full set
    total_entropy = calculate_entropy(Y)

    # step 2: split training data using each feature & calculate information gain for each split
    # step 3: choose the split that yields the highest information gain
    # IG = H - sum(p(t)H(t))
    best_split_index = 0
    best_info_gain = 0
    for index in range(len(X[0])):
        probability_feature_true = calculate_probability_true(X[:, index])
        probability_feature_false = 1 - probability_feature_true
        entropy_feature_true = calculate_entropy(feature_true_labels(X, Y, index, 1))
        entropy_feature_false = calculate_entropy(feature_true_labels(X, Y, index, 0))
        
        info_gain = total_entropy - probability_feature_true * entropy_feature_true - probability_feature_false * entropy_feature_false

        if info_gain >= best_info_gain:
            best_info_gain = info_gain
            best_split_index = index
    
    if best_info_gain == 0:
        return -1 # this means we shoud stop
    else:
        return best_split_index


def DT_train_binary(X, Y, max_depth):
    DT = [[], 0, []]

    if ((max_depth != 0) and (calculate_entropy(Y) != 0) and (len(X) != 0) and (len(X[0]) != 0)):
        best_split = choose_split(X, Y)

        false_samples = []
        true_samples = []

        for i in range(len(X)):
            if X[i, best_split] == 1:
                true_samples.append(i)
            elif X[i, best_split] == 0:
                false_samples.append(i)

        X_true = np.delete(X, false_samples, 0)
        X_false = np.delete(X, true_samples, 0)

        Y_true = np.delete(Y, false_samples)
        Y_false = np.delete(Y, true_samples)

        X_true = np.delete(X_true, best_split, 1)
        X_false = np.delete(X_false, best_split, 1)


        DT = [DT_train_binary(X_false, Y_false, max_depth - 1), best_split, DT_train_binary(X_true, Y_true, max_depth - 1)]
    else:
        t = calculate_probability_true(Y)
        if t >= 0.5:
            DT = [1]
        else:
            DT = [0]
    
    return DT


def DT_make_prediction(x, DT):
    prediction = 0
    if len(DT) > 1:
        if (x[DT[1]] == 0):
            x = np.delete(x, DT[1])
            prediction = DT_make_prediction(x, DT[0])
        else:
            x = np.delete(x, DT[1])
            prediction = DT_make_prediction(x, DT[2])
    else:
        prediction = DT[0]

    return prediction


def DT_test_binary(X, Y, DT):
    correct_counter = 0
    for sample in range(len(X)):
        if DT_make_prediction(X[sample], DT) == Y[sample]:
            correct_counter += 1
    return float(correct_counter) / float(len(Y))
        


# random forest implementation

def RF_build_random_forest(X, Y, max_depth, num_of_trees):
    random_forest = []
    for i in range(num_of_trees):
        not_sampling = np.random.choice(len(Y), (9 * len(Y)) // 10, replace = False)
        x_sample = np.delete(X, not_sampling, 0)
        y_sample = np.delete(Y, not_sampling)

        random_forest.append(DT_train_binary(x_sample, y_sample, max_depth))

    return random_forest
        

def RF_make_prediction(x, RF):
    predictions = []
    for DT in RF:
        pred = DT_make_prediction(x, DT)
        predictions.append(pred)
    if (calculate_probability_true(predictions) >= 0.5):
        return 1
    else:
        return 0


def RF_test_random_forest(X, Y, RF):
    DT_accuracy = []
    RF_accuracy = float(0)
    for DT in RF:
        acc = DT_test_binary(X, Y, DT)
        DT_accuracy.append(acc)
        RF_accuracy += acc
            
    for i in DT_accuracy:
        print(i)
    RF_accuracy /= len(RF)

    return RF_accuracy

    # another method:

    # return RF_accuracy

    # correct_counter = 0
    # for sample in range(len(X)):
    #     if RF_make_prediction(X[sample], RF) == Y[sample]:
    #         correct_counter += 1
    # return float(correct_counter) / float(len(Y))
