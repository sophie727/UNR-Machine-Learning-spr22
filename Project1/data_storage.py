import numpy as np

def build_nparray(data):
    training_feature_data = np.zeros([len(data) - 1, len(data[0]) - 1])
    training_label_data = np.zeros(len(data) - 1, dtype = int)

    for x in range(len(data) - 1):
        for y in range(len(data[0]) - 1):
            training_feature_data[x, y] = float(data[x + 1][y])
        training_label_data[x] = int(data[x + 1][len(data[0]) - 1])
    
    return training_feature_data, training_label_data
 

def build_list(data):
    training_feature_data = []
    training_label_data = []

    for x in range(len(data) - 1):
        training_feature_data.append([])

        for y in range(len(data[0]) - 1):
            training_feature_data[x].append(float(data[x + 1][y]))
        training_label_data.append(int(data[x + 1][len(data[0]) - 1]))

    return training_feature_data, training_label_data


def build_dict(data):
    training_feature_data = []
    training_label_data = {}

    for x in range(len(data) - 1):
        training_feature_data.append({})
        for y in range(len(data[0]) - 1):
            training_feature_data[x][data[0][y]] = float(data[x + 1][y])
        training_label_data[x] = int(data[x + 1][len(data[0]) - 1])

    return training_feature_data, training_label_data
