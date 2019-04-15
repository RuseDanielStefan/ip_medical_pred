import pandas as pd
import json
import math
import random
import matplotlib.pyplot as plt

print("predict imported")


min_feature = dict()
max_feature = dict()
to_predict = {  
    'age': 0.7083333333333334,
    'sex': 1.0, 
    'cp': 1.0,
    'trestbps': 0.4811320754716981,
    'chol': 0.24429223744292236,
    'fbs': 1.0, 'restecg': 0.0,
    'thalach': 0.6030534351145038,
    'exang': 0.0,
    'oldpeak': 0.3709677419354838,
    'slope': 0.0, 'ca': 0.0,
    'thal': 0.3333333333333333,
}
health_data = list()


def normalize_feature(feature):
    x_min = min([person[feature] for person in health_data])
    x_max = max([person[feature] for person in health_data])
    min_feature[feature] = x_min
    max_feature[feature] = x_max
    for i, val in enumerate( [person[feature] for person in health_data] ):
        health_data[i][feature] = (val - x_min) / (x_max - x_min)

    
def normalize_input(inp):
    for key in inp.keys():
        x_min = min_feature[key]
        x_max = max_feature[key]
        inp[key] = (inp[key] - x_min) / (x_max - x_min)
    return dict(inp)


def read_csv(path):
    health_file = open(path, 'r')
    header = health_file.readline()
    features = [feature.rstrip() for feature in header.split(',')]
    features.pop(0)
    for line in health_file.readlines():
        values = line.split(',')
        person = dict()
        for i, value in enumerate(values):
            person[features[i]] = float(value)
        health_data.append( dict(person) )
    health_file.close()


def read_n_normalize(path):
    read_csv(path)
    for key in health_data[0].keys():
        if key!='target':
            normalize_feature(key)


def euclidean_dist(item):
    sum_all = 0
    for key in item.keys():
        if key!='target':
            sum_all += (item[key] - to_predict[key]) ** 2
    return math.sqrt(sum_all)


def knn_class(to_predict, k=8):
    yes_votes = sum([ point['target'] for point in health_data[:k] ])
    return yes_votes/k
    

read_n_normalize('heart.csv')

#dupa numeroase teste am stabilit valoarea lui k = 8
def classify(predict_dict, k = 8):
    print(len(health_data))
    global to_predict
    to_predict = normalize_input(predict_dict)
    print(to_predict)
    health_data.sort(key=euclidean_dist)
    result = knn_class(to_predict, k)
    print(result)
    return json.dumps({'result' : result})




if __name__ == "__main__":
    read_n_normalize('heart.csv')
    print(health_data[:5])
    random.shuffle(health_data)
    bariera = (len(health_data)*4)//5
    test_data = list(health_data[ bariera:])
    health_data = list(health_data[:bariera])
    print(len(test_data) + len(health_data))
    results = list()
    for k in range(1, 31):
        corect = 0
        for person in test_data:
            to_predict = person
            health_data.sort(key=euclidean_dist)
            result = knn_class(to_predict, k)
            if result == person['target']: corect += 1
        results.append(corect/len(test_data))
        print("k=", k, ' accuracy = ', corect/len(test_data))
    plt.plot([x for x in range(1, 31)], results)
    plt.axis([0, 31, 0.5, 1])
    plt.show()





