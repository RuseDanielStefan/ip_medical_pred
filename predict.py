import pandas as pd
import json
import math
import random
import matplotlib.pyplot as plt

print("predict imported")


def normalize(feature):
    x_min = min(feature)
    x_max = max(feature)
    for i, val in enumerate(feature):
        feature.loc[i] = (val - x_min) / (x_max - x_min)


def normalize_csv(path):
    health_data = pd.read_csv(path)
    suma = 0
    for feature in health_data:
        if feature!='target': normalize(health_data[feature])
    health_data.to_csv('normalized_heart.csv', index=False)

health_data = list()

to_predict = {  
    'age': 0.7083333333333334,
    'sex': 1.0, 'cp': 1.0,
    'trestbps': 0.4811320754716981,
    'chol': 0.24429223744292236,
    'fbs': 1.0, 'restecg': 0.0,
    'thalach': 0.6030534351145038,
    'exang': 0.0,
    'oldpeak': 0.3709677419354838,
    'slope': 0.0, 'ca': 0.0,
    'thal': 0.3333333333333333,
}

def euclidean_dist(item):
    sum_all = 0
    for key in item.keys():
        if key!='target':
            sum_all += (item[key] - to_predict[key]) ** 2
    return math.sqrt(sum_all)


def knn_class(to_predict, k=1):
    yes_votes = sum([ point['target'] for point in health_data[:k] ])
    if yes_votes > k//2:
        return 1
    return 0
    



def read_csv(path):
    health_file = open(path, 'r')
    header = health_file.readline()
    features = [feature.rstrip() for feature in header.split(',')]
    for line in health_file.readlines():
        values = line.split(',')
        person = dict()
        for i, value in enumerate(values):
            person[features[i]] = float(value)
        health_data.append( dict(person) )
    health_file.close()





if __name__ == "__main__":
    read_csv('normalized_heart.csv')
    random.shuffle(health_data)
    test_data = list(health_data[(len(health_data)*2)//3 :])
    health_data = list(health_data[:(len(health_data)*2)//3])
    results = list()
    for k in range(0, 30):
        corect = 0
        for person in test_data:
            to_predict = person
            health_data.sort(key=euclidean_dist)
            result = knn_class(to_predict, k)
            if result == person['target']: corect += 1
        results.append(corect/len(test_data))
        print("k=", k, ' accuracy = ', corect/len(test_data))
    plt.plot([x for x in range(30)], results)
    plt.axis([0, 31, 0.5, 1])
    plt.show()







