import predict

def test_euclidean_dist():
    item = dict()
    for key in predict.to_predict.keys():
        predict[key] = 0
        item[key] = 0
    distance = predict.euclidean_dist(item)
    assert distance == 0 


def test_euclidean_dist1():
    item = dict()
    for key in predict.to_predict.keys():
        predict[key] = 1
        item[key] = 1
    distance = predict.euclidean_dist(item)
    assert distance == 0


def test_algorithm():
    result = predict.acc_test()
    for item in result:
        assert item>0.7



