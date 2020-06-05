import json
import os

import numpy

from encoder import get_tf_idf_vector
from neural_network import NeuralNetwork


def instantiate_neural_network(learning_coefficient, load=False):
    with open('topics_dictionary.json', 'r') as f:
        topics = json.load(f)
    with open('dictionary.json', 'r') as f:
        dictionary = json.load(f)
    if load == True:
        wih = numpy.load('npy_weights/wih.npy')
        who = numpy.load('npy_weights/who.npy')
        return NeuralNetwork(len(dictionary),
                             int((len(dictionary) + len(topics)) / 2),
                             len(topics), learning_coefficient, wih, who)
    return NeuralNetwork(len(dictionary),
                         int((len(dictionary) + len(topics)) / 2),
                         len(topics), learning_coefficient, None, None)


def train_neural_network(nn, string_test, save=False):
    file_names = [f for f in os.listdir('encoded/tf_idf')
                  if os.path.isfile(os.path.join('encoded/tf_idf', f))]
    j = 0
    print(query_neural_network(nn, string_test))
    for name in file_names:
        name = '1.json'
        newid = name.split('.json')[0]
        print(newid)
        with open(f'encoded/tf_idf/{name}', 'r') as fi:
            inp = json.load(fi)
        with open(f'training_nn/{name}', 'r') as fi:
            target = json.load(fi)
        nn.train(inp, target)
        j += 1
        print(j)
        if (j % 2) == 0:
            print(query_neural_network(nn, string_test))
        if j == 20:
            break
    if save:
        numpy.save('npy_weights/wih.npy', nn.wih)
        numpy.save('npy_weights/who.npy', nn.who)


def query_neural_network(nn, string):
    with open('topics_dictionary.json', "r") as fi:
        topics = json.loads(fi.read())
    array_topics = [key for key in topics]
    result = nn.query(get_tf_idf_vector(string))
    result = result.T[0]
    s = len(topics)
    sorted_indexes = result.argsort()[-s:][::-1]
    top_s_topics = []
    for i in range(s):
        top_s_topics.append((array_topics[sorted_indexes[i]]))
    return top_s_topics
    # index = numpy.where(result == result.max())[0][0]
    # print(index)
    # return array_topics[index]


query2 = 'the Bahia cocoa zone, alleviating the drought since early ' \
         'January and improving prospects for the coming temporao, ' \
         'although normal humidity levels have not been restored, ' \
         'Comissaria Smith said in its weekly review.'


def test_nn():
    nn = instantiate_neural_network(0.01, False)
    train_neural_network(nn, query2, True)


if __name__ == '__main__':
    test_nn()
