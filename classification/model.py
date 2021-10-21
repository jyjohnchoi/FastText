import numpy as np
import pickle
from config import Config
import math
from tqdm import tqdm
import matplotlib.pyplot as plt
import logging
from optim import AdaGrad, AdaGradHS, SGD

cfg = Config()


class FastText(object):
    def __init__(self, vocab_size, dim, num_labels, lr=None):
        self.word_lookup = np.sqrt(2 / (vocab_size + dim)) * np.random.uniform(low=-1, high=1, size=(vocab_size, dim))
        self.linear = np.sqrt(2 / (num_labels + dim)) * np.random.uniform(low=-1, high=1, size=(num_labels, dim))
        self.bias = np.sqrt(1 / num_labels) * np.random.uniform(low=-1, high=1, size=(num_labels, ))
        # self.word_lookup = 0.01 * np.random.normal(0, 1, size=(vocab_size, dim))
        # self.linear = np.random.normal(0, np.sqrt(2 / (num_labels + dim)), size=(num_labels, dim))
        # self.bias = np.random.uniform(0, np.sqrt(1 / num_labels), size=(num_labels, ))
        self.num_labels = num_labels
        self.lr = lr
        self.optimizer = AdaGrad(self.lr)
        # self.optimizer = SGD(self.lr)

    def step(self, input_indices, label, train=True):
        input_indices = [i for i in input_indices if i != 0]
        text_rep = np.sum(self.word_lookup[input_indices], axis=0) / len(input_indices)
        linear_output = np.matmul(self.linear, text_rep) + self.bias

        if not train:  # No need to calculate softmax
            return np.argmax(linear_output) == label

        p = self.softmax(linear_output)
        loss = -math.log(p[label] + 1e-9)
        p[label] -= 1

        dw = np.matmul(self.linear.T, p).squeeze() / len(input_indices)
        dl = np.matmul(p.reshape(-1, 1), text_rep.reshape(1, -1))
        db = p

        params = {0: self.word_lookup, 1: self.linear, 2: self.bias}
        grads = {0: dw, 1: dl, 2: db}
        self.optimizer.step(params, grads, input_indices)

        return loss

    def save_params(self, bigrams, data_path, hs):
        pickle.dump(self.word_lookup, open('./results/softmax/word_lookup_{}_{}.pkl'.format(bigrams, data_path), 'wb'))
        pickle.dump((self.linear, self.bias), open('./results/softmax/linear_{}_{}.pkl'.format(bigrams, data_path),
                                                   'wb'))

    def load(self, word_lookup, linear, bias):
        self.word_lookup = word_lookup
        self.linear = linear
        self.bias = bias

    @ staticmethod
    def softmax(vector):
        max_val = max(vector)
        result = np.exp(np.array(vector)-max_val)
        total = np.sum(result)
        result /= total
        return result


class FastTextHS(object):
    def __init__(self, vocab_size, dim, num_labels, lr=None, tree=None, max_depth=None):
        self.word_lookup = np.sqrt(2 / (vocab_size + dim)) * np.random.uniform(low=-1, high=1, size=(vocab_size, dim))
        self.linear = np.sqrt(2 / (num_labels + dim)) * np.random.uniform(low=-1, high=1, size=(num_labels-1, dim))
        self.bias = np.sqrt(1 / num_labels) * np.random.uniform(low=-1, high=1, size=(num_labels-1, ))
        # self.word_lookup = np.random.normal(0, np.sqrt(2 / (vocab_size + dim)), size=(vocab_size, dim))
        # self.linear = np.random.normal(0, np.sqrt(2 / (num_labels + dim - 1)), size=(num_labels-1, dim))
        # self.bias = np.random.uniform(0, np.sqrt(1 / (num_labels-1)), size=(num_labels-1, ))
        self.tree = tree
        self.max_depth = max_depth
        self.lr = lr
        self.optimizer = AdaGradHS(self.lr)

    def step(self, input_indices, label, train=True):
        self.optimizer.set_lr(self.lr)
        text_rep = np.sum(self.word_lookup[input_indices], axis=0) / len(input_indices)
        linear_output = np.dot(self.linear, text_rep) + self.bias
        label_info = self.tree[label]
        depth = label_info[-1]
        directions = label_info[:depth]
        nodes = np.array(label_info[self.max_depth:self.max_depth+depth])
        probs = self.sigmoid(linear_output[nodes])

        assert len(directions) == len(nodes), "Direction path and node path must be of same length."

        if not train:
            probs = np.abs(probs - directions)
            return min(probs) > 0.5

        p = directions + probs - 1
        probs = np.abs(probs - directions)
        prob = np.prod(probs)
        loss = -math.log(prob + 1e-9)

        dw = np.dot(np.transpose(self.linear[nodes, :]), p).squeeze() / len(input_indices)
        dl = np.dot(np.reshape(p, (-1, 1)), np.reshape(text_rep, (1, -1)))
        db = p
        params = {0: self.word_lookup, 3: self.linear, 4: self.bias}
        grads = {0: dw, 3: dl, 4: db}
        self.optimizer.step(params, grads, input_indices, nodes)
        return loss

    def save_params(self, bigrams, data_path, hs):
        pickle.dump(self.word_lookup, open('./results/hs/word_lookup_{}_{}.pkl'.format(bigrams, data_path), 'wb'))
        pickle.dump((self.linear, self.bias), open('./results/hs/linear_{}_{}.pkl'.format(bigrams, data_path), 'wb'))
        pickle.dump((self.tree, self.max_depth), open('./results/hs/tree_{}_{}.pkl'.format(bigrams, data_path), 'wb'))

    def load(self, word_lookup, linear, bias, tree, max_depth):
        self.word_lookup = word_lookup
        self.linear = linear
        self.bias = bias
        self.tree = tree
        self.max_depth = max_depth

    @ staticmethod
    def sigmoid(vector):
        return 1 / (1 + np.exp(-vector))
