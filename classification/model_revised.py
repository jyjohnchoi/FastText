import numpy as np
import pickle
from config import Config
import math
from tqdm import tqdm
import matplotlib.pyplot as plt
import logging
from optim import AdaGrad, AdaGradHS

cfg = Config()


class FastTextR(object):
    def __init__(self, vocab_size, dim, num_labels, lr=None):
        in_dim = 10
        self.word_lookup = np.sqrt(2 / (vocab_size + dim)) * np.random.uniform(low=-1, high=1, size=(vocab_size, dim))
        # (V, 10)
        self.linear = np.sqrt(2 / (dim + dim)) * np.random.uniform(low=-1, high=1, size=(in_dim, dim))
        self.bias = np.sqrt(1 / dim) * np.random.uniform(low=-1, high=1, size=(in_dim, ))
        self.linear2 = np.sqrt(2 / (num_labels + dim)) * np.random.uniform(low=-1, high=1, size=(num_labels, in_dim))
        self.bias2 = np.sqrt(1 / num_labels) * np.random.uniform(low=-1, high=1, size=(num_labels, ))
        self.num_labels = num_labels
        self.optimizer = AdaGrad(lr)

    def step(self, input_indices, label, train=True):
        text_rep = np.sum(self.word_lookup[input_indices], axis=0) / len(input_indices)  # (dim, )
        # text_rep_norm = np.linalg.norm(text_rep)
        # text_rep /= text_rep_norm
        y1 = np.dot(self.linear, text_rep) + self.bias  # (dim, )
        y2 = np.dot(self.linear2, y1) # + self.bias2  # (num_labels, )

        if not train:  # No need to calculate softmax
            return np.argmax(y2) == label

        p = self.softmax(y2)
        loss = -math.log(p[label] + 1e-9)
        p[label] -= 1

        dw = np.linalg.multi_dot([self.linear.T, self.linear2.T, p]).squeeze() / len(input_indices)  # / text_rep_norm
        dl = np.linalg.multi_dot([self.linear2.T, np.reshape(p, (-1, 1)), np.reshape(text_rep, (1, -1))])
        db = np.dot(self.linear2.T, p)
        dl2 = np.matmul(np.reshape(p, (-1, 1)), np.reshape(y1, (1, -1)))
        db2 = p
        params = {0: self.word_lookup, 1: self.linear, 2: self.bias, 3: self.linear2, 4: self.bias2}
        grads = {0: dw, 1: dl, 2: db, 3: dl2, 4: db2}
        self.optimizer.step(params, grads, input_indices)

        return loss

    def save_params(self, bigrams, data_path, hs):
        pickle.dump(self.word_lookup, open('./results/softmax/word_lookup_{}_{}.pkl'.format(bigrams, data_path), 'wb'))
        pickle.dump((self.linear, self.bias), open('./results/softmax/linear_{}_{}.pkl'.format(bigrams, data_path),
                                                   'wb'))
        pickle.dump((self.linear2, self.bias2), open('./results/softmax/linear2_{}_{}.pkl'.format(bigrams, data_path),
                                                     'wb'))

    def load(self, word_lookup, linear1, linear2, bias1, bias2):
        self.word_lookup = word_lookup
        self.linear = linear1
        self.bias = bias1
        self.linear2 = linear2
        self.bias2 = bias2

    @ staticmethod
    def softmax(vector):
        max_val = max(vector)
        result = np.exp(np.array(vector)-max_val)
        total = np.sum(result)
        result /= total
        return result


class FastTextHSR(object):
    def __init__(self, vocab_size, dim, num_labels, lr=None, tree=None, max_depth=None):
        in_dim = 128
        self.word_lookup = np.sqrt(2 / in_dim + dim) * np.random.uniform(low=-1, high=1, size=(vocab_size, dim))
        self.linear = np.sqrt(2 / (in_dim + dim)) * np.random.uniform(low=-1, high=1, size=(in_dim, dim))
        self.bias = np.sqrt(1 / in_dim) * np.random.uniform(low=-1, high=1, size=(in_dim, ))
        self.linear2 = np.sqrt(2 / (num_labels - 1 + dim)) * np.random.uniform(low=-1, high=1,
                                                                               size=(num_labels-1, in_dim))
        self.bias2 = np.sqrt(1 / num_labels) * np.random.uniform(low=-1, high=1, size=(num_labels-1, ))
        self.tree = tree
        self.max_depth = max_depth
        self.optimizer = AdaGradHS(lr)

    def step(self, input_indices, label, train=True):
        text_rep = np.sum(self.word_lookup[input_indices], axis=0) / len(input_indices)  # (dim, )
        text_rep_norm = np.linalg.norm(text_rep)
        text_rep /= text_rep_norm
        y1 = np.dot(self.linear, text_rep) + self.bias  # (in_dim, )
        y2 = np.dot(self.linear2, y1) + self.bias2  # (num_labels-1, )
        label_info = self.tree[label]
        depth = label_info[-1]
        directions = label_info[:depth]
        nodes = np.array(label_info[self.max_depth:self.max_depth+depth])
        assert len(directions) == len(nodes), "Direction path and node path must be of same length."
        probs = self.sigmoid(y2[nodes])

        if not train:
            probs = np.abs(probs - directions)
            return min(probs) >= 0.5

        p = directions + probs - 1
        probs = np.abs(probs - directions)
        prob = np.prod(probs)
        loss = -math.log(prob + 1e-9)

        dw = np.linalg.multi_dot([self.linear.T, self.linear2[nodes].T, p]).squeeze() \
             / len(input_indices) / text_rep_norm
        dl = np.linalg.multi_dot([self.linear2[nodes].T, p.reshape(-1, 1), text_rep.reshape(1, -1)])
        db = np.dot(self.linear2[nodes].T, p)
        dl2 = np.dot(np.reshape(p, (-1, 1)), np.reshape(y1, (1, -1)))
        db2 = p
        params = {0: self.word_lookup, 1: self.linear, 2: self.bias, 3: self.linear2, 4: self.bias2}
        grads = {0: dw, 1: dl, 2: db, 3: dl2, 4: db2}
        self.optimizer.step(params, grads, input_indices, nodes)
        return loss

    def save_params(self, bigrams, data_path, hs):
        pickle.dump(self.word_lookup, open('./results/hs/word_lookup_{}_{}.pkl'.format(bigrams, data_path), 'wb'))
        pickle.dump((self.linear, self.bias), open('./results/hs/linear_{}_{}.pkl'.format(bigrams, data_path), 'wb'))
        pickle.dump((self.linear2, self.bias2), open('./results/hs/linear2_{}_{}.pkl'.format(bigrams, data_path), 'wb'))
        pickle.dump((self.tree, self.max_depth), open('./results/hs/tree_{}_{}.pkl'.format(bigrams, data_path), 'wb'))

    def load(self, word_lookup, linear, bias, linear2, bias2, tree, max_depth):
        self.word_lookup = word_lookup
        self.linear = linear
        self.bias = bias
        self.linear2 = linear2
        self.bias2 = bias2
        self.tree = tree
        self.max_depth = max_depth

    @ staticmethod
    def sigmoid(vector):
        return 1 / (1 + np.exp(-vector))
