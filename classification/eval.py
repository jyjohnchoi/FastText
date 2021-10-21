import numpy as np
import pandas as pd
from preprocess import *
import pickle
from config import Config
from model import FastText, FastTextHS
from model_revised import FastTextR, FastTextHSR
import argparse
from distutils.util import strtobool as _bool

"""
if __name__ == '__main__':
    cfg = Config()
    path_list = cfg.path_list
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=int, default=0)
    parser.add_argument('--bigrams', type=_bool, default=True)
    parser.add_argument('--hs', type=_bool, default=False)
    parser.add_argument('--revised', type=_bool, default=False)
    args = parser.parse_args()
    eval_path = path_list[args.data_path] + 'test.csv'
    if args.bigrams:
        w2i = pickle.load(open(cfg.word_to_index_path_bigram + "_" + str(args.data_path) + ".pkl", 'rb'))
    else:
        w2i = pickle.load(open(cfg.word_to_index_path + "_" + str(args.data_path) + ".pkl", 'rb'))
    eval_data, num_labels = generate_dataset(eval_path, w2i, ngrams=args.bigrams)

    if not args.hs:
        word_lookup = pickle.load(
            open('./results/softmax/word_lookup_{}_{}.pkl'.format(args.bigrams, args.data_path), 'rb'))
        linear, bias = pickle.load(
            open('./results/softmax/linear_{}_{}.pkl'.format(args.bigrams, args.data_path), 'rb'))
        if not args.revised:
            model = FastText(len(w2i), cfg.hidden, num_labels)
            model.load(word_lookup, linear, bias)
        else:
            model = FastTextR(len(w2i), cfg.hidden, num_labels)
            linear2, bias2 = pickle.load(
                open('./results/softmax/linear2_{}_{}.pkl'.format(args.bigrams, args.data_path), 'rb'))
            model.load(word_lookup, linear, linear2, bias, bias2)
    else:
        word_lookup = pickle.load(
            open('./results/hs/word_lookup_{}_{}.pkl'.format(args.bigrams, args.data_path), 'rb'))
        linear, bias = pickle.load(
            open('./results/hs/linear_{}_{}.pkl'.format(args.bigrams, args.data_path), 'rb'))
        tree, max_depth = pickle.load(
            open('./results/hs/tree_{}_{}.pkl'.format(args.bigrams, args.data_path), 'rb'))
        if not args.revised:
            model = FastTextHS(len(w2i), cfg.hidden, num_labels)
            model.load(word_lookup, linear, bias, tree, max_depth)
        else:
            model = FastTextHSR(len(w2i), cfg.hidden, num_labels)
            linear2, bias2 = pickle.load(
                open('./results/hs/linear2_{}_{}.pkl'.format(args.bigrams, args.data_path), 'rb'))

            model.load(word_lookup, linear, bias, linear2, bias2, tree, max_depth)

    total = 0
    correct = 0
    for data in tqdm(eval_data, desc="Evaluating", ncols=70):
        x = data[0]
        answer = data[1]
        total += 1
        correct += int(model.step(x, answer, train=False))
    print(total)
    print(correct)
    print("Test Accuracy: {:.1f}".format(correct/total * 100))
    print("\n\n")
"""


def evaluate(data_path, bigrams, hs, revised):
    """
    cfg = Config()
    eval_path = cfg.path_list[data_path] + 'test.csv'
    if bigrams:
        w2i = pickle.load(open(cfg.word_to_index_path_bigram + "_" + str(data_path) + ".pkl", 'rb'))
    else:
        w2i = pickle.load(open(cfg.word_to_index_path + "_" + str(data_path) + ".pkl", 'rb'))
    eval_data, num_labels = generate_dataset(eval_path, w2i, ngrams=bigrams)
    """
    if not hs:
        word_lookup = pickle.load(
            open('./results/softmax/word_lookup_{}_{}.pkl'.format(bigrams, data_path), 'rb'))
        linear, bias = pickle.load(
            open('./results/softmax/linear_{}_{}.pkl'.format(bigrams, data_path), 'rb'))
        if not revised:
            model = FastText(len(w2i), cfg.hidden, num_labels)
            model.load(word_lookup, linear, bias)
        else:
            model = FastTextR(len(w2i), cfg.hidden, num_labels)
            linear2, bias2 = pickle.load(
                open('./results/softmax/linear2_{}_{}.pkl'.format(bigrams, data_path), 'rb'))
            model.load(word_lookup, linear, linear2, bias, bias2)
    else:
        word_lookup = pickle.load(
            open('./results/hs/word_lookup_{}_{}.pkl'.format(bigrams, data_path), 'rb'))
        linear, bias = pickle.load(
            open('./results/hs/linear_{}_{}.pkl'.format(bigrams, data_path), 'rb'))
        tree, max_depth = pickle.load(
            open('./results/hs/tree_{}_{}.pkl'.format(bigrams, data_path), 'rb'))
        if not revised:
            model = FastTextHS(len(w2i), cfg.hidden, num_labels)
            model.load(word_lookup, linear, bias, tree, max_depth)
        else:
            model = FastTextHSR(len(w2i), cfg.hidden, num_labels)
            linear2, bias2 = pickle.load(
                open('./results/hs/linear2_{}_{}.pkl'.format(bigrams, data_path), 'rb'))

            model.load(word_lookup, linear, bias, linear2, bias2, tree, max_depth)

    total = 0
    correct = 0
    for data in tqdm(eval_data, desc="Evaluating", ncols=70):
        x = data[0]
        answer = data[1]
        total += 1
        correct += int(model.step(x, answer, train=False))
    print(total)
    print(correct)
    print("Test Accuracy: {:.1f}".format(correct / total * 100))
    print("\n\n")
