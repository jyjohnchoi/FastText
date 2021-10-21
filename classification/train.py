import numpy as np
import random
import math
from config import Config
from model import FastText, FastTextHS
from model_revised import FastTextR, FastTextHSR
import pickle
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
from distutils.util import strtobool as _bool
import time
from preprocess import *
from eval import evaluate

if __name__ == '__main__':
    cfg = Config()
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=int, default=0)
    parser.add_argument('--bigrams', type=_bool, default=True)
    parser.add_argument('--hs', type=_bool, default=False)
    parser.add_argument('--revised', type=_bool, default=False)
    parser.add_argument('--lr', type=float, default=0.25)
    parser.add_argument('--epochs', type=int, default=5)
    args = parser.parse_args()

    if args.bigrams:
        w2i = pickle.load(open(cfg.word_to_index_path_bigram + "_" + str(args.data_path) + ".pkl", 'rb'))
        dataset, num_labels = pickle.load(open(cfg.dataset_path_bigram + "_" + str(args.data_path) + ".pkl", 'rb'))
    else:
        w2i = pickle.load(open(cfg.word_to_index_path + "_" + str(args.data_path) + ".pkl", 'rb'))
        dataset, num_labels = pickle.load(open(cfg.dataset_path + "_" + str(args.data_path) + ".pkl", 'rb'))

    min_loss = math.inf
    if args.hs:
        tree, depth = huffman_tree(dataset, num_labels)
        if not args.revised:
            model = FastTextHS(len(w2i), cfg.hidden, num_labels, args.lr, tree, depth)
        else:
            model = FastTextHSR(len(w2i), cfg.hidden, num_labels, args.lr, tree, depth)

    else:
        if not args.revised:
            model = FastText(len(w2i), cfg.hidden, num_labels, args.lr)
        else:
            model = FastTextR(len(w2i), cfg.hidden, num_labels, args.lr)

    eval_path = cfg.path_list[args.data_path] + 'test.csv'
    if args.bigrams:
        w2i = pickle.load(open(cfg.word_to_index_path_bigram + "_" + str(args.data_path) + ".pkl", 'rb'))
    else:
        w2i = pickle.load(open(cfg.word_to_index_path + "_" + str(args.data_path) + ".pkl", 'rb'))
    eval_data, _ = generate_test_dataset(eval_path, w2i, ngrams=args.bigrams)

    max_acc = 0
    max_epoch = 0
    start_time = time.time()
    for i in range(args.epochs):
        loss_sum = 0
        random.shuffle(dataset)
        for j, data in tqdm(enumerate(dataset), desc='Training epoch {}'.format(i+1), total=len(dataset), ncols=70):
            x = data[0]
            if len(x) == 0:
                print("Something's going wrong")
                break
            label = data[1]
            loss = model.step(x, label)
            loss_sum += loss
        avg_loss = loss_sum / len(dataset)
        print("Loss for epoch {}: {:.4f}".format(i+1, avg_loss))
        if avg_loss < min_loss:
            min_loss = avg_loss

        total = 0
        correct = 0
        for data in tqdm(eval_data, desc="Evaluating", ncols=70):
            x = data[0]
            if len(x) == 0:
                continue
            answer = data[1]
            total += 1
            correct += int(model.step(x, answer, train=False))
        test_acc = correct / total * 100
        print("Test Accuracy for epoch {}: {:.1f}".format(i+1, test_acc))

        if test_acc > max_acc:
            max_acc = test_acc
            max_epoch = i+1

    print("Data path {} | Bigrams {} | Hierarchical Softmax {} | Revised {} | Learning Rate {} | Epochs {}"
          .format(args.data_path, args.bigrams, args.hs, args.revised, args.lr, args.epochs))
    print("Took {:.2f} minutes".format((time.time() - start_time) / 60))
    print("Max test accuracy {:.1f} at epoch {}".format(max_acc, max_epoch))
    print("\n")



