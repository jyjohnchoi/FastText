import numpy as np
import random
import math
import pandas as pd
import re
import pickle
from config import Config
from tqdm import tqdm
import argparse
from distutils.util import strtobool as _bool
import heapq
from nltk.tokenize import word_tokenize
import nltk


cfg = Config()


def tokenize(sent, ngrams):
    split_tokens = ['\t', '\v', '\r', '\f', '\0']
    punctuation = ['.', ',', '!', '/', ':', ';',
                   '+', '-', '*', '?', '~', '|',
                   '[', ']', '{', '}', '(', ')',
                   '_', '=', '%', '&', '$', '#',
                   '"', '`', '^', "'", '\\', '<', '>']
    for split_token in split_tokens:
        sent.replace(split_token, ' ')
    for p in punctuation:
        # sent = sent.replace(p, ' ' + p + ' ')  # 남기기
        sent = sent.replace(p, ' ')  # 제거
    tokens = sent.split()
    # tokens = word_tokenize(sent)
    if ngrams:
        tokens += ['{}_{}'.format(tokens[i], tokens[i + 1]) for i in range(len(tokens) - 1)]
    return tokens


def make_dict(data_path, ngrams):
    data_path += 'train.csv'
    word_to_index = {'<unk>': 0, '<s>': 1, '</s>': 2}
    index_to_word = ['<unk>', '<s>', '</s>']
    dataset = []
    df = pd.read_csv(data_path, names=['Label', 'A', 'B', 'C'])
    text1 = df.A.to_list()
    text2 = df.B.to_list()
    text3 = df.C.to_list()
    labels = df.Label.to_list()

    for a, b, c, label in tqdm(zip(text1, text2, text3, labels), desc="Creating Dictionary and Dataset",
                               total=len(labels), ncols=70):
        final_input = []
        texts = []
        if isinstance(a, str):
            texts.append(a.strip())
        if isinstance(b, str):
            texts.append(b.strip())
        if isinstance(c, str):
            texts.append(c.strip())
        if not isinstance(label, int):
            continue
        for text in texts:
            tokens = tokenize(text, ngrams)
            for token in tokens:
                if token not in word_to_index.keys():
                    word_to_index[token] = len(index_to_word)
                    index_to_word.append(token)
            input_idx = sent_to_idx(text, word_to_index, ngrams)
            final_input.extend(input_idx)
        pair = (final_input, label-1)
        dataset.append(pair)
    print("Number of words in dictionary: {}".format(len(word_to_index)))
    n_labels = len(set(labels))
    print("Dataset size: {}".format(len(dataset)))
    print("Number of labels: {}".format(n_labels))
    return dataset, n_labels, word_to_index, index_to_word


def sent_to_idx(sent, word_to_index, ngrams):
    sent_indices = [1]
    tokens = tokenize(sent, ngrams)
    for token in tokens:
        if token not in word_to_index.keys():
            sent_indices.append(0)
        else:
            sent_indices.append(word_to_index[token])
    sent_indices.append(2)
    return sent_indices


def generate_test_dataset(data_path, word_to_index, ngrams):
    dataset = []
    df = pd.read_csv(data_path, names=['Label', 'A', 'B', 'C'])
    text1 = df.A.to_list()
    text2 = df.B.to_list()
    text3 = df.C.to_list()
    labels = df.Label.to_list()
    for a, b, c, label in tqdm(zip(text1, text2, text3, labels), desc='Generating dataset',
                               total=len(labels), ncols=70):
        final_input = []
        texts = []
        if isinstance(a, str):
            texts.append(a.strip())
        if isinstance(b, str):
            texts.append(b.strip())
        if isinstance(c, str):
            texts.append(c.strip())
        if len(texts) == 0:
            continue
        if not isinstance(label, int):
            continue
        for text in texts:
            input_idx = sent_to_idx(text, word_to_index, ngrams)
            final_input.extend(input_idx)
        pair = (final_input, label-1)
        dataset.append(pair)
    n_labels = len(set(labels))
    print("Dataset size: {}".format(len(dataset)))
    print("Number of labels: {}".format(n_labels))
    return dataset, n_labels


def huffman_tree(dataset, n_labels):
    freq = {i: 0 for i in range(n_labels)}
    for pair in dataset:
        freq[pair[1]] += 1
    length = len(freq)
    heap = [[item[1], i] for i, item in enumerate(freq.items())]
    heapq.heapify(heap)
    for i in tqdm(range(length - 1), desc="Creating Huffman Tree", ncols=70):
        min1 = heapq.heappop(heap)
        min2 = heapq.heappop(heap)
        heapq.heappush(heap, [min1[0] + min2[0], i + length, min1, min2])

    # node of heap : [frequency, index, left child, right child]
    word_stack = []
    stack = [[heap[0], [], []]]
    max_depth = 0

    while len(stack) != 0:
        node, direction_path, node_path = stack.pop()

        if node[1] >= length:
            current_node = [node[1] - length]
            stack.append([node[2], direction_path + [0], node_path + current_node])
            stack.append([node[3], direction_path + [1], node_path + current_node])

        else:  # leaf node of tree
            node.append(np.array(direction_path))
            node.append(np.array(node_path))
            max_depth = max(max_depth, len(direction_path))
            word_stack.append(node)

    # sort by index to fit with frequency order
    word_stack = np.array(sorted(word_stack, key=lambda items: items[1]))
    word_stack = word_stack[:, 2:4]  # only paths

    paths = np.zeros((length, 2 * max_depth + 1)).astype(int)
    for i in tqdm(range(length), desc="Padding paths...", ncols=70):
        true_depth = len(word_stack[i, 0])
        paths[i, 0:true_depth] = word_stack[i, 0]
        paths[i, max_depth:max_depth + true_depth] = word_stack[i, 1]
        paths[i, -1] = true_depth

    return paths, max_depth


if __name__ == '__main__':
    path_list = cfg.path_list
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=int, default=0)
    parser.add_argument('--bigrams', type=_bool, default=True)
    args = parser.parse_args()
    path = path_list[args.data_path]
    data, num_labels, w2i, i2w = make_dict(path, args.bigrams)
    paths, depth = huffman_tree(data, num_labels)
    if args.bigrams:
        pickle.dump(w2i, open(cfg.word_to_index_path_bigram + "_" + str(args.data_path) + ".pkl", 'wb'))
        pickle.dump((data, num_labels), open(cfg.dataset_path_bigram + "_" + str(args.data_path) + ".pkl", 'wb'))
    else:
        pickle.dump(w2i, open(cfg.word_to_index_path + "_" + str(args.data_path) + ".pkl", 'wb'))
        pickle.dump((data, num_labels), open(cfg.dataset_path + "_" + str(args.data_path) + ".pkl", 'wb'))
    print("\n")

