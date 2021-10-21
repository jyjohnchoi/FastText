import numpy as np
import math
from tqdm.auto import tqdm
import pickle
import random
import os
import time
from config import Config
if __name__ == '__main__':
    import sys
    sys.path.append(os.getcwd())


cfg = Config()


def ngrams(word, n):
    target = '<' + word + '>'
    if len(target) < n:
        return [target]
    else:
        return [target[i:i+n] for i in range(len(target)-n + 1)] + [target]

# -*- coding: utf-8 -*-


def tokenize(sent):
    split_tokens = ['\t', '\v', '\r', '\f', '\0']
    punctuation = ['.', ',', '!', '/', ':', ';',
                   '+', '-', '*', '?', '~', '|',
                   '[', ']', '{', '}', '(', ')',
                   '_', '=', '%', '&', '$', '#',
                   '"', '`', '^', "'", '\\', '<', '>']
    for split_token in split_tokens:
        sent.replace(split_token, ' ')
    for p in punctuation:
        sent = sent.replace(p, ' ' + p + ' ')  # 남기기
        # sent = sent.replace(p, ' ')  # 제거
    tokens = sent.split()
    return tokens


def create_dictionary(train_path):
    """
    @param train_path : list of paths to training file
    Creates a dictionary including every word from the corpus, and save it.
    """
    # frequency = {'</s>': 0}
    frequency = {}
    with open(train_path, 'rt', encoding="UTF-8") as f:
        lines = f.readlines()
        for line in tqdm(lines, "Creating dictionary from training data", ncols=80):
            if line.startswith("=="):
                continue
            words_in_line = tokenize(line)
            for word in words_in_line:
                if word in frequency.keys():
                    frequency[word] += 1
                else:
                    frequency[word] = 1

    # list of tuples (word, frequency), ordered by max to min via frequency
    frequency = sorted(frequency.items(), key=lambda item: item[1], reverse=True)
    frequency = {vocab: i for vocab, i in frequency if i >= cfg.MIN_COUNT}  # vocab freq < MIN_COUNT removed

    word_to_ngrams = {}
    for word in frequency.keys():
        if word in word_to_ngrams.keys():
            pass
        else:
            word_ngrams = []
            for n in range(3, 7):
                word_ngrams.extend(ngrams(word, n))
            word_to_ngrams[word] = set(word_ngrams)

    word_to_index = {word: i for i, word in enumerate(frequency.keys())}
    index_to_word = list(word_to_index.keys())

    ngram_to_index = {('<{}>'.format(word)): i for i, word in enumerate(frequency.keys())}
    for word in frequency.keys():
        for ngram in word_to_ngrams[word]:
            if ngram not in ngram_to_index.keys():
                ngram_to_index[ngram] = len(ngram_to_index)

    word_to_ngram_indices = {}
    for word in word_to_ngrams.keys():
        word_idx = word_to_index[word]
        ngram_indices = []
        for ngram in word_to_ngrams[word]:
            ngram_indices.append(ngram_to_index[ngram])
        word_to_ngram_indices[word_idx] = ngram_indices

    pickle.dump(frequency, open(cfg.freq_path, 'wb'))
    pickle.dump(word_to_index, open(cfg.word_to_index_path, 'wb'))
    pickle.dump(index_to_word, open(cfg.index_to_word_path, 'wb'))
    pickle.dump(word_to_ngrams, open(cfg.word_to_ngrams_path, 'wb'))
    pickle.dump(ngram_to_index, open(cfg.ngram_to_index_path, 'wb'))
    pickle.dump(word_to_ngram_indices, open(cfg.word_to_ngram_indices_path, 'wb'))
    print("Frequencies and indices saved!")

    table = init_unigram_table(frequency, word_to_index)

    pickle.dump(table, open(cfg.unigram_table_path, 'wb'))
    print("Negative sample table saved!")

    print("Number of vocabulary: {}".format(len(frequency)))  # {word: frequency}
    print("Number of vocabulary + n-grams: {}".format(len(ngram_to_index)))  # {ngram: index}, full words in <word> form
    print("len(word_to_index): {}".format(len(word_to_index)))  # {word: index}
    print("len(index_to_word): {}".format(len(index_to_word)))  # [word]
    print("len(word_to_ngrams): {}".format(len(word_to_ngrams))) # {word: [ngrams]}
    print("len(word_to_ngram_indices): {}".format(len(word_to_ngram_indices))) # {word_index: [ngram_indices]}


def generate_training_data(sentence, word_to_index, word_to_ngram_indices,
                           window_size, frequency, total, subsampling_t):
    sentence = tokenize(sentence)
    length = len(sentence)
    data = []
    # Dynamic window scaling. This allows updating data by every epoch
    for i, target in enumerate(sentence):
        if target not in word_to_index.keys():
            continue
        if subsampling_t:
            if subsampling(target, frequency, total, threshold=subsampling_t):
                continue
        window_size = random.randint(1, window_size + 1)
        nbr_indices = list(range(max(0, i - window_size), i)) + list(range(i + 1, min(length, i + window_size + 1)))

        for idx in nbr_indices:
            if sentence[idx] in word_to_index.keys():

                data.append((word_to_ngram_indices[word_to_index[sentence[idx]]], word_to_index[target]))
    return data


def preprocess(path, frequency, total, window_size=5, subsampling_t=1e-4, test=False):
    word_to_index = pickle.load(open(cfg.word_to_index_path, 'rb'))
    word_to_ngram_indices = pickle.load(open(cfg.word_to_ngram_indices_path, 'rb'))
    start_time = time.time()
    dataset = list()
    with open(path, 'rt', encoding="UTF-8") as f:
        lines = f.readlines()
        if test:
            lines = lines[:100]
        for line in tqdm(lines, desc="Generating input, output pairs", ncols=70):
            data = generate_training_data(line, word_to_index, word_to_ngram_indices,
                                          window_size=window_size, subsampling_t=subsampling_t,
                                          frequency=frequency, total=total)
            dataset.extend(data)
    random.shuffle(dataset)
    print("Data generated including {0} pairs, took {1:0.3f} minutes.".format(len(dataset),
                                                                              (time.time() - start_time) / 60))
    return dataset


def subsampling(word, frequency, total, threshold=1e-4):
    freq = frequency[word]
    ratio = freq / (threshold * total)
    p = (np.sqrt(ratio) + 1) / ratio
    draw = random.random()
    return p < draw


def init_unigram_table(frequency, word_to_index, power=0.5):
    """
    Return a uni-gram table from the index of word to its probability of appearance.
    P(w) = count(w)^power / sum(count^power)
    """
    table = []
    for word in tqdm(frequency.keys(), desc="Generating unigram table", ncols=70):
        if word == '</s>':
            continue
        occurrence = int(math.pow(frequency[word], power))
        idx = word_to_index[word]
        table.extend([idx] * occurrence)
    print(len(table))
    return table


if __name__ == '__main__':
    print("No implementation")
    # sentence = "This is an example sentence."
    # word_to_index = pickle.load(open(cfg.word_to_index_path, 'rb'))
    # word_to_ngram_indices = pickle.load(open(cfg.word_to_ngram_indices_path, 'rb'))
    # window_size = 5
    # frequency = pickle.load(open(cfg.freq_path, 'rb'))
    # total = sum([item[1] for item in frequency.items()])
    # subsampling_t = 1e-4
    # print(generate_training_data(sentence, word_to_index, word_to_ngram_indices,
    #                              window_size, frequency, total, subsampling_t))
