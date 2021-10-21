import pandas as pd
from tqdm import tqdm
from nltk.tokenize import word_tokenize
from preprocess import *


cfg = Config()


def make_dict(data_path, ngrams):
    data_path += 'train.csv'
    df = pd.read_csv(data_path)
    print(len(df.columns))
    if len(df.columns) == 2:
        df = pd.read_csv(data_path, names=['Label', 'Text'])
    elif len(df.columns) == 3:
        df = pd.read_csv(data_path, names=['Label', 'Title', 'Text'])
    else:
        df = pd.read_csv(data_path, names=['Label', 'A', 'B', 'Text'])
    print(df.columns)
    print(len(df))
    a = df.A.to_list()
    b = df.B.to_list()
    c = df.Text.to_list()
    labels = df.Label.to_list()
    word_to_index = {}
    index_to_word = {}
    dataset = []
    for t1, t2, t3, label in tqdm(zip(a, b, c, labels), desc="Creating Dictionary and Dataset", total=len(c), ncols=70):
        text = ""
        if isinstance(t1, str):
            text += t1
        if isinstance(t2, str):
            text += t2
        if isinstance(t3, str):
            text += t3
        if len(text) == 0:
            print("TEXT", text)
        if not isinstance(label, int):
            continue

        tokens = tokenize(text, ngrams)
        for token in tokens:
            if token not in word_to_index.keys():
                word_to_index[token] = len(index_to_word)
                index_to_word[len(index_to_word)] = token
        input_idx = sent_to_idx(text, word_to_index, ngrams)
        if len(input_idx) == 0:
            continue
        pair = (input_idx, label-1)
        dataset.append(pair)

    print("Number of words in dictionary: {}".format(len(word_to_index)))
    n_labels = len(set(labels))
    print("Dataset size: {}".format(len(dataset)))
    print("Number of labels: {}".format(n_labels))
    if ngrams:
        pickle.dump(word_to_index, open(cfg.word_to_index_path_bigram + "_" + str(data_path) + ".pkl", 'wb'))
        pickle.dump((dataset, num_labels), open(cfg.dataset_path_bigram + "_" + str(data_path) + ".pkl", 'wb'))
    else:
        pickle.dump(word_to_index, open(cfg.word_to_index_path + "_" + str(data_path) + ".pkl", 'wb'))
        pickle.dump((dataset, num_labels), open(cfg.dataset_path + "_" + str(data_path) + ".pkl", 'wb'))
    return dataset, n_labels, word_to_index, index_to_word


if __name__ == '__main__':
    # make_dict('./dataset/ag_news_csv/', ngrams=True)
    # make_dict('./dataset/sogou_news_csv/', ngrams=True)
    # make_dict('./dataset/dbpedia_csv/', ngrams=True)
    # make_dict('./dataset/yelp_review_full_csv/', ngrams=True)
    # make_dict('./dataset/yelp_review_polarity_csv/', ngrams=True)
    make_dict('./dataset/yahoo_answers_csv/', ngrams=True)
    # make_dict('./dataset/amazon_review_full_csv/', ngrams=True)
    # make_dict('./dataset/amazon_review_polarity_csv/', ngrams=True)

