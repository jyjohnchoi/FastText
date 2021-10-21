import argparse
from distutils.util import strtobool as _bool
import numpy as np
# from torch.utils.tensorboard import SummaryWriter
import os
from preprocess import *


def train(epochs=5, subsampling_t=1e-4):
    cfg = Config()
    random.seed(1128)
    hidden_size = cfg.hidden_size
    neg_num = 5

    # log_dir = 'log/{}_{}epoch.tb'.format(subsampling_t, epochs)
    # if not os.path.exists(log_dir):
    #     os.mkdir(log_dir)
    # writer = SummaryWriter(log_dir)
    # log_steps = 1000000

    if not os.path.isfile(cfg.freq_path):
        create_dictionary(cfg.train_path)
    frequency = pickle.load(open(cfg.freq_path, 'rb'))
    ngram_to_index = pickle.load(open(cfg.ngram_to_index_path, 'rb'))
    word_to_ngram_indices = pickle.load(open(cfg.word_to_ngram_indices_path, 'rb'))
    ngram_size = len(ngram_to_index)
    vocab_size = len(frequency)
    unigram_table = np.array(pickle.load(open(cfg.unigram_table_path, 'rb')))
    len_unigram_table = len(unigram_table)
    total = sum([item[1] for item in frequency.items()])

    w_in = np.random.uniform(low=-0.5 / 300, high=0.5 / 300, size=(ngram_size, hidden_size)).astype('f')
    w_out = np.zeros_like(w_in).astype('f')  # for negative sampling

    starting_lr = 0.025
    min_loss = math.inf

    print("Start training on {} words".format(vocab_size))
    step = 0
    # logging_loss = 0
    start_time = time.time()
    lr = starting_lr
    for epoch in range(epochs):
        data_paths = []
        total_pairs = 0
        print("======= Epoch {} training =======".format(epoch + 1))
        for i in range(len(cfg.train_files)):
            path = cfg.train_files[i]
            print("======= File number {} =======".format(i + 1))
            dataset = preprocess(path=path, frequency=frequency, total=total, subsampling_t=subsampling_t)
            data_path = "./preprocessed/data_{}.pkl".format(i)
            pickle.dump(dataset, open(data_path, 'wb'))
            total_pairs += len(dataset)
            data_paths.append(data_path)

        for i, data_path in enumerate(data_paths):
            print("======= File number {} =======".format(i + 1))
            print("Learning rate: {:.4f}".format(lr))

            dataset = pickle.load(open(data_path, 'rb'))
            loss = 0

            lr_update_count = 0
            file_start_time = time.time()
            for input_idx, tgt_idx in tqdm(dataset, desc="Training", ncols=70):
                lr_update_count += 1
                if lr_update_count == 10000:
                    lr -= starting_lr * 10000 / (total_pairs * epochs)
                    if lr < starting_lr * 1e-4:
                        lr = starting_lr * 1e-4
                    lr_update_count = 0
                input_idx = np.array(input_idx)
                hidden = np.sum(w_in[input_idx], axis=0)  # (300, )
                hidden = hidden.reshape(1, 300)  # (1, 300)

                while 1:
                    negs = np.random.randint(low=0, high=len_unigram_table, size=neg_num)
                    negs = unigram_table[negs]
                    if tgt_idx in negs:
                        continue
                    else:
                        break
                tgt_indices = np.append(tgt_idx, negs)  # 6개
                target_indices_list = []  # list of lists, each list is n-gram indices for target word and neg samples
                targets = []  # list containing all elements of target_indices_list
                for target in tgt_indices:
                    target_ngrams = word_to_ngram_indices[target]
                    targets.extend(target_ngrams)
                    target_indices_list.append(target_ngrams)

                ct = np.zeros((1 + neg_num, hidden_size))
                for j, target_indices in enumerate(target_indices_list):
                    ct[j] = np.sum(w_out[target_indices], axis=0)  # Sum of n-grams for positive and 5 negative samples
                out = sigmoid(np.dot(hidden, ct.T)).squeeze()  # (1, 6)
                p_loss = -np.log(out[0] + 1e-7)
                n_loss = -np.sum(np.log(1 - out[1:] + 1e-7))
                loss += (p_loss.item() + n_loss.item())
                # logging_loss += (p_loss.item() + n_loss.item())

                out[0] -= 1
                dout = np.tile(out, (len(input_idx), 1))  # (#input indices, 6)
                context_grad = np.dot(dout.T, w_in[input_idx])  # (6, 1) * (1, 300) = (6, 300)
                emb_grad = np.dot(dout, ct)
                for j, target in enumerate(target_indices_list):
                    w_out[target] -= lr * context_grad[j] / 64
                w_in[input_idx] -= lr * emb_grad / 64
                step += 1
                # if step % 100000 == 0:
                #     print(w_in)
                # if step % log_steps == 0:
                #     writer.add_scalar('Training loss', logging_loss / log_steps, int((step - 1) / log_steps))
                #     logging_loss = 0

            print("Loss: {:.5f}".format(loss / len(dataset)))
            print("Took {:.2f} hours for single file".format((time.time() - file_start_time) / 3600))

            if loss < min_loss:
                min_loss = loss
                pickle.dump(w_in, open("./results/embedding_{}_{}epochs".format(subsampling_t, epochs), 'wb'))
                print("Embedding matrix saved!")
            similar_word(w_in)

    print("Took {:.2f} hours".format((time.time() - start_time) / 3600))
    return w_in, w_out


def sigmoid(xs):
    ans = 1 / (1 + np.exp(-xs))
    top = 1 / (1 + math.exp(6))
    bottom = 1 / (1 + math.exp(-6))
    for i, num in enumerate(ans[0]):
        if num < top:
            ans[0, i] = 0
        elif num > bottom:
            ans[0, i] = 1
    return ans


def similar_word(emb):
    index_to_word = pickle.load(open(cfg.index_to_word_path, 'rb'))
    word_to_index = pickle.load(open(cfg.word_to_index_path, 'rb'))
    word_to_ngram_indices = pickle.load(open(cfg.word_to_ngram_indices_path, 'rb'))
    embedding_norm = np.linalg.norm(emb, axis=1)
    norm_emb = emb / embedding_norm[:, None]
    w1 = word_to_index['king']
    w2 = word_to_index['queen']
    w3 = word_to_index['husband']
    ans = word_to_index['wife']
    word1 = word_to_ngram_indices[w1]
    word2 = word_to_ngram_indices[w2]
    word3 = word_to_ngram_indices[w3]
    answer = word_to_ngram_indices[ans]

    target = np.sum(norm_emb[word2], axis=0) - np.sum(norm_emb[word1], axis=0) + np.sum(norm_emb[word3], axis=0)
    target = target / np.linalg.norm(target)

    max_sim = cosine_similarity(target, np.sum(norm_emb[answer], axis=0))
    print(max_sim)
    max_index = word_to_index['wife']
    for i in tqdm(range(len(word_to_index)), desc="Finding closest word to queen-king+husband", ncols=70):
        if i == w1 or i == w2 or i == w3 or i == ans:
            pass
        else:
            sim = cosine_similarity(np.sum(norm_emb[word_to_ngram_indices[i]], axis=0), target)
            if sim > max_sim:
                max_sim = sim
                max_index = i
    print(index_to_word[max_index])


def cosine_similarity(v1, v2):
    m1 = np.linalg.norm(v1)
    m2 = np.linalg.norm(v2)
    return np.dot(v1, v2) / (m1 * m2)


if __name__ == '__main__':
    train(epochs=5)
    # emb = pickle.load(open('./results/embedding_0.0001_3epochs', 'rb'))
    # similar_word(emb)
