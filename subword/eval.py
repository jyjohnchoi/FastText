import pickle
import numpy as np
from tqdm import tqdm
from config import Config
import argparse

cfg = Config()


def preprocess_file(test_data):
    with open(test_data) as f:
        lines = f.readlines()
    result = []
    current_set = []
    for line in lines:
        line = line.strip()
        line_words = line.split()
        if line_words[0] == ':' \
                            '':
            if len(current_set) > 0:
                result.append(current_set)
            current_set = []
        else:
            current_set.append(line_words)
    result.append(current_set)

    semantic_temp = result[:5]
    syntactic_temp = result[5:]
    semantic = []
    syntactic = []
    for category in semantic_temp:
        semantic.extend(category)
    for category in syntactic_temp:
        syntactic.extend(category)

    return semantic, syntactic


def test_words(data, embedding_path):
    word_to_index = pickle.load(open(cfg.word_to_index_path, 'rb'))
    embedding = pickle.load(open(embedding_path, 'rb'))
    index_to_word = pickle.load(open(cfg.index_to_word_path, 'rb'))
    embedding = get_word_embeddings(embedding, len(word_to_index))
    embedding_norm = np.linalg.norm(embedding, axis=1)
    embedding_normalized = embedding / embedding_norm[:, None]
    correct = 0
    count = 0
    # for question in data:
    for question in tqdm(data, desc="Evaluating word2vec embedding", ncols=70):
        indices = []
        for word in question:
            if word not in word_to_index.keys():
                break
            index = word_to_index[word]
            indices.append(index)
        # OOV -> wrong.
        if len(indices) < 4:
            continue
        count += 1
        output_vec = embedding_normalized[indices[1]] - embedding_normalized[indices[0]] \
                    + embedding_normalized[indices[2]]
        label_idx = indices[3]

        cos_sim = np.dot(embedding_normalized, output_vec / np.linalg.norm(output_vec))
        sort_idx = (-cos_sim).argsort()
        answer_cands = sort_idx[:4]
        answer = -1
        for idx in answer_cands:
            if idx in indices[:-1]:
                pass
            else:
                # print(question)
                # print(idx)
                # print(index_to_word[idx])
                answer = idx
                break
        if answer == label_idx:
            correct += 1
    return correct, count


def cosine_similarity(v1, v2):
    m1 = np.linalg.norm(v1)
    m2 = np.linalg.norm(v2)
    return np.dot(v1, v2) / (m1 * m2)


def get_word_embeddings(embedding, length):
    result = np.zeros((length, 300))
    word_to_ngram_indices = pickle.load(open(cfg.word_to_ngram_indices_path, 'rb'))
    for idx in range(length):
        result[idx, :] = np.sum(embedding[word_to_ngram_indices[idx], :], axis=0)
    return result


def main(emb_path):
    questions_words = cfg.eval_root_path
    data_sem, data_syn = preprocess_file(questions_words)
    sem_count, sem_total = test_words(data_sem, emb_path)
    print("Semantic test accuracy: %.5f%%" % (sem_count / sem_total * 100))

    syn_count, syn_total = test_words(data_syn, emb_path)
    print("Syntactic test accuracy: %.5f%%" % (syn_count / syn_total * 100))

    print("Overall test accuracy: %.5f%%" % ((sem_count + syn_count) / (len(data_sem) + len(data_syn)) * 100))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--emb_path', type=str)
    args = parser.parse_args()

    main(args.emb_path)
