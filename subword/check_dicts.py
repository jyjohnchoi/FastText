from config import Config
import pickle


if __name__ == "__main__":
    cfg = Config()
    freq = cfg.freq_path
    w2n = cfg.word_to_ngrams_path
    n2i = cfg.ngram_to_index_path
    w2i = cfg.word_to_index_path
    i2w = cfg.index_to_word_path
    w2ni = cfg.word_to_ngram_indices_path


    f = pickle.load(open(n2i, 'rb'))
    g = pickle.load(open(w2i, 'rb'))
    print(len(f))
    print(len(g))
    # i = 0
    # for k in f.keys():
    #     i += 1
    #     if i > 10:
    #         break
    #     print(k, f[k])
    # print(f['he>'])
    i = 0
    for k, k2 in zip(f.keys(), g.keys()):
        i += 1
        # print(k, k2)

