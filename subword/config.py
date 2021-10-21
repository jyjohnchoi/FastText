import os


class Config:
    def __init__(self):
        self.train_data_path = \
            './data/data_split/'
        self.train_files = \
            [os.path.join(self.train_data_path, filename) for filename in os.listdir(self.train_data_path)]

        self.train_path = './data/enwik9'
        self.freq_path = './dicts/frequency.pkl'
        self.word_to_ngrams_path = './dicts/word_to_ngrams.pkl'
        self.ngram_to_index_path = './dicts/ngram_to_index.pkl'
        self.word_to_index_path = './dicts/word_to_index.pkl'
        self.index_to_word_path = './dicts/index_to_word.pkl'
        self.word_to_ngram_indices_path = './word_to_ngram_indices.pkl'
        self.unigram_table_path = './dicts/unigram_table.pkl'
        self.eval_root_path = "./test_data/questions-words.txt"  # data for evaluation

        self.hidden_size = 300
        self.THRESHOLD = 1e-4
        self.MIN_COUNT = 5


if __name__ == "__main__":
    cfg = Config()
    print(cfg.train_files)
