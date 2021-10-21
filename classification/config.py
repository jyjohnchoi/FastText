class Config(object):
    def __init__(self):
        self.hidden = 10
        self.ngrams = True
        self.word_to_index_path = './dicts/word_to_index'
        self.dataset_path = './dicts/dataset'
        self.word_to_index_path_bigram = './dicts_bigram/word_to_index'
        self.dataset_path_bigram = './dicts_bigram/dataset'
        self.path_list = ["./dataset/ag_news_csv/", "./dataset/sogou_news_csv/", "./dataset/dbpedia_csv/",
                          "./dataset/yelp_review_polarity_csv/", "./dataset/yelp_review_full_csv/",
                          "./dataset/yahoo_answers_csv/", "./dataset/amazon_review_full_csv/",
                          "./dataset/amazon_review_polarity_csv/"]