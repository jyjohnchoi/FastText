import numpy as np


class SkipGramNS:
    def __init__(self, vocab_size, hidden_dim):
        self.W_in = np.random.uniform(low=-0.5/300, high=0.5/300, size=(vocab_size, hidden_dim))  # (V, 300)
        self.W_out = np.zeros((vocab_size, hidden_dim))  # (V, 300)
        self.cache = None

    def forward(self, input_idx):
        hidden = np.mean(self.W_in[input_idx], axis=0).reshape(1, -1)  # (1, 300)
        output = np.dot(self.W_out, hidden.T)
        return output

    def backward(self, ):
