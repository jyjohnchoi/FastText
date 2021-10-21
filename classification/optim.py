import numpy as np
import math


class AdaGrad:
    def __init__(self, lr):
        self.lr = lr
        self.h = None

    def step(self, params, grads, input_indices):
        if self.h is None:
            self.h = dict()
            for key, val in params.items():
                self.h[key] = np.zeros_like(val)

        for key in params.keys():
            if key == 0:
                self.h[key][input_indices] += grads[key] * grads[key]
                params[key][input_indices] -= self.lr * grads[key] / (np.sqrt(self.h[key][input_indices]) + 1e-7)
            else:
                self.h[key] += grads[key] * grads[key]
                params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key]) + 1e-7)

    def set_lr(self, lr):
        self.lr = lr


class SGD:
    def __init__(self, lr):
        self.lr = lr

    def step(self, params, grads, input_indices):
        for key, val in params.items():
            if key == 0:
                params[key][input_indices] -= self.lr * grads[key]
            else:
                params[key] -= self.lr * grads[key]

    def set_lr(self, lr):
        self.lr = lr

class AdaGradHS:
    def __init__(self, lr):
        self.lr = lr
        self.h = None

    def step(self, params, grads, input_indices, nodes):
        if self.h is None:
            self.h = dict()
            for key, val in params.items():
                self.h[key] = np.zeros_like(val)

        for key in params.keys():
            if key == 0:
                self.h[key][input_indices] += grads[key] * grads[key]
                params[key][input_indices] -= self.lr * grads[key] / (np.sqrt(self.h[key][input_indices]) + 1e-7)

            elif key == 3 or key == 4:
                self.h[key][nodes] += grads[key] * grads[key]
                params[key][nodes] -= self.lr * grads[key] / (np.sqrt(self.h[key][nodes]) + 1e-7)

            else:
                self.h[key] += grads[key] * grads[key]
                params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key]) + 1e-7)

    def set_lr(self, lr):
        self.lr = lr
