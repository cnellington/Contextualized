import torch
import torch.nn.functional as F

zero_vector = lambda x, *args: torch.zeros((len(x), 1))
zero = lambda x: torch.zeros_like(x)
identity = lambda x: x
linear = lambda x, m, b: x*m+b
softmax = lambda x, m, b: F.softmax(x*m+b, dim=1)
logistic = lambda x, m, b: 1 / (1 + torch.exp(-x*m - b))


def make_fn(base_fn, **params):
    return lambda x: base_fn(x, **params)


def linear_fn(m=1, b=0):
    return make_fn(linear, m, b)


def softmax_fn(m=1, b=0):
    return make_fn(softmax, m, b)


def logistic_fn(m=1, b=0):
    return make_fn(logistic, m, b)
