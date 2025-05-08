import math

num_model = 3
series_per_model = 50
ts_len = 300
gmm_components = 2
hidden_lstm_space = 10
train_test_split = 0.2


def fid(x):
    return x


def fprop(x):
    return 0.3 * x


def fsin(x):
    return math.sin(x)
