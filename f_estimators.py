import numpy as np


def split_sequence(sequence, n_steps, n_steps_y=1):
    X, y = list(), list()
    for i in range(len(sequence)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the sequence
        if end_ix + n_steps_y > len(sequence):
            break
        # gather input and output parts of the pattern
        seq_x = sequence[i:end_ix]
        seq_y = []
        if n_steps_y > 0:
            seq_y = sequence[end_ix:end_ix + n_steps_y]
        # print(seq_x,seq_y)
        X.append(seq_x)
        y.append(seq_y)

    return np.array(X), np.array(y)


def naive(process_realization):
    xi = []
    xi.append(process_realization[0])
    for i in range(1, len(process_realization)):
        # xi = x(t)-x(t-1)
        xi.append(process_realization[i] - process_realization[i - 1])
    xi = np.array(xi)
    xi = xi.reshape(len(xi))
    return xi
