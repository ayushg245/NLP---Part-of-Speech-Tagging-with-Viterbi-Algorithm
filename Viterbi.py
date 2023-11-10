"""Viterbi Algorithm for inferring the most likely sequence of states from an HMM.

Patrick Wang, 2021
"""
from typing import Sequence, Tuple, TypeVar
import numpy as np
import nltk
from sympy.abc import x

nltk.download("brown")
nltk.download("universal_tagset")

Q = TypeVar("Q")
V = TypeVar("V")


def viterbi(
    obs: Sequence[int],
    pi: np.ndarray[Tuple[V], np.dtype[np.float_]],
    A: np.ndarray[Tuple[Q, Q], np.dtype[np.float_]],
    B: np.ndarray[Tuple[Q, V], np.dtype[np.float_]],
) -> tuple[list[int], float]:
    """Infer most likely state sequence using the Viterbi algorithm.

    Args:
        obs: An iterable of ints representing observations.
        pi: A 1D numpy array of floats representing initial state probabilities.
        A: A 2D numpy array of floats representing state transition probabilities.
        B: A 2D numpy array of floats representing emission probabilities.

    Returns:
        A tuple of:
        * A 1D numpy array of ints representing the most likely state sequence.
        * A float representing the probability of the most likely state sequence.
    """
    N = len(obs)
    Q, V = B.shape  # num_states, num_observations

    # d_{ti} = max prob of being in state i at step t
    #   AKA viterbi
    # \psi_{ti} = most likely state preceeding state i at step t
    #   AKA backpointer

    # initialization
    log_d = [np.log(pi) + np.log(B[:, obs[0]])]
    log_psi = [np.zeros((Q,))]

    # recursion
    for z in obs[1:]:
        log_da = np.expand_dims(log_d[-1], axis=1) + np.log(A)
        log_d.append(np.max(log_da, axis=0) + np.log(B[:, z]))
        log_psi.append(np.argmax(log_da, axis=0))

    # termination
    log_ps = np.max(log_d[-1])
    qs = [-1] * N
    qs[-1] = int(np.argmax(log_d[-1]))
    for i in range(N - 2, -1, -1):
        qs[i] = log_psi[i + 1][qs[i + 1]]

    return qs, np.exp(log_ps)


corpus = nltk.corpus.brown.tagged_sents(tagset="universal")[:10000]
dict_types = {}

# pi: A 1D numpy array of floats representing initial state probabilities.
for i in range(len(corpus)):
    # for j in range(len(corpus[i])):
    if corpus[i][0][1] not in dict_types:
        dict_types[corpus[i][0][1]] = 1
    else:
        dict_types[corpus[i][0][1]] += 1
total_words = sum(dict_types.values())
data_list = list(dict_types.keys())
data_tags = list(dict_types.values())
length = len(data_tags)
pi = np.array(data_tags) / total_words


# A: A 2D numpy array of floats representing state transition probabilities.
type_words = {}
key1 = []
for i in range(len(corpus)):
    for j in range(len(corpus[i]) - 1):
        key1 = (corpus[i][j][1], corpus[i][j + 1][1])
        if key1 not in type_words:
            type_words[tuple(key1)] = 1
        else:
            type_words[tuple(key1)] += 1


keys_tag = list(type_words.keys())
A = np.zeros((length, length))
for i in data_list:
    for j in data_list:
        key = (i, j)
        idx1 = data_list.index(i)
        idx2 = data_list.index(j)
        if (i, j) in type_words:
            A[idx1, idx2] = type_words[key]
A = A + 1
A = A / A.sum(axis=1)[:, None]

# B: A 2D numpy array of floats representing emission probabilities.
words_tags = {}
vocab = []
for i in range(len(corpus)):
    for j in range(len(corpus[i])):
        key2 = corpus[i][j]
        if key2 not in words_tags:
            words_tags[tuple(key2)] = 1
        else:
            words_tags[tuple(key2)] += 1
        if key2[0] not in vocab:
            vocab.append(key2[0])


vocab.append("OOV")

len1 = len(vocab)
B = np.zeros((length, len1))
for ia in words_tags.keys():
    row = data_list.index(ia[1])
    col = vocab.index(ia[0])
    B[row, col] = words_tags[ia]
B = B + 1
B = B / B.sum(axis=1)[:, None]

# obs: An iterable of ints representing observations.
test = nltk.corpus.brown.tagged_sents(tagset="universal")[10150:10153]
for i in test:
    obs = []
    tag = []
    for x in i:
        tag.append(x[1])
        try:
            pos = vocab.index(x[0])
        except:
            pos = vocab.index("OOV")
        obs.append(pos)
    (a, b) = viterbi(obs, pi, A, B)
    print(b)
    for i in range(len(a)):
        print(f"{tag[i]} {data_list[a[i]]}")
