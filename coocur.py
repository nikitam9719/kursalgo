#!/usr/bin/env python
# -*- coding: utf-8 -*-

from collections import Counter
import itertools
import numpy as np
from scipy import sparse

from util import listify



def build_vocab(corpus):
    vocab = Counter()
    for line in corpus:
        tokens = line.strip().split()
        vocab.update(tokens)
    return {word: (i, freq) for i, (word, freq) in enumerate(vocab.iteritems())}


@listify
def build_cooccur(vocab, corpus, window_size=10, min_count=None):
    f7 = open('coocur.txt', 'w')
    vocab_size = len(vocab)
    id2word = dict((i, word) for word, (i, _) in vocab.iteritems())
    cooccurrences = sparse.lil_matrix((vocab_size, vocab_size),
                                      dtype=np.float64)
    for i, line in enumerate(corpus):
        tokens = line.strip().split()
        token_ids = [vocab[word][0] for word in tokens]

        for center_i, center_id in enumerate(token_ids):
            context_ids = token_ids[max(0, center_i - window_size) : center_i]
            contexts_len = len(context_ids)

            for left_i, left_id in enumerate(context_ids):
                distance = contexts_len - left_i
                increment = 1.0 / float(distance)

                cooccurrences[center_id, left_id] += increment
                cooccurrences[left_id, center_id] += increment
    # Now yield our tuple sequence (dig into the LiL-matrix internals to
    # quickly iterate through all nonzero cells)
    for i, (row, data) in enumerate(itertools.izip(cooccurrences.rows,
                                                   cooccurrences.data)):
        if min_count is not None and vocab[id2word[i]][1] < min_count:
            continue
        for data_idx, j in enumerate(row):
            if min_count is not None and vocab[id2word[j]][1] < min_count:
                continue
            f7.write(id2word[i]+'\t'+id2word[j]+'\t'+str(data[data_idx])+'\n')
            yield i, j, data[data_idx]


