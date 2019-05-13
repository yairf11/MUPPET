#!/usr/bin/env python3
# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""A script to build the tf-idf document matrices for retrieval."""

import numpy as np
import scipy.sparse as sp
import argparse
import os
import math
import logging

from multiprocessing import Pool as ProcessPool
from multiprocessing.util import Finalize
from functools import partial
from collections import Counter

from hotpot.tfidf_retriever import utils
from hotpot.tokenizers.charoffset_tokenizer import CharOffsetTokenizer
from hotpot.tokenizers.corenlp_tokenizer import CoreNLPTokenizer
from hotpot.tfidf_retriever.doc_db import DocDB
from hotpot import config

logger = logging.getLogger()
logger.setLevel(logging.INFO)
fmt = logging.Formatter('%(asctime)s: [ %(message)s ]', '%m/%d/%Y %I:%M:%S %p')
console = logging.StreamHandler()
console.setFormatter(fmt)
logger.addHandler(console)


# ------------------------------------------------------------------------------
# Multiprocessing functions
# ------------------------------------------------------------------------------

DOC2IDX = None
PROCESS_TOK = None
PROCESS_DB = None


def init(db_path):
    global PROCESS_TOK, PROCESS_DB
    PROCESS_TOK = CoreNLPTokenizer()
    Finalize(PROCESS_TOK, PROCESS_TOK.shutdown, exitpriority=100)
    PROCESS_DB = DocDB(db_path)
    Finalize(PROCESS_DB, PROCESS_DB.close, exitpriority=100)


def fetch_sentences(doc_title):
    global PROCESS_DB
    return PROCESS_DB.get_doc_sentences(doc_title)


def tokenize(text):
    global PROCESS_TOK
    return PROCESS_TOK.tokenize(text)


# ------------------------------------------------------------------------------
# Build article --> word count sparse matrix.
# ------------------------------------------------------------------------------


def count(ngram, hash_size, doc_title):
    """Fetch the text of a document and compute hashed ngrams counts."""
    global DOC2IDX
    row, col, data = [], [], []
    # Tokenize
    tokens = tokenize(' '.join(fetch_sentences(doc_title)))

    # Get ngrams from tokens, with stopword/punctuation filtering.
    ngrams = tokens.ngrams(
        n=ngram, uncased=True, filter_fn=utils.filter_ngram
    )

    # Hash ngrams and count occurences
    counts = Counter([utils.hash(gram, hash_size) for gram in ngrams])

    # Return in sparse matrix data format.
    row.extend(counts.keys())
    col.extend([DOC2IDX[doc_title]] * len(counts))
    data.extend(counts.values())
    return row, col, data


def get_count_matrix():
    """Form a sparse word to document count matrix (inverted index).

    M[i, j] = # times word i appears in document j.
    """
    # Map doc_ids to indexes
    global DOC2IDX
    with DocDB(args.db_path) as doc_db:
        doc_titles = doc_db.get_doc_titles()
    DOC2IDX = {doc_title: i for i, doc_title in enumerate(doc_titles)}

    # Setup worker pool
    workers = ProcessPool(
        args.num_workers,
        initializer=init,
        initargs=[args.db_path]
    )

    # Compute the count matrix in steps (to keep in memory)
    logger.info('Mapping...')
    row, col, data = [], [], []
    step = max(int(len(doc_titles) / 10), 1)
    batches = [doc_titles[i:i + step] for i in range(0, len(doc_titles), step)]
    _count = partial(count, args.ngram, args.hash_size)
    for i, batch in enumerate(batches):
        logger.info('-' * 25 + 'Batch %d/%d' % (i + 1, len(batches)) + '-' * 25)
        for b_row, b_col, b_data in workers.imap_unordered(_count, batch):
            row.extend(b_row)
            col.extend(b_col)
            data.extend(b_data)
    workers.close()
    workers.join()

    logger.info('Creating sparse matrix...')
    count_matrix = sp.csr_matrix(
        (data, (row, col)), shape=(args.hash_size, len(doc_titles))
    )
    count_matrix.sum_duplicates()
    return count_matrix, (DOC2IDX, doc_titles)


# ------------------------------------------------------------------------------
# Transform count matrix to different forms.
# ------------------------------------------------------------------------------


def get_tfidf_matrix(cnts):
    """Convert the word count matrix into tfidf one.

    tfidf = log(tf + 1) * log((N - Nt + 0.5) / (Nt + 0.5))
    * tf = term frequency in document
    * N = number of documents
    * Nt = number of occurences of term in all documents
    """
    Ns = get_doc_freqs(cnts)
    idfs = np.log((cnts.shape[1] - Ns + 0.5) / (Ns + 0.5))
    idfs[idfs < 0] = 0
    idfs = sp.diags(idfs, 0)
    tfs = cnts.log1p()
    tfidfs = idfs.dot(tfs)
    return tfidfs


def get_doc_freqs(cnts):
    """Return word --> # of docs it appears in."""
    binary = (cnts > 0).astype(int)
    freqs = np.array(binary.sum(1)).squeeze()
    return freqs


# ------------------------------------------------------------------------------
# Main.
# ------------------------------------------------------------------------------


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('db_path', type=str, default=config.DOC_DB,
                        help='Path to sqlite db holding document texts')
    parser.add_argument('out_dir', type=str, default=None,
                        help='Directory for saving output files')
    parser.add_argument('--name', type=str, default='', help='name of tfidf file')
    parser.add_argument('--ngram', type=int, default=2,
                        help=('Use up to N-size n-grams '
                              '(e.g. 2 = unigrams + bigrams)'))
    parser.add_argument('--hash-size', type=int, default=int(math.pow(2, 24)),
                        help='Number of buckets to use for hashing ngrams')
    # parser.add_argument('--tokenizer', type=str, default='simple',
    #                     help=("String option specifying tokenizer type to use "
    #                           "(e.g. 'corenlp')"))
    parser.add_argument('--num-workers', type=int, default=None,
                        help='Number of CPU processes (for tokenizing, etc)')
    args = parser.parse_args()

    logging.info('Counting words...')
    count_matrix, doc_dict = get_count_matrix()

    logger.info('Making tfidf vectors...')
    tfidf = get_tfidf_matrix(count_matrix)

    logger.info('Getting word-doc frequencies...')
    freqs = get_doc_freqs(count_matrix)

    basename = os.path.splitext(os.path.basename(args.db_path))[0]
    basename += ('-tfidf-ngram=%d-hash=%d-tokenizer=%s_%s' %
                 (args.ngram, args.hash_size, 'charoffset', args.name))
    filename = os.path.join(args.out_dir, basename)

    logger.info('Saving to %s.npz' % filename)
    metadata = {
        'doc_freqs': freqs,
        'tokenizer': 'charoffset',
        'hash_size': args.hash_size,
        'ngram': args.ngram,
        'doc_dict': doc_dict
    }
    utils.save_sparse_csr(filename, tfidf, metadata)
