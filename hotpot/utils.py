import json
from typing import TypeVar, Iterable, List

from os.path import join

from hotpot.data_handling.word_vectors import load_word_vectors


class ResourceLoader(object):
    """
    Abstraction for models the need access to external resources to setup, currently just
    for word-vectors.
    """

    def __init__(self, load_vec_fn=load_word_vectors):
        self.load_vec_fn = load_vec_fn

    def load_word_vec(self, vec_name, voc=None):
        return self.load_vec_fn(vec_name, voc)


class LoadFromPath(object):
    def __init__(self, path):
        self.path = path

    def load_word_vec(self, vec_name, voc=None):
        return load_word_vectors(join(self.path, vec_name), voc, True)


class CachingResourceLoader(ResourceLoader):

    def __init__(self, load_vec_fn=load_word_vectors):
        super().__init__(load_vec_fn)
        self.word_vec = {}

    def load_word_vec(self, vec_name, voc=None):
        if vec_name not in self.word_vec:
            self.word_vec[vec_name] = super().load_word_vec(vec_name)
        return self.word_vec[vec_name]


def load_json_dataset(json_path):
    with open(json_path, 'r') as f:
        return json.load(f)


T = TypeVar('T')


def flatten_iterable(listoflists: Iterable[Iterable[T]]) -> List[T]:
    return [item for sublist in listoflists for item in sublist]


def max_or_none(a, b):
    if a is None or b is None:
        return None
    return max(a, b)


def print_table(table: List[List[str]]):
    """ Print the lists with evenly spaced columns """

    # print while padding each column to the max column length
    col_lens = [0] * len(table[0])
    for row in table:
        for i,cell in enumerate(row):
            col_lens[i] = max(len(cell), col_lens[i])

    formats = ["{0:<%d}" % x for x in col_lens]
    for row in table:
        print(" ".join(formats[i].format(row[i]) for i in range(len(row))))


def transpose_lists(lsts: List[List[T]]) -> List[List[T]]:
    return [list(i) for i in zip(*lsts)]
