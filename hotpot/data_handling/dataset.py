import itertools
from typing import Optional, Dict, Iterator, List, Callable

import numpy as np

from hotpot.configurable import Configurable
from hotpot.data_handling.data import RelevanceQuestion
from hotpot.utils import ResourceLoader, max_or_none, flatten_iterable


class Dataset(object):
    """ Data iterator we can use to train or test a model on, responsible for both storing data
    and deciding how to batch it. """

    def get_epoch(self):
        """ Returns an iterator of batches/elements to train on, these elements are what will get
        passed to model.encode. Usually (but not necessarily) a list/batch of training examples """
        raise NotImplementedError(self.__class__)

    def get_batches(self, n_batches):
        if len(self) < n_batches:
            raise ValueError()
        return itertools.islice(self.get_epoch(), n_batches)

    def get_epochs(self, n_epochs: int):
        for _ in range(n_epochs):
            for batch in self.get_epoch():
                yield batch

    def get_samples(self, n_samples: int):
        """
        Sample for the data, be default we sample batches but subclasses can
        override this method to provide other kinds of sampling (like sampling individual elements).
        Must return both an iterator and the exact size of the iterator.
        """
        return self.get_batches(n_samples), n_samples

    def percent_filtered(self):
        # TODO nicer to just have "unfiltered_size"?
        """ If any filtering was done, the percent of examples that were filtered. Exposed so evaluators
         can compute percentages fairly even if some examples were removed during pre-processing """
        return None

    def get_vocab(self):
        """
        return a set of all tokens in the dataset. This is useful in many places, but still a hack because
        it is not a must-have attribute.
        """
        raise NotImplementedError(self.__class__)

    def get_word_counts(self):
        """
        return a Counter of all tokens in the dataset. This is useful in many places, but still a hack because
        it is not a must-have attribute.
        """
        raise NotImplementedError(self.__class__)

    def __len__(self):
        """ Number of batches per an epoch """
        raise NotImplementedError(self.__class__)


class TrainingData(Configurable):

    def get_train(self) -> Dataset:
        raise NotImplementedError()

    def get_eval(self) -> Dict[str, Dataset]:
        raise NotImplementedError()

    def get_train_corpus(self) -> object:
        """
        Return an object derived from the training data that will be passed to the model's initialization phase,
        what exactly is returned can be arbitrary, but will need to be compatible with
        the model's requirements. Example, return word counts to be used to decide what word vecs to train.
        """
        raise NotImplementedError()

    def get_resource_loader(self) -> ResourceLoader:
        return ResourceLoader()


def shuffle_list_buckets(data, key, rng):
    start = 0
    end = 0
    while start < len(data):
        while end < len(data) and key(data[start]) == key(data[end]):
            end += 1
        rng.shuffle(data[start:end])
        start = end
    return data


class ListBatcher(Configurable):
    def get_epoch(self, data: List):
        raise NotImplementedError()

    def get_fixed_batch_size(self):
        """ Return the batch size if it is constant, else None """
        raise NotImplementedError()

    def get_max_batch_size(self):
        """ Return upper bound on the batch size """
        raise NotImplementedError()

    def epoch_size(self, n_elements):
        raise NotImplementedError()


class FixedOrderBatcher(ListBatcher):
    def __init__(self, batch_size: int, truncate_batches=False):
        self.batch_size = batch_size
        self.truncate_batches = truncate_batches

    def get_fixed_batch_size(self):
        return None if self.truncate_batches else self.batch_size

    def get_max_batch_size(self):
        return self.batch_size

    def get_epoch(self, data: List):
        n_batches = len(data) // self.batch_size
        for i in range(n_batches):
            yield data[i*self.batch_size:(i + 1)*self.batch_size]
        if self.truncate_batches and (len(data) % self.batch_size) > 0:
            yield data[self.batch_size * (len(data) // self.batch_size):]

    def epoch_size(self, n_elements):
        size = n_elements // self.batch_size
        if self.truncate_batches and (n_elements % self.batch_size) > 0:
            size += 1
        return size


class ShuffledBatcher(ListBatcher):
    def __init__(self,
                 batch_size: int,
                 truncate_batches=False):
        self.batch_size = batch_size
        self.truncate_batches = truncate_batches

    def get_fixed_batch_size(self):
        return None if self.truncate_batches else self.batch_size

    def get_max_batch_size(self):
        return self.batch_size

    def get_epoch(self, data: List):
        data = list(data)
        np.random.shuffle(data)
        n_batches = len(data) // self.batch_size
        for i in range(n_batches):
            yield data[i*self.batch_size:(i + 1)*self.batch_size]
        if self.truncate_batches and (len(data) % self.batch_size) > 0:
            yield data[self.batch_size * (len(data) // self.batch_size):]

    def epoch_size(self, n_elements):
        size = n_elements // self.batch_size
        if self.truncate_batches and (n_elements % self.batch_size) > 0:
            size += 1
        return size


class ClusteredBatcher(ListBatcher):
    def __init__(self,
                 batch_size: int,
                 clustering: Callable,
                 shuffle_buckets=False,
                 truncate_batches=False):
        self.batch_size = batch_size
        self.clustering = clustering
        self.shuffle_buckets = shuffle_buckets
        self.truncate_batches = truncate_batches

    def get_fixed_batch_size(self):
        return None if self.truncate_batches else self.batch_size

    def get_max_batch_size(self):
        return self.batch_size

    def get_epoch(self, data: List):
        data = sorted(data, key=self.clustering)
        if self.shuffle_buckets:
            shuffle_list_buckets(data, self.clustering, np.random)
        n_batches = len(data) // self.batch_size
        intervals = [(i*self.batch_size, (i + 1)*self.batch_size) for i in range(0, n_batches)]
        remainder = len(data) % self.batch_size
        if self.truncate_batches and remainder > 0:
            intervals.append((len(data) - remainder, len(data)))
        np.random.shuffle(intervals)
        for i, j in intervals:
            yield data[i:j]

    def epoch_size(self, n_elements):
        size = n_elements // self.batch_size
        if self.truncate_batches and (n_elements % self.batch_size) > 0:
            size += 1
        return size


class ListDataset(Dataset):
    """ Dataset with a fixed list of elements """

    def __init__(self, data: List, batching: ListBatcher, unfiltered_len: Optional[int]=None):
        self.data = data
        self.batching = batching
        self.unfiltered_len = unfiltered_len

    def get_samples(self, n_examples) -> Iterator:
        n_batches = n_examples // self.batching.get_max_batch_size()
        return self.get_batches(n_batches), n_batches

    @property
    def batch_size(self):
        return self.batching.get_fixed_batch_size()

    def get_epoch(self):
        return self.batching.get_epoch(self.data)

    def percent_filtered(self):
        if self.unfiltered_len is None:
            return None
        return (self.unfiltered_len - len(self.data)) / self.unfiltered_len

    def get_n_examples(self):
        return len(self.data)

    def __len__(self):
        return self.batching.epoch_size(len(self.data))


class QuestionAndParagraphs(object):
    """ A class for storing data points of a question with some paragraphs (can be any number)"""

    def __init__(self, question: List[str], paragraphs: List[List[str]],
                 sentence_segments: Optional[List[List[int]]]=None):
        """

        :param question: tokenized question
        :param paragraphs: tokenized paragraphs, all sentences are flattened to a single sequence of tokens
        :param sentence_segments: segments representing sentences, used for tf's segment_max etc.
        """
        self.question = question
        self.paragraphs = paragraphs
        self.sentence_segments = sentence_segments

    @property
    def num_contexts(self):
        return len(self.paragraphs)


class QuestionAndParagraphsSpec(object):
    """ Bound on the size of `QuestionAndParagraphs` objects """

    def __init__(self, batch_size, max_num_contexts=None, max_num_question_words=None,
                 max_num_context_words=None, max_word_size=None, max_batch_size=None):
        if batch_size is not None:
            if max_batch_size is None:
                max_batch_size = batch_size
            elif max_batch_size != batch_size:
                raise ValueError()
        self.batch_size = batch_size
        self.max_num_contexts = max_num_contexts
        self.max_num_quesiton_words = max_num_question_words
        self.max_num_context_words = max_num_context_words
        self.max_word_size = max_word_size
        self.max_batch_size = max_batch_size

    def __add__(self, o):
        return QuestionAndParagraphsSpec(
            max_or_none(self.batch_size, o.batch_size),
            max_or_none(self.max_num_contexts, o.max_num_contexts),
            max_or_none(self.max_num_quesiton_words, o.max_num_quesiton_words),
            max_or_none(self.max_num_context_words, o.max_num_context_words),
            max_or_none(self.max_word_size, o.max_word_size),
            max_or_none(self.max_batch_size, o.max_batch_size)
        )


def build_spec(batch_size: int,
               max_batch_size: int,
               num_contexts: int,
               data: List[RelevanceQuestion]) -> QuestionAndParagraphsSpec:
    max_ques_size = 0
    max_word_size = 0
    max_para_size = 0
    max_num_contexts = num_contexts
    for data_point in data:
        contexts = data_point.distractors + data_point.supporting_facts
        # max_num_contexts = num_contexts
        max_word_size = max(max_word_size, max(len(word) for word in flatten_iterable(contexts)))
        max_para_size = max(max_para_size, max(len(context) for context in contexts))
        max_ques_size = max(max_ques_size, len(data_point.question_tokens))
        max_word_size = max(max_word_size, max(len(word) for word in data_point.question_tokens))
    return QuestionAndParagraphsSpec(batch_size, max_num_contexts, max_ques_size, max_para_size,
                                     max_word_size, max_batch_size)


class QuestionAndParagraphsDataset(Dataset):
    """ Base class for datasets with multiple contexts for each question """

    def get_spec(self) -> QuestionAndParagraphsSpec:
        pass


def multiple_contexts_len(sample):
    return max(len(c) for c in sample.paragraphs)


class Preprocessor(Configurable):
    def preprocess(self, question):
        raise NotImplementedError()


class SampleFilter(Configurable):
    def keep(self, sample) -> bool:
        raise NotImplementedError()


class TrainingDataHandler(TrainingData):
    def __init__(self, train_batcher: ListBatcher, dev_batcher: ListBatcher,
                 sample_filter: Optional[SampleFilter] = None, preprocessor: Optional[Preprocessor] = None,
                 sample_train=None, sample_dev=None, sample_seed=18):
        self.train_batcher = train_batcher
        self.dev_batcher = dev_batcher
        self.sample_train = sample_train
        self.sample_dev = sample_dev
        self.filter = sample_filter
        self.preprocessor = preprocessor

        self.random = np.random.RandomState(seed=sample_seed)

        self.corpus = None  # This is some kind of an abstract parent class for datasets, so no corpus.
                            # however, we still treat it as if it were a real corpus in the following functions.
        self._train = None
        self._dev = None

    @property
    def name(self):
        return self.corpus.name

    def _load_data(self):
        if self._train is None:
            self._train = self.corpus.get_train()
            if self.filter is not None:
                self._train = [x for x in self._train if self.filter.keep(x)]
            if self.preprocessor is not None:
                self._train = [self.preprocessor.preprocess(x) for x in self._train
                               if self.preprocessor.preprocess(x) is not None]
            if self.sample_train is not None:
                self._train = self.random.choice(self._train, size=self.sample_train, replace=False)
        if self._dev is None:
            self._dev = self.corpus.get_dev()
            if self.filter is not None:
                self._dev = [x for x in self._dev if self.filter.keep(x)]
            if self.preprocessor is not None:
                self._dev = [self.preprocessor.preprocess(x) for x in self._dev
                             if self.preprocessor.preprocess(x) is not None]
            if self.sample_dev is not None:
                self._dev = self.random.choice(self._dev, size=self.sample_dev, replace=False)

    def get_train_corpus(self) -> object:
        return self.get_train().get_word_counts()

    def get_resource_loader(self) -> ResourceLoader:
        return self.corpus.get_resource_loader()