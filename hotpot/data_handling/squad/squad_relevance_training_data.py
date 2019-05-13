import itertools
from typing import List, Optional, Dict
from collections import Counter

import numpy as np

from hotpot.data_handling.dataset import ListBatcher, Dataset, QuestionAndParagraphsSpec, QuestionAndParagraphsDataset, \
    Preprocessor, SampleFilter, TrainingDataHandler
from hotpot.data_handling.relevance_training_data import BinaryQuestionAndParagraphs
from hotpot.data_handling.squad.squad_data import SquadQuestionWithDistractors, SquadRelevanceCorpus, SquadQuestion, \
    SquadDocument


class SquadStratifiedBinaryRelevanceDataset(QuestionAndParagraphsDataset):
    """ A class for handling a binary classification dataset for squad:
    - each sample is a question and a paragraph
    - each sample is labeled with 0 or 1 - is the paragraph the gold one not
    - in each epoch, each question will appear 4 times: once with the gold paragraph, and 3 times with distractors.
        The distractors will be chosen at random each epoch
    """

    def __init__(self, questions: List[SquadQuestionWithDistractors], batcher: ListBatcher,
                 fixed_dataset=False, sample_seed=18):
        self.questions = questions
        self.batcher = batcher
        self.fixed_dataset = fixed_dataset

        self.random = np.random.RandomState(seed=sample_seed)

        self.gold_samples = self._build_gold_samples()
        self.epoch_samples = None

    def _build_gold_samples(self):
        gold_samples = []
        for question in self.questions:
            pars = [question.paragraph.par_text]
            gold_samples.append(BinaryQuestionAndParagraphs(question.question, pars, 1, num_distractors=0,
                                                            question_id=question.question_id))
        return gold_samples

    def get_batches(self, n_batches):
        if len(self) < n_batches:
            raise ValueError()
        return itertools.islice(self.get_epoch(new_epoch=False), n_batches)

    def get_samples(self, n_samples: int):
        n_batches = self.batcher.epoch_size(n_samples)
        self.get_epoch()
        return self.batcher.get_epoch(self.random.choice(self.epoch_samples, n_samples, replace=False)), n_batches

    def get_epoch(self, new_epoch=True):
        if self.fixed_dataset:
            new_epoch = False
        if not new_epoch and self.epoch_samples is not None:
            return self.batcher.get_epoch(self.epoch_samples)
        false_samples = []
        for question in self.questions:
            distractors = self.random.choice(question.distractors, size=3, replace=False)
            false_samples.append(
                BinaryQuestionAndParagraphs(question.question, [distractors[0].par_text], 0, num_distractors=1,
                                            question_id=question.question_id))
            false_samples.append(
                BinaryQuestionAndParagraphs(question.question, [distractors[1].par_text], 0, num_distractors=1,
                                            question_id=question.question_id))
            false_samples.append(
                BinaryQuestionAndParagraphs(question.question, [distractors[2].par_text], 0, num_distractors=1,
                                            question_id=question.question_id))
        for gold in self.gold_samples:
            self.random.shuffle(gold.paragraphs)
        self.epoch_samples = self.gold_samples + false_samples
        np.random.shuffle(self.epoch_samples)
        return self.batcher.get_epoch(self.epoch_samples)

    def get_spec(self):
        batch_size = self.batcher.get_fixed_batch_size()
        num_contexts = 1
        max_q_words = max(len(q.question) for q in self.questions)
        max_c_words = max(max(c.num_tokens for c in (q.distractors + [q.paragraph])) for q in self.questions)
        return QuestionAndParagraphsSpec(batch_size=batch_size, max_num_contexts=num_contexts,
                                         max_num_question_words=max_q_words, max_num_context_words=max_c_words)

    def get_vocab(self):
        voc = set()
        for q in self.questions:
            voc.update(q.question)
            for para in (q.distractors + [q.paragraph]):
                voc.update(para.par_text)
        return voc

    def get_word_counts(self):
        count = Counter()
        for q in self.questions:
            count.update(q.question)
            for para in (q.distractors + [q.paragraph]):
                count.update(para.par_text)
        return count

    def __len__(self):
        return self.batcher.epoch_size(len(self.gold_samples) * 4)


class SquadFullQuestionParagraphPairsDataset(QuestionAndParagraphsDataset):
    def __init__(self, questions: List[SquadQuestionWithDistractors], batcher: ListBatcher):
        self.questions = questions
        self.batcher = batcher

        self.samples = self._build_full_dataset()

    def _build_full_dataset(self):
        samples = []
        for question in self.questions:
            samples.append(BinaryQuestionAndParagraphs(question.question, [question.paragraph.par_text], 1,
                                                       num_distractors=0,
                                                       question_id=question.question_id))
            for distractor in question.distractors:
                samples.append(BinaryQuestionAndParagraphs(question.question, [distractor.par_text], 0,
                                                           num_distractors=1,
                                                           question_id=question.question_id))
        return samples

    def get_batches(self, n_batches):
        if len(self) < n_batches:
            raise ValueError()
        return itertools.islice(self.get_epoch(), n_batches)

    def get_samples(self, n_samples: int):
        n_batches = self.batcher.epoch_size(n_samples)
        return self.batcher.get_epoch(np.random.choice(self.samples, n_samples, replace=False)), n_batches

    def get_epoch(self):
        return self.batcher.get_epoch(self.samples)

    def get_spec(self):
        batch_size = self.batcher.get_fixed_batch_size()
        num_contexts = 1
        max_q_words = max(len(q.question) for q in self.questions)
        max_c_words = max(max(c.num_tokens for c in (q.distractors + [q.paragraph])) for q in self.questions)
        return QuestionAndParagraphsSpec(batch_size=batch_size, max_num_contexts=num_contexts,
                                         max_num_question_words=max_q_words, max_num_context_words=max_c_words)

    def get_vocab(self):
        voc = set()
        for q in self.questions:
            voc.update(q.question)
            for para in (q.distractors + [q.paragraph]):
                voc.update(para.par_text)
        return voc

    def get_word_counts(self):
        count = Counter()
        for q in self.questions:
            count.update(q.question)
            for para in (q.distractors + [q.paragraph]):
                count.update(para.par_text)
        return count

    def __len__(self):
        return self.batcher.epoch_size(len(self.samples))


class SquadFullDocumentDataset(QuestionAndParagraphsDataset):
    def __init__(self, questions: List[SquadQuestion], batcher: ListBatcher, title2doc: Dict[str, SquadDocument]):
        self.questions = questions
        self.batcher = batcher
        self.title2doc = title2doc

        self.samples = self._build_full_dataset()

    def _build_full_dataset(self):
        title2max = {key: max(x.paragraph.par_id for x in group) for key, group in
                     itertools.groupby(sorted(self.questions, key=lambda x: x.paragraph.doc_title),
                                       key=lambda x: x.paragraph.doc_title)}
        samples = []
        for question in self.questions:
            samples.append(BinaryQuestionAndParagraphs(question.question, [question.paragraph.par_text], 1,
                                                       num_distractors=0,
                                                       question_id=question.question_id))
            title = question.paragraph.doc_title
            distractors = [x for x in self.title2doc[title].paragraphs[:title2max[title] + 1]
                           if x.par_id != question.paragraph.par_id]
            for distractor in distractors:
                samples.append(BinaryQuestionAndParagraphs(question.question, [distractor.par_text], 0,
                                                           num_distractors=1,
                                                           question_id=question.question_id))
        return samples

    def get_batches(self, n_batches):
        if len(self) < n_batches:
            raise ValueError()
        return itertools.islice(self.get_epoch(), n_batches)

    def get_samples(self, n_samples: int):
        n_batches = self.batcher.epoch_size(n_samples)
        return self.batcher.get_epoch(np.random.choice(self.samples, n_samples, replace=False)), n_batches

    def get_epoch(self):
        return self.batcher.get_epoch(self.samples)

    def get_spec(self):
        batch_size = self.batcher.get_fixed_batch_size()
        num_contexts = 1
        max_q_words = max(len(q.question) for q in self.questions)
        max_c_words = max(max(c.num_tokens for c in d.paragraphs) for d in self.title2doc.values())
        return QuestionAndParagraphsSpec(batch_size=batch_size, max_num_contexts=num_contexts,
                                         max_num_question_words=max_q_words, max_num_context_words=max_c_words)

    def get_vocab(self):
        voc = set()
        for q in self.questions:
            voc.update(q.question)

        for doc in self.title2doc.values():
            for para in doc.paragraphs:
                voc.update(para.par_text)
        return voc

    def get_word_counts(self):
        count = Counter()
        for q in self.questions:
            count.update(q.question)
        for doc in self.title2doc.values():
            for para in doc.paragraphs:
                count.update(para.par_text)
        return count

    def __len__(self):
        return self.batcher.epoch_size(len(self.samples))


class SquadTextLengthPreprocessor(Preprocessor):
    def __init__(self, num_tokens_th):
        self.num_tokens_th = num_tokens_th

    def preprocess(self, question: SquadQuestionWithDistractors):
        for par in question.distractors + [question.paragraph]:
            par.par_text = par.par_text[:self.num_tokens_th]
        return question


class SquadBinaryRelevanceTrainingData(TrainingDataHandler):
    def __init__(self, corpus: SquadRelevanceCorpus, train_batcher: ListBatcher, dev_batcher: ListBatcher,
                 sample_filter: Optional[SampleFilter] = None, preprocessor: Optional[Preprocessor] = None,
                 sample_train=None, sample_dev=None, sample_seed=18):
        super().__init__(train_batcher, dev_batcher, sample_filter, preprocessor, sample_train, sample_dev, sample_seed)
        self.corpus = corpus

        self._train = None
        self._dev = None

    def get_train(self) -> Dataset:
        self._load_data()
        return SquadStratifiedBinaryRelevanceDataset(self._train, self.train_batcher, fixed_dataset=False)

    def get_eval(self) -> Dict[str, Dataset]:  # TODO: are we sure wo don't want to use fixed datasets for evaluation?
        self._load_data()
        eval_sets = dict(train=SquadStratifiedBinaryRelevanceDataset(self._train,
                                                                     self.dev_batcher,
                                                                     fixed_dataset=False),
                         dev=SquadStratifiedBinaryRelevanceDataset(self._dev, self.dev_batcher,
                                                                   fixed_dataset=False))
        return eval_sets

    def __getstate__(self):
        state = self.__dict__
        state["_train"] = None
        state["_dev"] = None
        return state

    def __setstate__(self, state):
        self.__dict__ = state
