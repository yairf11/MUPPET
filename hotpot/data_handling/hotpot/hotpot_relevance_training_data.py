import itertools
from collections import Counter
from typing import List, Optional, Dict, Tuple
import copy

import numpy as np

from hotpot.data_handling.dataset import ListBatcher, Dataset, QuestionAndParagraphsSpec, QuestionAndParagraphsDataset, \
    Preprocessor, SampleFilter, TrainingDataHandler
from hotpot.data_handling.hotpot.hotpot_data import HotpotQuestion, HotpotQuestions
from hotpot.data_handling.relevance_training_data import BinaryQuestionAndParagraphs, \
    IterativeQuestionAndParagraphs
from hotpot.utils import flatten_iterable


def get_segments_from_sentences(sentences: List[List[str]]):
    # return flatten_iterable([[idx] * len(sent) for idx, sent in enumerate(sentences)])
    segments = []
    i = 0
    for s in sentences:
        if len(s) == 0:
            continue
        segments.extend([i] * len(s))
        i += 1
    return segments


class HotpotStratifiedBinaryQuestionParagraphPairsDataset(QuestionAndParagraphsDataset):
    """ A class for handling a binary classification dataset for hotpot:
    - each sample is a question and two paragraphs
    - each sample is labeled with 0 or 1 - is the pair the gold one not
    - in each epoch, each question will appear 4 times: with both gold paragraphs,
        with one gold and one false (for each gold), and one without gold at all.
        The distractors will be chosen at random each epoch
    """

    def __init__(self, questions: List[HotpotQuestion], batcher: ListBatcher, fixed_dataset=False, sample_seed=18,
                 add_gold_distractor=True):
        self.questions = questions
        self.batcher = batcher
        self.fixed_dataset = fixed_dataset
        self.add_gold_distractor = add_gold_distractor

        self.random = np.random.RandomState(seed=sample_seed)

        self.gold_samples = self._build_gold_samples()
        self.epoch_samples = None

    def _build_gold_samples(self):
        gold_samples = []
        for question in self.questions:
            pars = [flatten_iterable(question.supporting_facts[0].sentences),
                    flatten_iterable(question.supporting_facts[1].sentences)]
            self.random.shuffle(pars)
            gold_samples.append(BinaryQuestionAndParagraphs(question.question_tokens, pars, 1, num_distractors=0,
                                                            question_id=question.question_id, q_type=question.q_type))
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
            two_distractors = [flatten_iterable(x.sentences) for x in self.random.choice(question.distractors, size=2,
                                                                                         replace=False)]
            true_and_false_1 = [flatten_iterable(question.supporting_facts[0].sentences), two_distractors[0]]
            true_and_false_2 = [flatten_iterable(question.supporting_facts[1].sentences), two_distractors[1]]
            self.random.shuffle(true_and_false_1)
            self.random.shuffle(true_and_false_2)
            false_samples.append(
                BinaryQuestionAndParagraphs(question.question_tokens, true_and_false_1, 0, num_distractors=1,
                                            question_id=question.question_id, q_type=question.q_type))
            false_samples.append(
                BinaryQuestionAndParagraphs(question.question_tokens, true_and_false_2, 0, num_distractors=1,
                                            question_id=question.question_id, q_type=question.q_type))
            false_samples.append(BinaryQuestionAndParagraphs(question.question_tokens, [flatten_iterable(x.sentences)
                                                                                        for x in self.random.choice(
                    question.distractors, size=2,
                    replace=False)], 0, num_distractors=2, question_id=question.question_id, q_type=question.q_type))
            if self.add_gold_distractor:
                rand_q_idx = self.random.randint(len(self.gold_samples))
                while self.gold_samples[rand_q_idx].question_id == question.question_id:
                    rand_q_idx = self.random.randint(len(self.gold_samples))
                selected_q = self.gold_samples[rand_q_idx]
                self.random.shuffle(selected_q.paragraphs)
                false_samples.append(BinaryQuestionAndParagraphs(question.question_tokens,
                                                                 selected_q.paragraphs,
                                                                 label=0, num_distractors=2,
                                                                 question_id=question.question_id,
                                                                 q_type=question.q_type))
        for gold in self.gold_samples:
            self.random.shuffle(gold.paragraphs)
        self.epoch_samples = self.gold_samples + false_samples
        np.random.shuffle(self.epoch_samples)
        return self.batcher.get_epoch(self.epoch_samples)

    def get_spec(self):
        batch_size = self.batcher.get_fixed_batch_size()
        num_contexts = 2
        max_q_words = max(len(q.question_tokens) for q in self.questions)
        max_c_words = max(max(c.num_tokens for c in (q.distractors + q.supporting_facts)) for q in self.questions)
        return QuestionAndParagraphsSpec(batch_size=batch_size, max_num_contexts=num_contexts,
                                         max_num_question_words=max_q_words, max_num_context_words=max_c_words)

    def get_vocab(self):
        voc = set()
        for q in self.questions:
            voc.update(q.question_tokens)
            for para in (q.distractors + q.supporting_facts):
                voc.update(flatten_iterable(para.sentences))
        return voc

    def get_word_counts(self):
        count = Counter()
        for q in self.questions:
            count.update(q.question_tokens)
            for para in (q.distractors + q.supporting_facts):
                count.update(flatten_iterable(para.sentences))
        return count

    def __len__(self):
        len_mult = 5 if self.add_gold_distractor else 4
        return self.batcher.epoch_size(len(self.gold_samples) * len_mult)


class HotpotIterativeRetrievalDataset(QuestionAndParagraphsDataset):
    """ A class for handling an iterative retrieval dataset for hotpot.
    This dataset is far more complex than HotpotStratifiedBinaryQuestionParagraphPairsDataset, for the following reasons:
    - The order of the paragraphs matters:
        * The two paragraphs are supposed to act as paragraphs that have been retrieved in
          two iterations one after the other
        * In gold bridge examples, the first paragraph is the one with a higher tfidf score, as it is more likely to be
          the first supporting fact in the bridge
          We probably shouldn't use the opposite direction, as there might be glitches. So the lower ranked supporting
          fact should never be the first one, no matter the label
        * In comparison questions, both orders are fine
    - Each sample has two labels - one for each iteration. The second is 1 iff both paragraphs are gold.

    Sampling methods:
    - Comparison questions:
        * Gold: the two gold paragraphs, in both possible orders
        * first gold, second false: first is either one of the supporting facts, second is one of the distractors
          (or the same paragraphs twice)
    - Bridge questions:
        * Gold: the two gold paragraphs, with the one with higher score first
        * first gold, second false: the higher scored paragraphs with some distractor (or the same paragraphs twice)
    - General:
        * False 1: all distractors
        * False 2: first distractor, second one of supporting facts
        * False 3: a gold example from another question
    """

    def __init__(self, questions: List[HotpotQuestion], batcher: ListBatcher, fixed_dataset=False, sample_seed=18,
                 bridge_as_comparison=False, group_pairs_in_batches=False, label_by_span=False,
                 num_distractors_in_batch=1):
        self.questions = questions
        self.batcher = batcher
        self.fixed_dataset = fixed_dataset
        self.bridge_as_comparison = bridge_as_comparison
        self.group_pairs_in_batches = group_pairs_in_batches
        self.label_by_span = label_by_span
        self.num_distractors_in_batch = num_distractors_in_batch

        if fixed_dataset and group_pairs_in_batches:
            raise NotImplementedError()

        if label_by_span and bridge_as_comparison:
            raise ValueError()

        self.qid2question = {q.question_id: q for q in questions}

        self.random = np.random.RandomState(seed=sample_seed)

        self.gold_samples = self._build_gold_samples()
        self.epoch_samples = None

    def _get_no_span_containing_golds(self, qid):
        """ we assume that a passage not containing the answer should be the first in the hops.
        If both contain (or not), they are regarded equal """
        return [idx for idx, span in enumerate(self.qid2question[qid].gold_spans) if len(span) == 0]

    def _build_gold_samples(self):
        gold_samples = []
        for question in self.questions:
            if question.q_type == 'comparison' or self.bridge_as_comparison:
                pars_order = [0, 1]
                self.random.shuffle(pars_order)
            else:
                if not self.label_by_span:
                    pars_order = [0, 1] if question.gold_scores[0] > question.gold_scores[1] else [1, 0]
                else:
                    gold_idxs = self._get_no_span_containing_golds(question.question_id)
                    pars_order = [0, 1] if 0 in gold_idxs else [1, 0]
                    if len(gold_idxs) != 1:  # either both contain the answer or both don't contain, so regarded equal
                        self.random.shuffle(pars_order)
            pars = [flatten_iterable(question.supporting_facts[i].sentences) for i in pars_order]
            sentence_segs = [get_segments_from_sentences(question.supporting_facts[i].sentences) for i in pars_order]
            gold_samples.append(IterativeQuestionAndParagraphs(question.question_tokens, pars,
                                                               first_label=1, second_label=1,
                                                               question_id=question.question_id, q_type=question.q_type,
                                                               sentence_segments=sentence_segs))
        return gold_samples

    def get_batches(self, n_batches):
        if len(self) < n_batches:
            raise ValueError()
        return itertools.islice(self.get_epoch(new_epoch=False), n_batches)

    def get_samples(self, n_samples: int):
        n_batches = self.batcher.epoch_size(n_samples * (5 if not self.group_pairs_in_batches else 1))
        if not self.group_pairs_in_batches:
            return self._build_regular_batches(self.random.choice(self.gold_samples, n_samples, replace=False).tolist()), n_batches
        return self._build_pair_batches(self.random.choice(self.gold_samples, n_samples, replace=False).tolist()), n_batches

    def _sample_rand_par_other_q(self, qid):
        rand_q_idx = self.random.randint(len(self.questions))
        while self.questions[rand_q_idx].question_id == qid:
            rand_q_idx = self.random.randint(len(self.questions))
        return self.random.choice(self.questions[rand_q_idx].supporting_facts + self.questions[rand_q_idx].distractors)

    def _sample_first_gold_second_false(self, qid):
        question = self.qid2question[qid]
        rand_par_other_q = self._sample_rand_par_other_q(qid)
        if question.q_type == 'comparison' or self.bridge_as_comparison:
            first_gold_par = question.supporting_facts[self.random.randint(2)]
        else:
            if not self.label_by_span:
                first_gold_idx = 0 if question.gold_scores[0] > question.gold_scores[1] else 1
                first_gold_par = question.supporting_facts[first_gold_idx]
            else:
                gold_idxs = self._get_no_span_containing_golds(question.question_id)
                if len(gold_idxs) == 1:
                    first_gold_par = question.supporting_facts[gold_idxs[0]]
                else:
                    first_gold_par = question.supporting_facts[self.random.randint(2)]

        rand_par = self.random.choice([rand_par_other_q, first_gold_par, self.random.choice(question.distractors)],
                                      p=[0.05, 0.1, 0.85])

        pars = [flatten_iterable(first_gold_par.sentences), flatten_iterable(rand_par.sentences)]
        segs = [get_segments_from_sentences(first_gold_par.sentences),
                get_segments_from_sentences(rand_par.sentences)]
        return IterativeQuestionAndParagraphs(question=question.question_tokens, paragraphs=pars,
                                              first_label=1, second_label=0,
                                              question_id=question.question_id,
                                              q_type=question.q_type,
                                              sentence_segments=segs)

    def _sample_false_1(self, qid):
        """ False sample of type 1: all distractors.
        No sampling from other question here, as I think it's less effective in this case"""
        question = self.qid2question[qid]
        two_distractors = self.random.choice(question.distractors, size=2, replace=False)
        pars = [flatten_iterable(x.sentences) for x in two_distractors]
        segs = [get_segments_from_sentences(x.sentences) for x in two_distractors]
        return IterativeQuestionAndParagraphs(question=question.question_tokens, paragraphs=pars,
                                              first_label=0, second_label=0,
                                              question_id=question.question_id,
                                              q_type=question.q_type,
                                              sentence_segments=segs)

    def _sample_false_2(self, qid):
        """ False sample of type 2: first distractor, second one of supporting facts """
        question = self.qid2question[qid]
        rand_par_other_q = self._sample_rand_par_other_q(qid)
        distractor = self.random.choice([self.random.choice(question.distractors), rand_par_other_q], p=[0.9, 0.1])
        gold = self.random.choice(question.supporting_facts)
        pars = [flatten_iterable(x.sentences) for x in [distractor, gold]]
        segs = [get_segments_from_sentences(x.sentences) for x in [distractor, gold]]
        return IterativeQuestionAndParagraphs(question=question.question_tokens, paragraphs=pars,
                                              first_label=0, second_label=0,
                                              question_id=question.question_id,
                                              q_type=question.q_type,
                                              sentence_segments=segs)

    def _sample_false_3(self, qid):
        """ False sample of type 2: gold from other question """
        question = self.qid2question[qid]
        rand_q_idx = self.random.randint(len(self.gold_samples))
        while self.gold_samples[rand_q_idx].question_id == question.question_id:
            rand_q_idx = self.random.randint(len(self.gold_samples))
        selected_q = self.gold_samples[rand_q_idx]
        return IterativeQuestionAndParagraphs(question=question.question_tokens,
                                              paragraphs=[x for x in selected_q.paragraphs],
                                              first_label=0, second_label=0,
                                              question_id=question.question_id,
                                              q_type=question.q_type,
                                              sentence_segments=[x for x in
                                                                 selected_q.sentence_segments])

    def get_epoch(self, new_epoch=True):
        if self.group_pairs_in_batches:
            return self._build_pair_batches(self.gold_samples)
        if self.fixed_dataset:
            new_epoch = False
        if not new_epoch and self.epoch_samples is not None:
            return self.batcher.get_epoch(self.epoch_samples)
        return self._build_regular_batches(self.gold_samples, set_epoch_samples=True)
        # false_samples = []
        # for question in self.questions:
        #     # false_samples.append(self._sample_first_gold_second_false(question.question_id))
        #     # false_samples.append(self._sample_false_1(question.question_id))
        #     # false_samples.append(self._sample_false_2(question.question_id))
        #     # false_samples.append(self._sample_false_3(question.question_id))
        #     for _ in range(4):
        #         false_samples.append(self.random.choice([self._sample_first_gold_second_false,
        #                                                  self._sample_false_1, self._sample_false_2,
        #                                                  self._sample_false_3],
        #                                                 p=[0.35, 0.25, 0.35, 0.05])(question.question_id))
        # for gold in self.gold_samples:
        #     if gold.q_type == 'comparison' or self.bridge_as_comparison or \
        #             (self.label_by_span and len(self._get_no_span_containing_golds(gold.question_id)) != 1):
        #         # shuffling order when we can
        #         gold.paragraphs = [gold.paragraphs[1], gold.paragraphs[0]]
        #         gold.sentence_segments = [gold.sentence_segments[1], gold.sentence_segments[0]]
        # self.epoch_samples = self.gold_samples + false_samples
        # np.random.shuffle(self.epoch_samples)
        # return self.batcher.get_epoch(self.epoch_samples)

    def _build_regular_batches(self, gold_questions, set_epoch_samples=False):
        false_samples = []
        for question in gold_questions:
            for _ in range(4):
                false_samples.append(self.random.choice([self._sample_first_gold_second_false,
                                                         self._sample_false_1, self._sample_false_2,
                                                         self._sample_false_3],
                                                        p=[0.35, 0.25, 0.35, 0.05])(question.question_id))
        for gold in gold_questions:
            if gold.q_type == 'comparison' or self.bridge_as_comparison or \
                    (self.label_by_span and len(self._get_no_span_containing_golds(gold.question_id)) != 1):
                # shuffling order when we can
                gold.paragraphs = [gold.paragraphs[1], gold.paragraphs[0]]
                gold.sentence_segments = [gold.sentence_segments[1], gold.sentence_segments[0]]

        epoch_samples = gold_questions + false_samples
        np.random.shuffle(epoch_samples)
        if set_epoch_samples:
            self.epoch_samples = epoch_samples
        return self.batcher.get_epoch(epoch_samples)

    def _build_pair_batches(self, gold_questions):
        np.random.shuffle(gold_questions)
        for q in gold_questions:
            if q.q_type == 'comparison' or self.bridge_as_comparison or \
                    (self.label_by_span and len(self._get_no_span_containing_golds(q.question_id)) != 1):
                # shuffling order when we can
                q.paragraphs = [q.paragraphs[1], q.paragraphs[0]]
                q.sentence_segments = [q.sentence_segments[1], q.sentence_segments[0]]

        for batch_golds in self.batcher.get_epoch(gold_questions):
            batch = []
            for gold in batch_golds:
                batch.append(gold)
                for _ in range(self.num_distractors_in_batch):
                    batch.append(self.random.choice([self._sample_first_gold_second_false,
                                                     self._sample_false_1, self._sample_false_2, self._sample_false_3],
                                                    p=[0.35, 0.25, 0.35, 0.05])(gold.question_id))
            yield batch

    def get_spec(self):
        batch_size = self.batcher.get_fixed_batch_size()
        num_contexts = 2
        max_q_words = max(len(q.question_tokens) for q in self.questions)
        max_c_words = max(max(c.num_tokens for c in (q.distractors + q.supporting_facts)) for q in self.questions)
        return QuestionAndParagraphsSpec(batch_size=batch_size, max_num_contexts=num_contexts,
                                         max_num_question_words=max_q_words, max_num_context_words=max_c_words)

    def get_vocab(self):
        voc = set()
        for q in self.questions:
            voc.update(q.question_tokens)
            for para in (q.distractors + q.supporting_facts):
                voc.update(flatten_iterable(para.sentences))
        return voc

    def get_word_counts(self):
        count = Counter()
        for q in self.questions:
            count.update(q.question_tokens)
            for para in (q.distractors + q.supporting_facts):
                count.update(flatten_iterable(para.sentences))
        return count

    def __len__(self):
        return self.batcher.epoch_size(len(self.gold_samples) * (5 if not self.group_pairs_in_batches else 1))


class HotpotFullIterativeDataset(QuestionAndParagraphsDataset):
    def __init__(self, questions: List[HotpotQuestion], batcher: ListBatcher, bridge_as_comparison=False):
        self.questions = questions
        self.batcher = batcher
        self.bridge_as_comparison = bridge_as_comparison

        self.samples = self._build_full_dataset()

    def _get_labels(self, is_gold1: bool, is_gold2: bool, q_type: str, are_same: bool,
                    is_first_higher: bool) -> Tuple[int, int]:
        if not is_gold1:
            return 0, 0
        if q_type == 'comparison' or self.bridge_as_comparison:
            return int(is_gold1), int(is_gold1 and is_gold2 and not are_same)
        else:
            return int(is_gold1 and is_first_higher), int(is_gold1 and is_first_higher and is_gold2 and not are_same)

    def _build_full_dataset(self):
        samples = []
        for question in self.questions:
            pars_and_scores = list(zip(question.supporting_facts + question.distractors,
                                       question.gold_scores + question.distractor_scores))
            higher_gold = question.supporting_facts[0] \
                if question.gold_scores[0] >= question.gold_scores[1] else question.supporting_facts[1]
            for p1, score1 in pars_and_scores:
                for p2, score2 in pars_and_scores:
                    first_label, second_label = self._get_labels(is_gold1=p1 in question.supporting_facts,
                                                                 is_gold2=p2 in question.supporting_facts,
                                                                 q_type=question.q_type,
                                                                 are_same=p1 == p2,
                                                                 is_first_higher=higher_gold == p1)
                    samples.append(IterativeQuestionAndParagraphs(question=question.question_tokens,
                                                                  paragraphs=[flatten_iterable(p1.sentences),
                                                                              flatten_iterable(p2.sentences)],
                                                                  first_label=first_label, second_label=second_label,
                                                                  question_id=question.question_id,
                                                                  q_type=question.q_type,
                                                                  sentence_segments=[get_segments_from_sentences(s)
                                                                                     for s in
                                                                                     [p1.sentences, p2.sentences]]))
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
        num_contexts = 2
        max_q_words = max(len(q.question_tokens) for q in self.questions)
        max_c_words = max(max(c.num_tokens for c in (q.distractors + q.supporting_facts)) for q in self.questions)
        return QuestionAndParagraphsSpec(batch_size=batch_size, max_num_contexts=num_contexts,
                                         max_num_question_words=max_q_words, max_num_context_words=max_c_words)

    def get_vocab(self):
        voc = set()
        for q in self.questions:
            voc.update(q.question_tokens)
            for para in (q.distractors + q.supporting_facts):
                voc.update(flatten_iterable(para.sentences))
        return voc

    def get_word_counts(self):
        count = Counter()
        for q in self.questions:
            count.update(q.question_tokens)
            for para in (q.distractors + q.supporting_facts):
                count.update(flatten_iterable(para.sentences))
        return count

    def __len__(self):
        return self.batcher.epoch_size(len(self.samples))


class HotpotFullQuestionParagraphPairsDataset(QuestionAndParagraphsDataset):
    def __init__(self, questions: List[HotpotQuestion], batcher: ListBatcher):
        self.questions = questions
        self.batcher = batcher

        self.samples = self._build_full_dataset()

    def _build_full_dataset(self):
        samples = []
        for question in self.questions:
            for i, p1 in enumerate(question.distractors + question.supporting_facts):
                for p2 in (question.distractors + question.supporting_facts)[i + 1:]:
                    label = 1 if ((p1 in question.supporting_facts) and (p2 in question.supporting_facts)) else 0
                    num_distractors = sum([p1 in question.distractors, p2 in question.distractors])
                    samples.append(BinaryQuestionAndParagraphs(question.question_tokens, [flatten_iterable(
                        p1.sentences),
                        flatten_iterable(
                            p2.sentences)], label, num_distractors=num_distractors, question_id=question.question_id,
                                                               q_type=question.q_type))
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
        num_contexts = 2
        max_q_words = max(len(q.question_tokens) for q in self.questions)
        max_c_words = max(max(c.num_tokens for c in (q.distractors + q.supporting_facts)) for q in self.questions)
        return QuestionAndParagraphsSpec(batch_size=batch_size, max_num_contexts=num_contexts,
                                         max_num_question_words=max_q_words, max_num_context_words=max_c_words)

    def get_vocab(self):
        voc = set()
        for q in self.questions:
            voc.update(q.question_tokens)
            for para in (q.distractors + q.supporting_facts):
                voc.update(flatten_iterable(para.sentences))
        return voc

    def get_word_counts(self):
        count = Counter()
        for q in self.questions:
            count.update(q.question_tokens)
            for para in (q.distractors + q.supporting_facts):
                count.update(flatten_iterable(para.sentences))
        return count

    def __len__(self):
        return self.batcher.epoch_size(len(self.samples))


class HotpotTextLengthPreprocessor(Preprocessor):
    def __init__(self, num_tokens_th):
        self.num_tokens_th = num_tokens_th

    def preprocess(self, question: HotpotQuestion):
        for par in question.distractors:
            while len(flatten_iterable(par.sentences)) > self.num_tokens_th:
                par.sentences = par.sentences[:-1]
        for par in question.supporting_facts:
            while len(flatten_iterable(par.sentences)) > self.num_tokens_th:
                if (len(par.sentences) - 1) in par.supporting_sentence_ids:
                    print("Warning: supporting fact above threshold. removing sample")
                    return None
                par.sentences = par.sentences[:-1]
        return question


class HotpotQuestionFilter(SampleFilter):
    def __init__(self, num_distractors_th):
        self.num_distractors_th = num_distractors_th

    def keep(self, question: HotpotQuestion) -> bool:
        return len(question.distractors) >= self.num_distractors_th


class HotpotBinaryRelevanceTrainingData(TrainingDataHandler):
    def __init__(self, corpus: HotpotQuestions, train_batcher: ListBatcher, dev_batcher: ListBatcher,
                 sample_filter: Optional[SampleFilter] = None, preprocessor: Optional[Preprocessor] = None,
                 sample_train=None, sample_dev=None, sample_seed=18, add_gold_distractors=True):
        super().__init__(train_batcher, dev_batcher, sample_filter, preprocessor, sample_train, sample_dev, sample_seed)
        self.corpus = corpus
        self.add_gold_distractors = add_gold_distractors
        self._train = None
        self._dev = None

    def get_train(self) -> Dataset:
        self._load_data()
        return HotpotStratifiedBinaryQuestionParagraphPairsDataset(self._train, self.train_batcher, fixed_dataset=False,
                                                                   add_gold_distractor=self.add_gold_distractors)

    def get_eval(self) -> Dict[str, Dataset]:  # TODO: are we sure wo don't want to use fixed datasets for evaluation?
        self._load_data()
        eval_sets = dict(
            train=HotpotStratifiedBinaryQuestionParagraphPairsDataset(self._train,
                                                                      self.dev_batcher,
                                                                      fixed_dataset=False,
                                                                      add_gold_distractor=self.add_gold_distractors),
            dev=HotpotStratifiedBinaryQuestionParagraphPairsDataset(self._dev, self.dev_batcher,
                                                                    fixed_dataset=False,
                                                                    add_gold_distractor=self.add_gold_distractors))
        return eval_sets

    def __getstate__(self):
        state = self.__dict__
        state["_train"] = None
        state["_dev"] = None
        return state

    def __setstate__(self, state):
        self.__dict__ = state


class HotpotIterativeRelevanceTrainingData(TrainingDataHandler):
    def __init__(self, corpus: HotpotQuestions, train_batcher: ListBatcher, dev_batcher: ListBatcher,
                 sample_filter: Optional[SampleFilter] = None, preprocessor: Optional[Preprocessor] = None,
                 sample_train=None, sample_dev=None, sample_seed=18, bridge_as_comparison=False,
                 group_pairs_in_batches=False, label_by_span=False, num_distractors_in_batch=1,
                 max_batch_size=None):
        super().__init__(train_batcher, dev_batcher, sample_filter, preprocessor, sample_train, sample_dev, sample_seed)
        self.corpus = corpus
        self.bridge_as_comparison = bridge_as_comparison
        self.group_pairs_in_batches = group_pairs_in_batches
        self.label_by_span = label_by_span
        self.num_distractors_in_batch = num_distractors_in_batch
        self.max_batch_size = max_batch_size
        self._train = None
        self._dev = None
        if self.label_by_span:
            print("Labeling first golds by spans")
        if self.bridge_as_comparison:
            print("Considering comparison and bridge alike")
        if self.group_pairs_in_batches:
            print("Grouping positives and negatives, for ranking loss")

    def get_train(self) -> Dataset:
        self._load_data()
        return HotpotIterativeRetrievalDataset(self._train, self.train_batcher, fixed_dataset=False,
                                               bridge_as_comparison=self.bridge_as_comparison,
                                               group_pairs_in_batches=self.group_pairs_in_batches,
                                               label_by_span=self.label_by_span,
                                               num_distractors_in_batch=self.num_distractors_in_batch)

    def get_eval(self) -> Dict[str, Dataset]:  # TODO: are we sure wo don't want to use fixed datasets for evaluation?
        self._load_data()
        bigger_batcher = self.dev_batcher
        if self.group_pairs_in_batches:
            bigger_batcher = copy.deepcopy(self.dev_batcher)
            bigger_batcher.batch_size *= (self.num_distractors_in_batch+1)
            if self.max_batch_size is not None:
                bigger_batcher.batch_size = min(bigger_batcher.batch_size, int(self.max_batch_size/2))
        eval_sets = dict(
            train=HotpotIterativeRetrievalDataset(self._train,
                                                  bigger_batcher,
                                                  fixed_dataset=False,
                                                  bridge_as_comparison=self.bridge_as_comparison,
                                                  label_by_span=self.label_by_span),
            dev=HotpotIterativeRetrievalDataset(self._dev,
                                                bigger_batcher,
                                                fixed_dataset=False, bridge_as_comparison=self.bridge_as_comparison,
                                                label_by_span=self.label_by_span))
        if self.group_pairs_in_batches:
            eval_sets.update(dict(
                train_grouped=HotpotIterativeRetrievalDataset(self._train,
                                                              self.dev_batcher,
                                                              fixed_dataset=False,
                                                              bridge_as_comparison=self.bridge_as_comparison,
                                                              group_pairs_in_batches=True,
                                                              label_by_span=self.label_by_span,
                                                              num_distractors_in_batch=self.num_distractors_in_batch),
                dev_grouped=HotpotIterativeRetrievalDataset(self._dev, self.dev_batcher,
                                                            fixed_dataset=False,
                                                            bridge_as_comparison=self.bridge_as_comparison,
                                                            group_pairs_in_batches=True,
                                                            label_by_span=self.label_by_span,
                                                            num_distractors_in_batch=self.num_distractors_in_batch)))
        return eval_sets

    def __getstate__(self):
        state = self.__dict__
        state["_train"] = None
        state["_dev"] = None
        return state

    def __setstate__(self, state):
        if "bridge_as_comparison" not in state:
            state["bridge_as_comparison"] = False
        if "group_pairs_in_batches" not in state:
            state["group_pairs_in_batches"] = False
        if "label_by_span" not in state:
            state["label_by_span"] = False
        if "num_distractors_in_batch" not in state:
            state["num_distractors_in_batch"] = 1
        if "max_batch_size" not in state:
            state["max_batch_size"] = None
        self.__dict__ = state
