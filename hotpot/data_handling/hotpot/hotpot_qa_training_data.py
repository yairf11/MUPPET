from typing import List, Tuple, Optional, Dict
import itertools
import numpy as np
from collections import Counter

from hotpot.data_handling.dataset import ListBatcher, QuestionAndParagraphsDataset, QuestionAndParagraphsSpec, \
    Preprocessor, SampleFilter, TrainingDataHandler, Dataset
from hotpot.data_handling.hotpot.hotpot_data import HotpotQuestion, HotpotParagraph, HotpotQuestions
from hotpot.data_handling.qa_training_data import SpanQuestionAndParagraphs
from hotpot.utils import flatten_iterable


def get_segments_from_sentences_fix_sup(sentences: List[List[str]], sup_idxs: np.ndarray):
    segments = []
    i = 0
    for s in sentences:
        if len(s) == 0:
            sup_idxs[sup_idxs > i] -= 1
            continue
        segments.extend([i] * len(s))
        i += 1
    return segments, sup_idxs


def merge_paragraphs(paragraphs: List[HotpotParagraph], spans: List[np.ndarray], supporting_fact_idxs: List[List[int]])\
        -> Tuple[List[str], np.ndarray, List[int], np.ndarray]:
    # todo this supporting fact fixing is a hack but easy to handle here so will do for now
    for i in range(len(paragraphs)):
        supporting_fact_idxs[i] = [x for x in supporting_fact_idxs[i] if x < len(paragraphs[i].sentences)]
    merged_text = []
    merged_spans = np.zeros((0, 2), dtype=np.int32)
    merged_sp_idxs = np.zeros(0, dtype=np.int32)
    merged_sentences = []
    for par, par_spans, par_facts in zip(paragraphs, spans, supporting_fact_idxs):
        merged_spans = np.concatenate([merged_spans, par_spans + len(merged_text)])
        merged_sp_idxs = np.concatenate([merged_sp_idxs, np.array(par_facts, dtype=np.int32) + len(merged_sentences)])
        merged_sentences.extend(par.sentences)
        merged_text.extend(flatten_iterable(par.sentences))
    segments, merged_sp_idxs = get_segments_from_sentences_fix_sup(merged_sentences, merged_sp_idxs)
    return merged_text, merged_spans, segments, merged_sp_idxs


class HotpotQADataset(QuestionAndParagraphsDataset):
    """ A class for handling a QA dataset for hotpot. """

    def __init__(self, questions: List[HotpotQuestion], batcher: ListBatcher, sample_seed=18,
                 group_pairs_in_batches=True, distractor_pairs=1):
        self.questions = questions
        self.batcher = batcher
        self.group_pairs_in_batches = group_pairs_in_batches
        self.distractor_pairs = distractor_pairs

        self.qid2question = {q.question_id: q for q in questions}

        self.random = np.random.RandomState(seed=sample_seed)

        self.gold_samples = self._build_gold_samples()
        self.epoch_samples = None

    def _build_gold_samples(self, sample=None):
        gold_samples = []
        questions = self.questions
        if sample is not None:
            questions = self.random.choice(questions, sample, replace=False)
        for question in questions:
            pars_order = [0, 1]
            self.random.shuffle(pars_order)
            pars = [question.supporting_facts[i] for i in pars_order]
            spans = [question.gold_spans[i] for i in pars_order]
            merged_text, merged_spans, sentence_segments, sup_idxs = \
                merge_paragraphs(pars, spans, [x.supporting_sentence_ids for x in pars])
            gold_samples.append(SpanQuestionAndParagraphs(question=question.question_tokens, paragraphs=[merged_text],
                                                          spans=merged_spans, question_id=question.question_id,
                                                          answer=question.answer, q_type=question.q_type,
                                                          sentence_segments=[sentence_segments],
                                                          supporting_facts=sup_idxs))

        return gold_samples

    def get_batches(self, n_batches):
        if len(self) < n_batches:
            raise ValueError()
        return itertools.islice(self.get_epoch(), n_batches)

    def get_samples(self, n_samples: int):
        n_batches = self.batcher.epoch_size(n_samples)
        if not self.group_pairs_in_batches:
            return self.batcher.get_epoch(self.random.choice(self.gold_samples, n_samples, replace=False)), n_batches
        return self._build_pair_batches(self.random.choice(self.gold_samples, n_samples, replace=False)), n_batches

    def get_epoch(self):
        self.gold_samples = self._build_gold_samples()  # Shuffling the merges
        if self.group_pairs_in_batches:
            return self._build_pair_batches(self.gold_samples)
        np.random.shuffle(self.gold_samples)
        return self.batcher.get_epoch(self.gold_samples)

    def _build_pair_batches(self, gold_questions):
        np.random.shuffle(gold_questions)

        for batch_golds in self.batcher.get_epoch(gold_questions):
            batch = []
            for gold in batch_golds:
                batch.append(gold)
                for _ in range(self.distractor_pairs):
                    rand_gold_idx = self.random.randint(2)
                    pool = [(par, span) for par, span in zip(self.qid2question[gold.question_id].distractors,
                                                             self.qid2question[gold.question_id].distractor_spans)]
                    pool += [(self.qid2question[gold.question_id].supporting_facts[rand_gold_idx],
                              self.qid2question[gold.question_id].gold_spans[rand_gold_idx])]
                    rand_pars_idxs = self.random.choice(len(pool), size=2, replace=False)
                    pars, spans = zip(*[pool[i] for i in rand_pars_idxs])
                    sup_facts = [[] if idx+1 < len(pool) else par.supporting_sentence_ids
                                 for idx, par in zip(rand_pars_idxs, pars)]
                    merged_text, merged_spans, sentence_segments, sup_idxs = merge_paragraphs(pars, spans, sup_facts)
                    batch.append(
                        SpanQuestionAndParagraphs(question=gold.question, paragraphs=[merged_text],
                                                  spans=merged_spans, question_id=gold.question_id,
                                                  answer=gold.answer, q_type=gold.q_type,
                                                  sentence_segments=[sentence_segments],
                                                  supporting_facts=sup_idxs))
            yield batch

    def get_spec(self):
        batch_size = self.batcher.get_fixed_batch_size()
        num_contexts = 1
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
        return self.batcher.epoch_size(len(self.gold_samples))


class HotpotFullQADistractorsDataset(QuestionAndParagraphsDataset):
    def __init__(self, questions: List[HotpotQuestion], batcher: ListBatcher):
        self.questions = questions
        self.batcher = batcher

        self.samples, self.gold_idxs = self._build_full_dataset()

    def _build_full_dataset(self):
        samples = []
        gold_idxs = []
        for question in self.questions:
            pars_ans_spans = list(zip(question.distractors + question.supporting_facts,
                                      question.distractor_spans + question.gold_spans))
            for i, (p1, s1) in enumerate(pars_ans_spans):
                for p2, s2 in pars_ans_spans[i + 1:]:
                    if p1 in question.supporting_facts and p2 in question.supporting_facts:
                        gold_idxs.append(len(samples))
                    pars = [p1, p2]
                    spans = [s1, s2]
                    pars_order = [0, 1]
                    np.random.shuffle(pars_order)
                    pars = [pars[i] for i in pars_order]
                    spans = [spans[i] for i in pars_order]
                    sup_facts = [[] if par not in question.supporting_facts else par.supporting_sentence_ids
                                 for par in pars]
                    merged_text, merged_spans, sentence_segments, sup_idxs = merge_paragraphs(pars, spans, sup_facts)
                    samples.append(
                        SpanQuestionAndParagraphs(question=question.question_tokens, paragraphs=[merged_text],
                                                  spans=merged_spans, question_id=question.question_id,
                                                  answer=question.answer, q_type=question.q_type,
                                                  sentence_segments=[sentence_segments],
                                                  supporting_facts=sup_idxs))
        return samples, gold_idxs

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


class HotpotTextLengthPreprocessorWithSpans(Preprocessor):
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
        for i, spans in enumerate(question.gold_spans):
            question.gold_spans[i] = spans[spans[:, 1] < self.num_tokens_th]
        for i, spans in enumerate(question.distractor_spans):
            question.distractor_spans[i] = spans[spans[:, 1] < self.num_tokens_th]
        return question


class HotpotQuestionFilterWithSpans(SampleFilter):
    def __init__(self, num_distractors_th, keep_yes_no=True):
        self.num_distractors_th = num_distractors_th
        self.keep_yes_no = keep_yes_no

    def keep(self, question: HotpotQuestion) -> bool:
        is_yes_no = question.q_type == 'comparison' and question.answer in {'yes', 'no'}
        return len(question.distractors) >= self.num_distractors_th and \
               (any(len(spans) > 0 for spans in question.gold_spans) or (is_yes_no and self.keep_yes_no))


class HotpotQATrainingData(TrainingDataHandler):
    def __init__(self, corpus: HotpotQuestions, train_batcher: ListBatcher, dev_batcher: ListBatcher,
                 sample_filter: Optional[SampleFilter] = None, preprocessor: Optional[Preprocessor] = None,
                 sample_train=None, sample_dev=None, sample_seed=18, group_pairs_in_batches=True,
                 distractor_pairs=1):
        super().__init__(train_batcher, dev_batcher, sample_filter, preprocessor, sample_train, sample_dev, sample_seed)
        self.corpus = corpus
        self.group_pairs_in_batches = group_pairs_in_batches
        self.distractor_pairs = distractor_pairs
        self._train = None
        self._dev = None

    def get_train(self) -> Dataset:
        self._load_data()
        return HotpotQADataset(self._train, self.train_batcher, group_pairs_in_batches=self.group_pairs_in_batches,
                               distractor_pairs=self.distractor_pairs)

    def get_eval(self) -> Dict[str, Dataset]:
        self._load_data()
        eval_sets = dict(
            train=HotpotQADataset(self._train,
                                  self.dev_batcher,
                                  group_pairs_in_batches=self.group_pairs_in_batches,
                                  distractor_pairs=self.distractor_pairs),
            dev=HotpotQADataset(self._dev, self.dev_batcher,
                                group_pairs_in_batches=self.group_pairs_in_batches,
                                distractor_pairs=self.distractor_pairs))
        return eval_sets

    def __getstate__(self):
        state = self.__dict__
        state["_train"] = None
        state["_dev"] = None
        return state

    def __setstate__(self, state):
        if 'distractor_pairs' not in state:
            state['distractor_pairs'] = 1
        self.__dict__ = state
