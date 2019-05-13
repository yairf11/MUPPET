from typing import List, Optional, Dict

import numpy as np
import tensorflow as tf
from itertools import chain
from tensorflow.keras.preprocessing.sequence import pad_sequences

from hotpot.configurable import Configurable
from hotpot.data_handling.qa_training_data import SpanQuestionAndParagraphs
from hotpot.data_handling.relevance_training_data import BinaryQuestionAndParagraphs, IterativeQuestionAndParagraphs
from hotpot.data_handling.dataset import QuestionAndParagraphs, QuestionAndParagraphsSpec
from hotpot.nn.embedder import WordEmbedder, CharEmbedder

"""
Classes to map python objects we want to classify into numpy arrays we can feed into Tensorflow,
e.i. to map (quesiton-context-answer) -> {tf.placeholder / numpy arrays}
"""


class AnswerEncoder(Configurable):
    """ Encode just the answer - can be anything, really """

    def init(self, input_spec: QuestionAndParagraphsSpec) -> None:
        raise NotImplementedError()

    def encode(self, batch) -> Dict:
        raise NotImplementedError()

    def get_placeholders(self) -> List:
        raise NotImplementedError()


class BinaryAnswerEncoder(AnswerEncoder):
    def __init__(self):
        self.answers = None
        self.batch_size = None

    def get_placeholders(self) -> List:
        return [self.answers]

    def init(self, input_spec: QuestionAndParagraphsSpec):
        self.batch_size = input_spec.batch_size
        self.answers = tf.placeholder('int32', [self.batch_size, ], name='answers')

    def encode(self, batch: List[BinaryQuestionAndParagraphs]) -> Dict:
        return {self.answers: np.array([x.label for x in batch])}

    def __getstate__(self):
        return {}

    def __setstate__(self, state):
        return self.__init__()


class IterativeAnswerEncoder(AnswerEncoder):
    def __init__(self, group: bool):
        self.first_answers = None
        self.second_answers = None
        self.batch_size = None
        self.group = group
        self.group_ids = None

    def get_placeholders(self) -> List:
        return [x for x in [self.first_answers, self.second_answers, self.group_ids] if x is not None]

    def init(self, input_spec: QuestionAndParagraphsSpec):
        self.batch_size = input_spec.batch_size
        self.first_answers = tf.placeholder('int32', [self.batch_size, ], name='first_answers')
        self.second_answers = tf.placeholder('int32', [self.batch_size, ], name='second_answers')
        if self.group:
            self.group_ids = tf.placeholder('int32', [self.batch_size, ], name='group_ids')

    def encode(self, batch: List[IterativeQuestionAndParagraphs]) -> Dict:
        if self.group:
            group_id = np.zeros(len(batch), dtype=np.int32)
            qid_to_group_id = {}
            for idx, question in enumerate(batch):
                if question.question_id in qid_to_group_id:
                    group_id[idx] = qid_to_group_id[question.question_id]
                else:
                    group_id[idx] = len(qid_to_group_id)
                    qid_to_group_id[question.question_id] = group_id[idx]
            return {self.first_answers: np.array([x.first_label for x in batch]),
                    self.second_answers: np.array([x.second_label for x in batch]),
                    self.group_ids: group_id}
        return {self.first_answers: np.array([x.first_label for x in batch]),
                self.second_answers: np.array([x.second_label for x in batch])}

    def __getstate__(self):
        return dict(group=self.group)

    def __setstate__(self, state):
        if 'group' not in state:
            state['group'] = False
        return self.__init__(state["group"])


class GroupedSpanAnswerEncoder(AnswerEncoder):
    """ Encode the answer spans into bool (span_start) and (span_end) arrays, and also record
    the group_id if one is present in the answer. Used for the "shared-norm" approach """

    def __init__(self, group=True):
        self.answer_starts = None
        self.answer_ends = None
        self.group_ids = None
        self.group = group

    def get_placeholders(self) -> List:
        return [self.answer_starts, self.answer_ends, self.group_ids]

    def init(self, input_spec: QuestionAndParagraphsSpec):
        if input_spec.max_num_contexts > 1:
            raise NotImplementedError()
        self.answer_starts = tf.placeholder('bool', [input_spec.batch_size, None],
                                            name='answer_starts')
        self.answer_ends = tf.placeholder('bool', [input_spec.batch_size, None],
                                          name='answer_ends')
        self.group_ids = tf.placeholder('int32', [input_spec.batch_size], name='group_ids')

    def encode(self, batch: List[SpanQuestionAndParagraphs]) -> Dict:
        context_word_dim = self.answer_starts.shape.as_list()[-1]
        if context_word_dim is None:
            context_word_dim = max(len(par) for point in batch for par in point.paragraphs)
        answer_starts = np.zeros((len(batch), context_word_dim), dtype=np.bool)
        answer_ends = np.zeros((len(batch), context_word_dim), dtype=np.bool)
        group_id = np.zeros(len(batch), dtype=np.int32)
        qid_to_group_id = {}
        for idx, question in enumerate(batch):
            if question.question_id in qid_to_group_id:
                group_id[idx] = qid_to_group_id[question.question_id]
            else:
                group_id[idx] = len(qid_to_group_id)
                qid_to_group_id[question.question_id] = group_id[idx]
            answer_spans = question.spans
            answer_spans = answer_spans[answer_spans[:, 1] < context_word_dim]

            answer_starts[idx, answer_spans[:, 0]] = True
            answer_ends[idx, answer_spans[:, 1]] = True
        if not self.group:
            group_id = np.arange(0, len(batch))
        return {self.answer_starts: answer_starts, self.answer_ends: answer_ends,
                self.group_ids: group_id}

    def __getstate__(self):
        return dict(group=self.group)

    def __setstate__(self, state):
        return self.__init__(state["group"])


class GroupedSpanAnswerEncoderWithYesNo(GroupedSpanAnswerEncoder):
    def __init__(self, group=True):
        super().__init__(group)
        self.is_yes_no = None
        self.yes_no_answer = None

    def get_placeholders(self):
        return super().get_placeholders() + [self.is_yes_no, self.yes_no_answer]

    def init(self, input_spec: QuestionAndParagraphsSpec):
        super().init(input_spec)
        self.is_yes_no = tf.placeholder('int32', [input_spec.batch_size], name='is_yes_no')
        self.yes_no_answer = tf.placeholder('int32', [input_spec.batch_size], name='yes_no_answer')

    def encode(self, batch: List[SpanQuestionAndParagraphs]):
        enc_dict = super().encode(batch)
        is_yes_no = np.array([int(x.q_type == 'comparison' and x.answer in {'yes', 'no'}) for x in batch])
        yes_no_answer = np.array([int(x.answer == 'yes') for x in batch])
        enc_dict.update({self.is_yes_no: is_yes_no,
                         self.yes_no_answer: yes_no_answer})
        return enc_dict

    def __getstate__(self):
        return dict(group=self.group)

    def __setstate__(self, state):
        return self.__init__(state["group"])


class GroupedSpanAnswerEncoderFullHotpot(GroupedSpanAnswerEncoderWithYesNo):
    def __init__(self, group=True):
        super().__init__(group)
        self.sentence_labels = None

    def get_placeholders(self):
        return super().get_placeholders() + [self.sentence_labels]

    def init(self, input_spec: QuestionAndParagraphsSpec):
        super().init(input_spec)
        self.sentence_labels = tf.placeholder('int32', [input_spec.batch_size, None], name='sentence_labels')

    def encode(self, batch: List[SpanQuestionAndParagraphs]):
        enc_dict = super().encode(batch)
        max_num_sentences = max(max(x.sentence_segments[0])+1 for x in batch)
        sentence_labels = np.zeros((len(batch), max_num_sentences), dtype=np.int32)
        for idx, x in enumerate(batch):
            sentence_labels[idx, x.supporting_facts] = 1
        enc_dict.update({self.sentence_labels: sentence_labels})
        return enc_dict

    def __getstate__(self):
        return dict(group=self.group)

    def __setstate__(self, state):
        return self.__init__(state["group"])


class QuestionsAndParagraphsEncoder(Configurable):
    """
    Uses a WordEmbedder/CharEmbedder (passed in by the client in `init`) to encode text into padded batches of arrays.
    Important to note: this handles any number of paragraphs with a single question. This is important because
     we might want to use this for a variable number of passages.
    This class encodes only the texts, not the answers!
    """

    def __init__(self,
                 answer_encoder: AnswerEncoder,
                 doc_size_th: Optional[int] = None,
                 use_sentence_segments: bool = False,
                 force_precomputed_sentences: bool = False,
                 paragraph_as_sentence: bool = False):
        # Parameters
        self.answer_encoder = answer_encoder
        self.doc_size_th = doc_size_th
        self.use_sentence_segments = use_sentence_segments
        self.force_precomputed_sentences = force_precomputed_sentences
        self.paragraph_as_sentence = paragraph_as_sentence

        self._word_embedder = None
        self._char_emb = None

        # Internal stuff we need to set on `init`
        self.len_opt = None
        self.batch_size = None
        self.max_context_word_dim = None
        self.max_ques_word_dim = None
        self.max_char_dim = None

        self.num_contexts_opt = None
        self.max_num_contexts = None

        self.context_words_multiple = None
        self.context_chars_multiple = None
        self.context_len_multiple = None
        self.context_word_len_multiple = None
        self.num_contexts = None
        self.question_words = None
        self.question_chars = None
        self.question_len = None
        self.question_word_len = None

        self.context_sentence_segments_multiple = None
        self.context_num_sentences = None

    @property
    def version(self):
        return 3

    def init(self, input_spec: QuestionAndParagraphsSpec, len_opt: bool,
             word_emb: WordEmbedder, char_emb: Optional[CharEmbedder], num_contexts_opt: bool):
        """
        Initialize the encoder with the datasets' specs and embedders.
        :param input_spec: The spec of the dataset we are going to encode. includes batch size, context lengths etc.
        :param len_opt: if True, there is no limit to context size (length dimension = None)
        :param word_emb: the word embedder
        :param char_emb: the character embedder
        :param num_contexts_opt: if True, there is no limit to the number of contexts (context dimension = None)
        :return:
        """

        self._word_embedder = word_emb
        self._char_emb = char_emb

        self.batch_size = input_spec.batch_size
        self.len_opt = len_opt
        self.num_contexts_opt = num_contexts_opt

        if self._char_emb is not None:
            if input_spec.max_word_size is not None:
                self.max_char_dim = min(self._char_emb.get_word_size_th(), input_spec.max_word_size)
            else:
                self.max_char_dim = self._char_emb.get_word_size_th()
        else:
            self.max_char_dim = 1

        if not self.len_opt:
            self.max_ques_word_dim = input_spec.max_num_quesiton_words
            self.max_context_word_dim = input_spec.max_num_context_words
            if self.max_ques_word_dim is None or self.max_context_word_dim is None:
                raise ValueError()
            if self.doc_size_th is not None:
                self.max_context_word_dim = min(self.max_context_word_dim, self.doc_size_th)
        else:
            self.max_ques_word_dim = None
            self.max_context_word_dim = None

        if not self.num_contexts_opt:
            self.max_num_contexts = input_spec.max_num_contexts
            if self.max_num_contexts is None:
                raise ValueError()
        else:
            self.max_num_contexts = None

        n_question_words = self.max_ques_word_dim
        n_context_words = self.max_context_word_dim
        num_contexts = self.max_num_contexts
        batch_size = self.batch_size

        self.num_contexts = tf.placeholder('int32', [batch_size], name='num_contexts')
        self.context_words_multiple = tf.placeholder('int32', [batch_size, num_contexts, n_context_words],
                                                     name=f'context_words')
        self.context_len_multiple = tf.placeholder('int32', [batch_size, num_contexts], name=f'context_lens')

        self.question_words = tf.placeholder('int32', [batch_size, n_question_words], name='question_words')
        self.question_len = tf.placeholder('int32', [batch_size], name='question_len')

        if self.use_sentence_segments:
            self.context_num_sentences = tf.placeholder('int32', [batch_size, num_contexts], name='num_sentences')
            self.context_sentence_segments_multiple = tf.placeholder('int32',
                                                                     [batch_size, num_contexts, n_context_words],
                                                                     name='sentence_segments')
        else:
            self.context_num_sentences = None
            self.context_sentence_segments_multiple = None

        if self._char_emb:
            self.question_chars = tf.placeholder('int32', [batch_size, n_question_words, self.max_char_dim],
                                                 name='question_chars')
            self.question_word_len = tf.placeholder('int32', [batch_size, n_question_words], name='question_char_len')
            self.context_chars_multiple = tf.placeholder('int32', [batch_size, num_contexts, n_context_words,
                                                                   self.max_char_dim],
                                                         name=f'context_chars')
            self.context_word_len_multiple = tf.placeholder('int32', [batch_size, num_contexts, n_context_words],
                                                            name=f'context_char_lens')
        else:
            self.context_chars_multiple = None
            self.question_chars = None
            self.context_word_len_multiple = None
            self.question_word_len = None

        self.answer_encoder.init(input_spec)

    def get_placeholders(self):
        return [x for x in
                [self.question_len, self.question_words, self.question_chars,
                 self.context_len_multiple, self.context_words_multiple, self.context_chars_multiple,
                 self.question_word_len, self.context_word_len_multiple, self.num_contexts,
                 self.context_num_sentences, self.context_sentence_segments_multiple]
                if
                x is not None] + self.answer_encoder.get_placeholders()

    def encode(self, batch: List[QuestionAndParagraphs], is_train: bool):
        batch_size = len(batch)
        if self.batch_size is not None:
            if self.batch_size < batch_size:
                raise ValueError("Batch sized we pre-specified as %d, "
                                 "but got a batch of %d" % (self.batch_size, batch_size))
            # We have a fixed batch size, so we will pad our inputs with zeros along the batch dimension
            batch_size = self.batch_size

        context_word_dim, ques_word_dim, max_char_dim = \
            self.max_context_word_dim, self.max_ques_word_dim, self.max_char_dim

        feed_dict = {}

        num_contexts = self.max_num_contexts

        if num_contexts is not None:
            batch_num_contexts = np.array(
                [min(point.num_contexts, num_contexts) for point in batch])  # Might remove contexts
        else:
            batch_num_contexts = np.array([point.num_contexts for point in batch])

        feed_dict[self.num_contexts] = batch_num_contexts

        context_dim = max(batch_num_contexts) if num_contexts is None else num_contexts

        # compute the question/word lengths
        if context_word_dim is not None:
            context_len_multiple = pad_sequences([[min(len(par), context_word_dim) for par in point.paragraphs]
                                                  for point in batch],
                                                 dtype='int32', padding='post', truncating='post',
                                                 maxlen=context_dim)  # Might truncate contexts
        else:
            context_len_multiple = pad_sequences([[len(par) for par in point.paragraphs]
                                                  for point in batch],
                                                 dtype='int32', padding='post', truncating='post', maxlen=context_dim)
            context_word_dim = context_len_multiple.max()

        question_len = np.array([len(x.question) for x in batch], dtype='int32')
        if ques_word_dim is not None:
            if question_len.max() > ques_word_dim:
                raise ValueError("Have a question of len %d but max ques dim is %d" %
                                 (question_len.max(), ques_word_dim))
        else:
            ques_word_dim = question_len.max()

        feed_dict[self.context_len_multiple] = context_len_multiple
        feed_dict[self.question_len] = question_len

        # Setup word placeholders
        if self._word_embedder is not None:
            context_words_multiple = np.zeros([batch_size, context_dim, context_word_dim], dtype='int32')
            question_words = np.zeros([batch_size, ques_word_dim], dtype='int32')
            feed_dict[self.context_words_multiple] = context_words_multiple
            feed_dict[self.question_words] = question_words
        else:
            question_words, context_words_multiple = None, None

        # Setup char placeholders
        if self._char_emb is not None:
            context_chars_multiple = np.zeros([batch_size, context_dim, context_word_dim, max_char_dim], dtype='int32')
            question_chars = np.zeros([batch_size, ques_word_dim, max_char_dim], dtype='int32')
            context_word_len_multiple = np.zeros([batch_size, context_dim, context_word_dim], dtype='int32')
            question_word_len = np.zeros([batch_size, ques_word_dim], dtype='int32')
            feed_dict[self.question_chars] = question_chars
            feed_dict[self.question_word_len] = question_word_len
            feed_dict[self.context_chars_multiple] = context_chars_multiple
            feed_dict[self.context_word_len_multiple] = context_word_len_multiple
        else:
            context_chars_multiple, question_chars, question_word_len, context_word_len_multiple = None, None, None, None

        # Now fill in the place holders by iterating through the data
        for doc_ix, doc in enumerate(batch):
            for word_ix, word in enumerate(doc.question):
                if self._word_embedder is not None:
                    ix = self._word_embedder.context_word_to_ix(word, is_train)
                    question_words[doc_ix, word_ix] = ix
                if self._char_emb is not None:
                    question_word_len[doc_ix, word_ix] = min(self.max_char_dim, len(word))
                    for char_ix, char in enumerate(word):
                        if char_ix == self.max_char_dim:
                            break
                        question_chars[doc_ix, word_ix, char_ix] = self._char_emb.char_to_ix(char)

            for context_ix, context in enumerate(doc.paragraphs[:context_dim]):
                for word_ix, word in enumerate(context[:context_word_dim]):
                    if self._word_embedder is not None:
                        ix = self._word_embedder.context_word_to_ix(word, is_train)
                        context_words_multiple[doc_ix, context_ix, word_ix] = ix

                    if self._char_emb is not None:
                        context_word_len_multiple[doc_ix, context_ix, word_ix] = min(self.max_char_dim, len(word))
                        for char_ix, char in enumerate(word):
                            if char_ix == self.max_char_dim:
                                break
                            context_chars_multiple[doc_ix, context_ix, word_ix, char_ix] = self._char_emb.char_to_ix(
                                char)

        # compute and setup sentence segments
        if self.use_sentence_segments:  # We are assuming that it is not possible for changing number of contexts.
            # by default, use the segments generated from get_sentence_segments
            if not self.force_precomputed_sentences:
                if self.paragraph_as_sentence:
                    context_segments = [[[0]*len(par)
                                         for idx, par in enumerate(x.paragraphs)] for x in batch]
                else:
                    context_segments = [[get_sentence_segments(par)
                                         # if x.sentence_segments is None
                                         # else x.sentence_segments[idx][:len(par)]  # in case the text was truncated
                                         for idx, par in enumerate(x.paragraphs)] for x in batch]
            else:
                context_segments = [[x.sentence_segments[idx][:len(par)]
                                     for idx, par in enumerate(x.paragraphs)] for x in batch]
            context_num_sents = np.array([[max(segment) + 1 for segment in segments]
                                          for segments in context_segments])
            padded_context_segments = np.array([pad_sequences(segments,
                                                              dtype='int32', padding='post', truncating='post',
                                                              value=-1,
                                                              maxlen=context_word_dim)
                                                for segments in context_segments])
            max_num_sents = np.max(context_num_sents, axis=0)
            accum_lens = np.zeros_like(max_num_sents)
            for sample_idx, sample_lens in enumerate(context_num_sents):
                for context_idx, context_len in enumerate(sample_lens):
                    padded_context_segments[sample_idx][context_idx][:len(context_segments[sample_idx][context_idx])] \
                        += accum_lens[context_idx]
                    accum_lens[context_idx] += max_num_sents[context_idx]
            feed_dict[self.context_num_sentences] = context_num_sents
            feed_dict[self.context_sentence_segments_multiple] = padded_context_segments

        # Answer placeholders
        feed_dict.update(self.answer_encoder.encode(batch))

        return feed_dict

    def __getstate__(self):
        # The placeholders are considered transient, the model
        # will be re-initailized when re-loaded
        state = dict(
            answer_encoder=self.answer_encoder,
            doc_size_th=self.doc_size_th,
            version=self.version,
            use_sentence_segments=self.use_sentence_segments,
            force_precomputed_sentences=self.force_precomputed_sentences,
            paragraph_as_sentence=self.paragraph_as_sentence
        )
        return state

    def __setstate__(self, state):
        if "use_sentence_segments" not in state:
            state["use_sentence_segments"] = False
        if "force_precomputed_sentences" not in state:
            state["force_precomputed_sentences"] = False
        if "paragraph_as_sentence" not in state:
            state["paragraph_as_sentence"] = False
        super().__setstate__(state)


def get_sentence_segments(words: List[str]):
    """
    returns a list of indexes indicating for each word in which sentence it is.
    Sentences are determined by '.' tokens. This is a hack, but will do for initial results
    """
    segments = []
    current_segment = 0
    for word in words:
        segments.append(current_segment)
        if '.' == word:
            current_segment += 1
    return segments
