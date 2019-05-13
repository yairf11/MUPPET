from typing import List, Set
import numpy as np
import tensorflow as tf
from keras_preprocessing.sequence import pad_sequences
from tqdm import tqdm

from hotpot.data_handling.dataset import Dataset, QuestionAndParagraphs, QuestionAndParagraphsSpec, \
    QuestionAndParagraphsDataset
from hotpot.data_handling.relevance_training_data import BinaryQuestionAndParagraphs, IterativeQuestionAndParagraphs
from hotpot.elmo.lm_model import load_elmo_pretrained_token_embeddings
from hotpot.model_dir import ModelDir
from hotpot.models.iterative_context_models import IterativeContextMaxSentenceModel
from hotpot.models.multiple_context_models import MultipleContextModel
from hotpot.models.single_context_models import SingleContextMaxSentenceModel


def optimistic_restore(session, save_file):
    reader = tf.train.NewCheckpointReader(save_file)
    saved_shapes = reader.get_variable_to_shape_map()
    var_names = sorted([(var.name, var.name.split(':')[0]) for var in tf.global_variables()
                        if var.name.split(':')[0] in saved_shapes])
    restore_vars = []
    name2var = dict(zip(map(lambda x: x.name.split(':')[0], tf.global_variables()), tf.global_variables()))
    with tf.variable_scope('', reuse=True):
        for var_name, saved_var_name in var_names:
            curr_var = name2var[saved_var_name]
            var_shape = curr_var.get_shape().as_list()
            if var_shape == saved_shapes[saved_var_name]:
                restore_vars.append(curr_var)
    saver = tf.train.Saver(restore_vars)
    saver.restore(session, save_file)


class TrainedEncoder(object):
    """
    Base class for encoders
    """

    def __init__(self, model_dir_path: str, vocabulary: Set[str], spec: QuestionAndParagraphsSpec, loader,
                 use_char_inputs: bool, use_ema: bool, checkpoint: str='best'):
        if checkpoint not in {'best', 'latest'}:
            raise ValueError("checkpoint value must be either 'best' or 'latest'.")
        self.model_dir = ModelDir(model_dir_path)
        self.checkpoint = None
        if checkpoint == 'best':
            self.checkpoint = self.model_dir.get_best_weights()
        if self.checkpoint is not None:
            print("Using best weights")
        else:
            print("Using latest checkpoint")
            self.checkpoint = self.model_dir.get_latest_checkpoint()
        print(f"Restoring checkpoint: {self.checkpoint}")
        self.model = self.model_dir.get_model()
        assert isinstance(self.model, MultipleContextModel)
        if self.model.use_elmo and use_char_inputs:
            self.model.lm_model.embed_weights_file = None
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True), graph=tf.Graph())
        with self.sess.graph.as_default():
            self.model.set_inputs(None, loader, voc=vocabulary, input_spec=spec)
            inputs = self.model.get_placeholders()
            input_dict = {p: x for p, x in zip(self.model.get_placeholders(), inputs)}
            with self.sess.as_default():
                _ = self.model.get_predictions_for(input_dict)  # for building the model
            if not self.model.use_elmo or not use_char_inputs:
                saver = tf.train.Saver()
                saver.restore(self.sess, self.checkpoint)
            else:
                self.sess.run(tf.global_variables_initializer())
                optimistic_restore(self.sess, self.checkpoint)

            if use_ema:
                ema = tf.train.ExponentialMovingAverage(0)
                reader = tf.train.NewCheckpointReader(self.checkpoint)
                expected_ema_names = {ema.average_name(x): x for x in tf.trainable_variables()
                                      if reader.has_tensor(ema.average_name(x))}
                if len(expected_ema_names) > 0:
                    print("Restoring EMA variables")
                    saver = tf.train.Saver(expected_ema_names)
                    saver.restore(self.sess, self.checkpoint)
        if self.model.use_elmo and not use_char_inputs:
            # TODO: Might be redundant! the weights are already saved in the checkpoint
            elmo_token_embed_placeholder, elmo_token_embed_init = self.model.get_elmo_token_embed_ph_and_op()
            print("Loading ELMo weights...")
            elmo_token_embed_weights = load_elmo_pretrained_token_embeddings(self.model.lm_model.embed_weights_file)
            self.sess.run(elmo_token_embed_init, feed_dict={elmo_token_embed_placeholder: elmo_token_embed_weights})

        self.sess.graph.finalize()


class SentenceEncoderSingleContext(TrainedEncoder):
    """
    Class for encoding a paragraph into sentences.
    Given a trained model, loads the model and returns per-sentence representations.
    """

    def __init__(self, model_dir_path: str, vocabulary: Set[str], spec: QuestionAndParagraphsSpec, loader,
                 use_char_inputs: bool, use_ema: bool, checkpoint: str = 'best'):
        super().__init__(model_dir_path=model_dir_path, vocabulary=vocabulary, spec=spec, loader=loader,
                         use_char_inputs=use_char_inputs, use_ema=use_ema, checkpoint=checkpoint)

        self.sentence_encodings_name = "sentences_enc/encode_context:0"
        self.question_embedding_name = "seq_enc/Maximum:0"  # TODO give a meaningful name to the representation..
        self.dense_weights_name = "sentence_level_predictions/dense/kernel:0"

    def encode_paragraphs(self, batch: List[QuestionAndParagraphs],
                          batch_size=None, show_progress=False) -> List[np.ndarray]:
        # TODO: allow encoding only the paragraphs without all the other stuff!
        final_encs = []
        batch_size = min(self.model.max_batch_size if self.model.max_batch_size else len(batch),
                         batch_size if batch_size is not None else len(batch))
        # [questions[i:i + 32] for i in range(0, len(questions), 32)]
        # todo divide longer sequences into smaller batches? or maybe handle this outside?
        for _ in range(0, len(batch), batch_size) if not show_progress else tqdm(range(0, len(batch), batch_size)):
            feed_dict = self.model.encode(batch[:batch_size], False)
            encodings = self.sess.run(self.sentence_encodings_name, feed_dict=feed_dict)
            num_sentences = [v for k, v in feed_dict.items() if 'num_sentences' in k.name][0].squeeze(axis=1)
            final_encs.extend([sentences[:num_sents] for sentences, num_sents in zip(encodings, num_sentences)])
            batch = batch[batch_size:]
        return final_encs

    def encode_questions(self, batch: List[QuestionAndParagraphs], return_search_vectors: bool, show_progress=False):
        final_encs = []
        batch_size = self.model.max_batch_size if self.model.max_batch_size else len(batch)
        for _ in range(0, len(batch), batch_size) if not show_progress else tqdm(range(0, len(batch), batch_size)):
            feed_dict = self.model.encode(batch[:batch_size], False)
            question_encodings = self.sess.run(self.question_embedding_name, feed_dict=feed_dict)
            final_encs.append(question_encodings)
            batch = batch[batch_size:]
        question_encodings = np.concatenate(final_encs, axis=0)
        if return_search_vectors:
            return self.question_rep_to_search_vector(question_encodings)
        return question_encodings

    def question_rep_to_search_vector(self, question_encodings: np.ndarray):
        weights = self.sess.run(self.dense_weights_name).squeeze(axis=1)
        # we now construct a vector that will preserve the model's ranking by performing a dot-product
        # with the sentence representations. we assume the merger is WithConcatOptions, which has the following form:
        # [p, p-q, p*q, p.dot(q), q]. only p is guaranteed to exist in the vector.
        # [w0, w1, w2,  w3 (1d), w4]
        # We want: p.dot(w0 + w1 + q*w2 + q*w3)
        rep_dim = question_encodings.shape[-1]
        w0 = weights[:rep_dim]
        weights = weights[rep_dim:]
        final_rep = np.expand_dims(w0, axis=0)
        if self.model.merger.sub:
            final_rep = final_rep + np.expand_dims(weights[:rep_dim], axis=0)
            weights = weights[rep_dim:]
        if self.model.merger.hadamard:
            final_rep = final_rep + np.multiply(question_encodings, weights[:rep_dim])
            weights = weights[rep_dim:]
        if self.model.merger.dot:
            final_rep = final_rep + np.multiply(question_encodings, weights[0])
        return final_rep

    def encode_text_questions(self, tokenized_questions: List[List[str]], return_search_vectors: bool,
                              show_progress=False):
        dummy_par = "Hello Hello".split()
        samples = [BinaryQuestionAndParagraphs(question=q, paragraphs=[dummy_par], label=1, num_distractors=0,
                                               question_id='dummy') for q in tokenized_questions]
        return self.encode_questions(samples, return_search_vectors=return_search_vectors, show_progress=show_progress)


class SentenceEncoderIterativeModel(TrainedEncoder):
    """
    Class for encoding a paragraph into sentences, questions into search vectors, and for question reformulation
    Given a trained model, loads the model and returns per-sentence representations.
    """

    def __init__(self, model_dir_path: str, vocabulary: Set[str], spec: QuestionAndParagraphsSpec, loader,
                 use_char_inputs: bool, use_ema: bool, checkpoint: str = 'best'):
        super().__init__(model_dir_path=model_dir_path, vocabulary=vocabulary, spec=spec, loader=loader,
                         use_char_inputs=use_char_inputs, use_ema=use_ema, checkpoint=checkpoint)

        self.question_embedding_name = "seq_enc/encode_question:0"
        self.context_sentence_encodings = {i: f"sentences_enc/encode_context{i}:0"
                                           for i in range(1, spec.max_num_contexts + 1)}
        self.context_prediction_weights = {i: f"context{i}_relevance/sentence_level_predictions/dense/kernel:0"
                                           for i in range(1, spec.max_num_contexts + 1)}
        self.reformulation_name = "reformulation/reformulated_question:0"

    def encode_first_paragraphs(self, batch: List[QuestionAndParagraphs],
                                batch_size=None, show_progress=False) -> List[np.ndarray]:
        # TODO: allow encoding only the paragraphs without all the other stuff!
        final_encs = []
        batch_size = min(self.model.max_batch_size if self.model.max_batch_size else len(batch),
                         batch_size if batch_size is not None else len(batch))
        for _ in range(0, len(batch), batch_size) if not show_progress else tqdm(range(0, len(batch), batch_size),
                                                                                 ncols=80):
            feed_dict = self.model.encode(batch[:batch_size], False)
            encodings = self.sess.run(self.context_sentence_encodings[1], feed_dict=feed_dict)
            num_sentences = [v for k, v in feed_dict.items() if 'num_sentences' in k.name][0][:, 0]
            final_encs.extend([sentences[:num_sents] for sentences, num_sents in zip(encodings, num_sentences)])
            batch = batch[batch_size:]
        return final_encs

    def encode_questions(self, batch: List[QuestionAndParagraphs], return_search_vectors: bool, show_progress=False):
        final_encs = []
        batch_size = self.model.max_batch_size if self.model.max_batch_size else len(batch)
        for _ in range(0, len(batch), batch_size) if not show_progress else tqdm(range(0, len(batch), batch_size)):
            feed_dict = self.model.encode(batch[:batch_size], False)
            question_encodings = self.sess.run(self.question_embedding_name, feed_dict=feed_dict)
            final_encs.append(question_encodings)
            batch = batch[batch_size:]
        question_encodings = np.concatenate(final_encs, axis=0)
        if return_search_vectors:
            return self.question_rep_to_search_vector(question_encodings, context_idx=1)
        return question_encodings

    def question_rep_to_search_vector(self, question_encodings: np.ndarray, context_idx=1):
        weights = self.sess.run(self.context_prediction_weights[context_idx]).squeeze(axis=1)
        # we now construct a vector that will preserve the model's ranking by performing a dot-product
        # with the sentence representations. we assume the merger is WithConcatOptions, which has the following form:
        # [p, p-q, p*q, p.dot(q), q]. only p is guaranteed to exist in the vector.
        # [w0, w1, w2,  w3 (1d), w4]
        # We want: p.dot(w0 + w1 + q*w2 + q*w3)
        rep_dim = question_encodings.shape[-1]
        w0 = weights[:rep_dim]
        weights = weights[rep_dim:]
        final_rep = np.expand_dims(w0, axis=0)
        if self.model.merger.sub:
            final_rep = final_rep + np.expand_dims(weights[:rep_dim], axis=0)
            weights = weights[rep_dim:]
        if self.model.merger.hadamard:
            final_rep = final_rep + np.multiply(question_encodings, weights[:rep_dim])
            weights = weights[rep_dim:]
        if self.model.merger.dot:
            final_rep = final_rep + np.multiply(question_encodings, weights[0])
        return final_rep

    def encode_text_questions(self, tokenized_questions: List[List[str]], return_search_vectors: bool,
                              show_progress=False):
        dummy_par = "Hello Hello".split()
        samples = [IterativeQuestionAndParagraphs(question=q, paragraphs=[dummy_par, dummy_par],
                                                  first_label=1, second_label=1,
                                                  question_id='dummy', sentence_segments=None)
                   for q in tokenized_questions]
        return self.encode_questions(samples, return_search_vectors=return_search_vectors, show_progress=show_progress)

    def reformulate_questions(self, questions_rep: np.ndarray, paragraphs_rep: List[np.ndarray],
                              return_search_vectors: bool, show_progress=False):
        final_encs = []
        batch_size = self.model.max_batch_size if self.model.max_batch_size else len(paragraphs_rep)
        for _ in range(0, len(paragraphs_rep), batch_size) if not show_progress \
                else tqdm(range(0, len(paragraphs_rep), batch_size)):
            batch_sent_lens = np.tile(np.expand_dims([len(x) for x in paragraphs_rep[:batch_size]], axis=1), (1, 2))
            batch_pars = pad_sequences(paragraphs_rep[:batch_size], dtype='float32',
                                       padding='post', truncating='post', value=0.)
            feed_dict = {self.question_embedding_name: questions_rep[:batch_size],
                         self.context_sentence_encodings[1]: batch_pars,
                         self.model.encoder.context_num_sentences: batch_sent_lens,
                         self.model._is_train_placeholder: False}
            reformulations = self.sess.run(self.reformulation_name, feed_dict=feed_dict)
            final_encs.append(reformulations)
            questions_rep = questions_rep[batch_size:]
            paragraphs_rep = paragraphs_rep[batch_size:]
        reformulations = np.concatenate(final_encs, axis=0)
        if return_search_vectors:
            return self.question_rep_to_search_vector(reformulations, context_idx=2)
        return reformulations

    def reformulate_questions_from_texts(self, tokenized_questions: List[List[str]],
                                         tokenized_pars: List[List[str]],
                                         return_search_vectors: bool, show_progress=False, max_batch=None):
        dummy_par = "Hello Hello".split()
        batch = [IterativeQuestionAndParagraphs(question=q, paragraphs=[p, dummy_par],
                                                first_label=1, second_label=1,
                                                question_id='dummy', sentence_segments=None)
                 for q, p in zip(tokenized_questions, tokenized_pars)]
        final_encs = []
        batch_size = self.model.max_batch_size if self.model.max_batch_size else len(batch)
        batch_size = batch_size if max_batch is None else min(batch_size, max_batch)
        for _ in range(0, len(batch), batch_size) if not show_progress else tqdm(range(0, len(batch), batch_size)):
            feed_dict = self.model.encode(batch[:batch_size], False)
            reformulations = self.sess.run(self.reformulation_name, feed_dict=feed_dict)
            final_encs.append(reformulations)
            batch = batch[batch_size:]
        reformulations = np.concatenate(final_encs, axis=0)
        if return_search_vectors:
            return self.question_rep_to_search_vector(reformulations, context_idx=2)
        return reformulations
