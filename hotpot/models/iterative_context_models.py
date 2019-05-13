from typing import Optional, Union
import tensorflow as tf
from tensorflow.python.layers.core import fully_connected

from hotpot.elmo.elmo import ElmoWrapper
from hotpot.elmo.lm_model import LanguageModel
from hotpot.encoder import QuestionsAndParagraphsEncoder
from hotpot.models.multiple_context_models import MultipleContextModel, INTERMEDIATE_LAYER_COLLECTION
from hotpot.nn.attention import AttentionWithPostMapper
from hotpot.nn.embedder import WordEmbedder, CharWordEmbedder
from hotpot.nn.layers import SequenceMapper, SequenceEncoder, FixedMergeLayer, Mapper, MaxPool, MeanPool, \
    get_keras_initialization, AttentionMapper, MergeLayer, get_keras_activation
from hotpot.nn.ops import VERY_NEGATIVE_NUMBER
from hotpot.nn.reformulation_layers import ReformulationLayer
from hotpot.nn.relevance_prediction import BinaryNullPredictor, MultipleBinaryPredictions
from hotpot.nn.sentence_layers import SentencesEncoder

EPSILON = 1e-10  # for numerical stability of log


def merge_weight_predict(is_train, context_rep, question_rep, context_mask,
                         merger, post_merger, max_pool, predictor, answer, multiply_probs=None):
    with tf.variable_scope("merger"):
        c_q_merged_rep = merger.apply(is_train, tensor=context_rep, fixed_tensor=question_rep,
                                      mask=context_mask)

    if post_merger is not None:
        with tf.variable_scope("post_merger"):
            c_q_merged_rep = post_merger.apply(is_train, c_q_merged_rep, mask=context_mask)

    with tf.variable_scope("sentence_level_predictions"):
        sentences_logits = fully_connected(c_q_merged_rep, 1,
                                           use_bias=True,
                                           activation=None,
                                           kernel_initializer=get_keras_initialization('glorot_uniform'))
        max_logits = max_pool.apply(is_train, sentences_logits, context_mask)

        if multiply_probs is not None:
            max_logits = tf.log(multiply_probs + EPSILON) - tf.log(1. + tf.exp(-max_logits) - multiply_probs + EPSILON)

    with tf.variable_scope("predictor"):
        pred = predictor.apply(is_train, max_logits, answer)
    return c_q_merged_rep, sentences_logits, pred


class IterativeContextMaxSentenceModel(MultipleContextModel):
    """
    Model for a iterative relevance prediction. Based on the single version of it - SingleContextMaxSentenceModel.
    This should serve as some simple baseline.
    The general scheme of this model, and other like it, is this:
    1. Encode contexts to sentence representations
    2. Predict first iteration as if it was a single context model
    3. Reformulate the question given the first iteration paragraph
    4. Repeat stage 2 for the second second iteration with the reformulated question
    """

    def __init__(self,
                 encoder: QuestionsAndParagraphsEncoder,
                 word_embed: Optional[WordEmbedder],
                 char_embed: Optional[CharWordEmbedder],
                 embed_mapper: Optional[Union[SequenceMapper, ElmoWrapper]],
                 sequence_encoder: SequenceEncoder,
                 sentences_encoder: SentencesEncoder,
                 sentence_mapper: Optional[SequenceMapper],
                 merger: FixedMergeLayer,
                 post_merger: Optional[Mapper],
                 reformulation_layer: ReformulationLayer,
                 max_batch_size: Optional[int] = None,
                 elmo_model: Optional[LanguageModel] = None
                 ):
        super().__init__(encoder=encoder, word_embed=word_embed, char_embed=char_embed, max_batch_size=max_batch_size,
                         elmo_model=elmo_model)
        self.embed_mapper = embed_mapper
        self.sequence_encoder = sequence_encoder
        self.sentences_encoder = sentences_encoder
        self.sentence_mapper = sentence_mapper
        self.merger = merger
        self.post_merger = post_merger
        self.reformulation_layer = reformulation_layer
        self.predictor = BinaryNullPredictor()
        self.max_pool = MaxPool(map_layer=None, min_val=VERY_NEGATIVE_NUMBER, regular_reshape=True)
        self.mean_pool = MeanPool()

    def _get_predictions_for(self,
                             is_train,
                             question_embed, question_mask,
                             context_embed, context_mask,
                             answer,
                             question_lm, context_lm, sentence_segments, sentence_mask):
        question_rep, context_rep = question_embed, context_embed
        context1_rep, context2_rep = tf.unstack(context_rep, axis=1, num=2)
        context1_mask, context2_mask = tf.unstack(context_mask, axis=1, num=2)
        context1_sentence_segments, context2_sentence_segments = tf.unstack(sentence_segments, axis=1, num=2)
        context1_sentence_mask, context2_sentence_mask = tf.unstack(sentence_mask, axis=1, num=2)
        q_lm_in, c1_lm_in, c2_lm_in = [], [], []
        if self.use_elmo:
            context1_lm, context2_lm = tf.unstack(context_lm, axis=1, num=2)
            q_lm_in = [question_lm]
            c1_lm_in = [context1_lm]
            c2_lm_in = [context2_lm]
        if self.embed_mapper is not None:
            with tf.variable_scope("map_embed"):
                context1_rep = self.embed_mapper.apply(is_train, context1_rep, context1_mask, *c1_lm_in)
            with tf.variable_scope("map_embed", reuse=True):
                context2_rep = self.embed_mapper.apply(is_train, context2_rep, context2_mask, *c2_lm_in)
                question_rep = self.embed_mapper.apply(is_train, question_rep, question_mask, *q_lm_in)

        with tf.variable_scope("seq_enc"):
            question_rep = self.sequence_encoder.apply(is_train, question_rep, question_mask)
            question_rep = tf.identity(question_rep, name='encode_question')
            tf.add_to_collection(INTERMEDIATE_LAYER_COLLECTION, question_rep)

        def encode_sentences(context, sentence_segs, sentence_mask, rep_name):
            context = self.sentences_encoder.apply(context,
                                                   sentence_segs, sentence_mask)
            if self.sentence_mapper is not None:
                with tf.variable_scope('sentence_mapper'):
                    context = self.sentence_mapper.apply(is_train, context, mask=sentence_mask)
            context = tf.identity(context, name=rep_name)
            tf.add_to_collection(INTERMEDIATE_LAYER_COLLECTION, context)
            return context

        with tf.variable_scope('sentences_enc'):
            context1_rep = encode_sentences(context1_rep, context1_sentence_segments, context1_sentence_mask,
                                            'encode_context1')
        with tf.variable_scope('sentences_enc', reuse=True):
            context2_rep = encode_sentences(context2_rep, context2_sentence_segments, context2_sentence_mask,
                                            'encode_context2')

        # First Iteration (same as in the single context model)
        with tf.variable_scope("context1_relevance"):
            c1_q_merged_rep, context1_sentences_logits, context1_pred = \
                merge_weight_predict(is_train=is_train, context_rep=context1_rep, question_rep=question_rep,
                                     context_mask=context1_sentence_mask, merger=self.merger,
                                     post_merger=self.post_merger, max_pool=self.max_pool,
                                     predictor=self.predictor, answer=[answer[0]])

        # Question Reformulation
        with tf.variable_scope("reformulation"):
            reformulated_q = self.reformulation_layer.apply(is_train, c1_q_merged_rep, context1_sentences_logits,
                                                            context1_sentence_mask)
            reformulated_q = tf.identity(reformulated_q, name='reformulated_question')
            tf.add_to_collection(INTERMEDIATE_LAYER_COLLECTION, reformulated_q)

        # Second Iteration
        with tf.variable_scope("context2_relevance"):
            c2_q_merged_rep, context2_sentences_logits, context2_pred = \
                merge_weight_predict(is_train=is_train, context_rep=context2_rep, question_rep=reformulated_q,
                                     context_mask=context2_sentence_mask, merger=self.merger,
                                     post_merger=self.post_merger, max_pool=self.max_pool,
                                     predictor=self.predictor, answer=[answer[1]])

        return MultipleBinaryPredictions([context1_pred, context2_pred])

    def __setstate__(self, state):
        if "sentence_mapper" not in state:
            state["sentence_mapper"] = None
        super().__setstate__(state)


class IterativeContextReReadModel(MultipleContextModel):
    """
    Model for a iterative relevance prediction. Very similar to IterativeContextMaxSentenceModel.
    The key difference here is the complete re-reading of the question and the paragraph from the first iteration
        to reformulate the question for the second iteration.
    This model is of course much less computation friendly and the retrieval will be slower,
        but performance could get much better
    The general scheme of this model, and other like it, is this:
    1. Encode contexts to sentence representations
    2. Predict first iteration as if it was a single context model
    3. Reformulate the question given the first iteration paragraph
    4. Repeat stage 2 for the second second iteration with the reformulated question
    """

    def __init__(self,
                 encoder: QuestionsAndParagraphsEncoder,
                 word_embed: Optional[WordEmbedder],
                 char_embed: Optional[CharWordEmbedder],
                 embed_mapper: Optional[Union[SequenceMapper, ElmoWrapper]],
                 sequence_encoder: SequenceEncoder,
                 sentences_encoder: SentencesEncoder,
                 sentence_mapper: Optional[SequenceMapper],
                 merger: FixedMergeLayer,
                 post_merger: Optional[Mapper],
                 reread_mapper: Optional[Union[SequenceMapper, ElmoWrapper]],
                 pre_attention_mapper: Optional[SequenceMapper],
                 context_to_question_attention: Optional[AttentionWithPostMapper],
                 question_to_context_attention: Optional[AttentionWithPostMapper],
                 first_predictor: BinaryNullPredictor,
                 second_predictor: BinaryNullPredictor,
                 reformulate_by_context: bool,
                 multiply_iteration_probs: bool,
                 max_batch_size: Optional[int] = None,
                 elmo_model: Optional[LanguageModel] = None
                 ):
        super().__init__(encoder=encoder, word_embed=word_embed, char_embed=char_embed, max_batch_size=max_batch_size,
                         elmo_model=elmo_model)
        self.embed_mapper = embed_mapper
        self.sequence_encoder = sequence_encoder
        self.sentences_encoder = sentences_encoder
        self.sentence_mapper = sentence_mapper
        self.merger = merger
        self.post_merger = post_merger
        self.reread_mapper = reread_mapper
        self.pre_attention_mapper = pre_attention_mapper
        self.question_to_context_attention = question_to_context_attention
        self.context_to_question_attention = context_to_question_attention
        self.reformulate_by_context = reformulate_by_context
        self.multiply_iteration_probs = multiply_iteration_probs
        self.first_predictor = first_predictor
        self.second_predictor = second_predictor
        self.max_pool = MaxPool(map_layer=None, min_val=VERY_NEGATIVE_NUMBER, regular_reshape=True)
        self.mean_pool = MeanPool()

        if (self.reformulate_by_context and question_to_context_attention is None) or \
                (not self.reformulate_by_context and context_to_question_attention is None):
            raise ValueError("The last attention must be defined")

    def _get_predictions_for(self,
                             is_train,
                             question_embed, question_mask,
                             context_embed, context_mask,
                             answer,
                             question_lm, context_lm, sentence_segments, sentence_mask):
        question_rep, context_rep = question_embed, context_embed
        context1_rep, context2_rep = tf.unstack(context_rep, axis=1, num=2)
        context1_mask, context2_mask = tf.unstack(context_mask, axis=1, num=2)
        context1_sentence_segments, context2_sentence_segments = tf.unstack(sentence_segments, axis=1, num=2)
        context1_sentence_mask, context2_sentence_mask = tf.unstack(sentence_mask, axis=1, num=2)
        q_lm_in, c1_lm_in, c2_lm_in = [], [], []
        if self.use_elmo:
            context1_lm, context2_lm = tf.unstack(context_lm, axis=1, num=2)
            q_lm_in = [question_lm]
            c1_lm_in = [context1_lm]
            c2_lm_in = [context2_lm]
        if self.embed_mapper is not None:
            with tf.variable_scope("map_embed"):
                context1_rep = self.embed_mapper.apply(is_train, context1_rep, context1_mask, *c1_lm_in)
            with tf.variable_scope("map_embed", reuse=True):
                context2_rep = self.embed_mapper.apply(is_train, context2_rep, context2_mask, *c2_lm_in)
                question_rep = self.embed_mapper.apply(is_train, question_rep, question_mask, *q_lm_in)

        with tf.variable_scope("seq_enc"):
            question_enc = self.sequence_encoder.apply(is_train, question_rep, question_mask)
            question_enc = tf.identity(question_enc, name='encode_question')
            tf.add_to_collection(INTERMEDIATE_LAYER_COLLECTION, question_enc)

        def encode_sentences(context, sentence_segs, sentence_mask, rep_name):
            context = self.sentences_encoder.apply(context,
                                                   sentence_segs, sentence_mask)
            if self.sentence_mapper is not None:
                with tf.variable_scope('sentence_mapper'):
                    context = self.sentence_mapper.apply(is_train, context, mask=sentence_mask)
            context = tf.identity(context, name=rep_name)
            tf.add_to_collection(INTERMEDIATE_LAYER_COLLECTION, context)
            return context

        with tf.variable_scope('sentences_enc'):
            context1_sent_rep = encode_sentences(context1_rep, context1_sentence_segments, context1_sentence_mask,
                                                 'encode_context1')
        with tf.variable_scope('sentences_enc', reuse=True):
            context2_sent_rep = encode_sentences(context2_rep, context2_sentence_segments, context2_sentence_mask,
                                                 'encode_context2')

        # First Iteration (same as in the single context model)
        with tf.variable_scope("context1_relevance"):
            c1_q_merged_rep, context1_sentences_logits, context1_pred = \
                merge_weight_predict(is_train=is_train, context_rep=context1_sent_rep, question_rep=question_enc,
                                     context_mask=context1_sentence_mask, merger=self.merger,
                                     post_merger=self.post_merger, max_pool=self.max_pool,
                                     predictor=self.first_predictor, answer=[answer[0]] + answer[2:])

        # Question Reformulation
        with tf.variable_scope("reformulation"):
            if self.reread_mapper is not None:
                question_rep, context_rep = question_embed, context_embed
                context1_rep, _ = tf.unstack(context_rep, axis=1, num=2)
                context1_mask, _ = tf.unstack(context_mask, axis=1, num=2)
                if not isinstance(self.reread_mapper, ElmoWrapper):
                    c1_lm_in, q_lm_in = [], []
                with tf.variable_scope("reread_map_embed"):
                    context1_rep = self.reread_mapper.apply(is_train, context1_rep, context1_mask, *c1_lm_in)
                with tf.variable_scope("reread_map_embed", reuse=True):
                    question_rep = self.reread_mapper.apply(is_train, question_rep, question_mask, *q_lm_in)
            if self.pre_attention_mapper is not None:
                with tf.variable_scope("pre_att"):
                    question_rep = self.pre_attention_mapper.apply(is_train, question_rep, question_mask)
                with tf.variable_scope("pre_att", reuse=True):
                    context1_rep = self.pre_attention_mapper.apply(is_train, context1_rep, context1_mask)
            if not self.reformulate_by_context:
                if self.question_to_context_attention is not None:
                    with tf.variable_scope('q2c'):
                        context1_rep = self.question_to_context_attention.apply(is_train, x=context1_rep,
                                                                                keys=question_rep,
                                                                                memories=question_rep,
                                                                                x_mask=context1_mask,
                                                                                memory_mask=question_mask)
                    if self.pre_attention_mapper is not None:
                        with tf.variable_scope("pre_att", reuse=True):
                            context1_rep = self.pre_attention_mapper.apply(is_train, context1_rep, context1_mask)
                with tf.variable_scope('c2q'):
                    question_rep = self.context_to_question_attention.apply(is_train, x=question_rep, keys=context1_rep,
                                                                            memories=context1_rep, x_mask=question_mask,
                                                                            memory_mask=context1_mask)
                reformulated_q = self.sequence_encoder.apply(is_train, question_rep, question_mask)
            else:
                if self.context_to_question_attention is not None:
                    with tf.variable_scope('c2q'):
                        question_rep = self.context_to_question_attention.apply(is_train, x=question_rep,
                                                                                keys=context1_rep,
                                                                                memories=context1_rep,
                                                                                x_mask=question_mask,
                                                                                memory_mask=context1_mask)
                    if self.pre_attention_mapper is not None:
                        with tf.variable_scope("pre_att", reuse=True):
                            question_rep = self.pre_attention_mapper.apply(is_train, question_rep, question_mask)
                with tf.variable_scope('q2c'):
                    context1_rep = self.question_to_context_attention.apply(is_train, x=context1_rep, keys=question_rep,
                                                                            memories=question_rep, x_mask=context1_mask,
                                                                            memory_mask=question_mask)
                reformulated_q = self.sequence_encoder.apply(is_train, context1_rep, context1_mask)
            reformulated_q = tf.identity(reformulated_q, name='reformulated_question')
            tf.add_to_collection(INTERMEDIATE_LAYER_COLLECTION, reformulated_q)

        # Second Iteration
        with tf.variable_scope("context2_relevance"):
            first_iter_probs = None
            if self.multiply_iteration_probs:
                first_iter_probs = tf.expand_dims(context1_pred.get_probs(), axis=1)
            c2_q_merged_rep, context2_sentences_logits, context2_pred = \
                merge_weight_predict(is_train=is_train, context_rep=context2_sent_rep, question_rep=reformulated_q,
                                     context_mask=context2_sentence_mask, merger=self.merger,
                                     post_merger=self.post_merger, max_pool=self.max_pool,
                                     predictor=self.second_predictor, answer=[answer[1]] + answer[2:],
                                     multiply_probs=first_iter_probs)

        return MultipleBinaryPredictions([context1_pred, context2_pred])

    def __setstate__(self, state):
        if "multiply_iteration_probs" not in state:
            state["multiply_iteration_probs"] = False
        if "reformulate_by_context" not in state:
            state["reformulate_by_context"] = False
        if "second_predictor" not in state:
            state["second_predictor"] = BinaryNullPredictor()
            state["first_predictor"] = BinaryNullPredictor()
        if "reread_mapper" not in state:
            state["reread_mapper"] = None
        if "pre_attention_mapper" not in state:
            state["pre_attention_mapper"] = None
        super().__setstate__(state)


class IterativeContextReReadSimpleScoreModel(MultipleContextModel):
    """
    Calculating the similarities by a simple dot product between question an paragraph representations.
    This is a more neat model which we should check to see if on par with the more complicated one above.
    """

    def __init__(self,
                 encoder: QuestionsAndParagraphsEncoder,
                 word_embed: Optional[WordEmbedder],
                 char_embed: Optional[CharWordEmbedder],
                 embed_mapper: Optional[Union[SequenceMapper, ElmoWrapper]],
                 sequence_encoder: SequenceEncoder,
                 sentences_encoder: SentencesEncoder,
                 sentence_mapper: Optional[SequenceMapper],
                 reread_mapper: Optional[Union[SequenceMapper, ElmoWrapper]],
                 pre_attention_mapper: Optional[SequenceMapper],
                 context_to_question_attention: Optional[AttentionWithPostMapper],
                 question_to_context_attention: Optional[AttentionWithPostMapper],
                 first_predictor: BinaryNullPredictor,
                 second_predictor: BinaryNullPredictor,
                 reformulate_by_context: bool,
                 max_batch_size: Optional[int] = None,
                 elmo_model: Optional[LanguageModel] = None
                 ):
        super().__init__(encoder=encoder, word_embed=word_embed, char_embed=char_embed, max_batch_size=max_batch_size,
                         elmo_model=elmo_model)
        self.embed_mapper = embed_mapper
        self.sequence_encoder = sequence_encoder
        self.sentences_encoder = sentences_encoder
        self.sentence_mapper = sentence_mapper
        self.reread_mapper = reread_mapper
        self.pre_attention_mapper = pre_attention_mapper
        self.question_to_context_attention = question_to_context_attention
        self.context_to_question_attention = context_to_question_attention
        self.reformulate_by_context = reformulate_by_context
        self.first_predictor = first_predictor
        self.second_predictor = second_predictor
        self.max_pool = MaxPool(map_layer=None, min_val=VERY_NEGATIVE_NUMBER, regular_reshape=True)

        if (self.reformulate_by_context and question_to_context_attention is None) or \
                (not self.reformulate_by_context and context_to_question_attention is None):
            raise ValueError("The last attention must be defined")

    def _get_predictions_for(self,
                             is_train,
                             question_embed, question_mask,
                             context_embed, context_mask,
                             answer,
                             question_lm, context_lm, sentence_segments, sentence_mask):
        question_rep, context_rep = question_embed, context_embed
        context1_rep, context2_rep = tf.unstack(context_rep, axis=1, num=2)
        context1_mask, context2_mask = tf.unstack(context_mask, axis=1, num=2)
        context1_sentence_segments, context2_sentence_segments = tf.unstack(sentence_segments, axis=1, num=2)
        context1_sentence_mask, context2_sentence_mask = tf.unstack(sentence_mask, axis=1, num=2)
        q_lm_in, c1_lm_in, c2_lm_in = [], [], []
        if self.use_elmo:
            context1_lm, context2_lm = tf.unstack(context_lm, axis=1, num=2)
            q_lm_in = [question_lm]
            c1_lm_in = [context1_lm]
            c2_lm_in = [context2_lm]
        if self.embed_mapper is not None:
            with tf.variable_scope("map_embed"):
                context1_rep = self.embed_mapper.apply(is_train, context1_rep, context1_mask, *c1_lm_in)
            with tf.variable_scope("map_embed", reuse=True):
                context2_rep = self.embed_mapper.apply(is_train, context2_rep, context2_mask, *c2_lm_in)
                question_rep = self.embed_mapper.apply(is_train, question_rep, question_mask, *q_lm_in)

        with tf.variable_scope("seq_enc"):
            question_enc = self.sequence_encoder.apply(is_train, question_rep, question_mask)
            question_enc = tf.identity(question_enc, name='encode_question')
            tf.add_to_collection(INTERMEDIATE_LAYER_COLLECTION, question_enc)

        def encode_sentences(context, sentence_segs, sentence_mask, rep_name):
            context = self.sentences_encoder.apply(context,
                                                   sentence_segs, sentence_mask)
            if self.sentence_mapper is not None:
                with tf.variable_scope('sentence_mapper'):
                    context = self.sentence_mapper.apply(is_train, context, mask=sentence_mask)
            context = tf.identity(context, name=rep_name)
            tf.add_to_collection(INTERMEDIATE_LAYER_COLLECTION, context)
            return context

        with tf.variable_scope('sentences_enc'):
            context1_sent_rep = encode_sentences(context1_rep, context1_sentence_segments, context1_sentence_mask,
                                                 'encode_context1')
        with tf.variable_scope('sentences_enc', reuse=True):
            context2_sent_rep = encode_sentences(context2_rep, context2_sentence_segments, context2_sentence_mask,
                                                 'encode_context2')

        # First Iteration (same as in the single context model)
        with tf.variable_scope("context1_relevance"):
            sentence_logits = tf.matmul(context1_sent_rep, tf.expand_dims(question_enc, axis=2))
            max_logits = self.max_pool.apply(is_train, sentence_logits, context1_sentence_mask)
            with tf.variable_scope("predictor"):
                context1_pred = self.first_predictor.apply(is_train, max_logits, [answer[0]] + answer[2:])

        # Question Reformulation
        with tf.variable_scope("reformulation"):
            if self.reread_mapper is not None:
                question_rep, context_rep = question_embed, context_embed
                context1_rep, _ = tf.unstack(context_rep, axis=1, num=2)
                context1_mask, _ = tf.unstack(context_mask, axis=1, num=2)
                if not isinstance(self.reread_mapper, ElmoWrapper):
                    c1_lm_in, q_lm_in = [], []
                with tf.variable_scope("reread_map_embed"):
                    context1_rep = self.reread_mapper.apply(is_train, context1_rep, context1_mask, *c1_lm_in)
                with tf.variable_scope("reread_map_embed", reuse=True):
                    question_rep = self.reread_mapper.apply(is_train, question_rep, question_mask, *q_lm_in)
            if self.pre_attention_mapper is not None:
                with tf.variable_scope("pre_att"):
                    question_rep = self.pre_attention_mapper.apply(is_train, question_rep, question_mask)
                with tf.variable_scope("pre_att", reuse=True):
                    context1_rep = self.pre_attention_mapper.apply(is_train, context1_rep, context1_mask)
            if not self.reformulate_by_context:
                if self.question_to_context_attention is not None:
                    with tf.variable_scope('q2c'):
                        context1_rep = self.question_to_context_attention.apply(is_train, x=context1_rep,
                                                                                keys=question_rep,
                                                                                memories=question_rep,
                                                                                x_mask=context1_mask,
                                                                                memory_mask=question_mask)
                    if self.pre_attention_mapper is not None:
                        with tf.variable_scope("pre_att", reuse=True):
                            context1_rep = self.pre_attention_mapper.apply(is_train, context1_rep, context1_mask)
                with tf.variable_scope('c2q'):
                    question_rep = self.context_to_question_attention.apply(is_train, x=question_rep, keys=context1_rep,
                                                                            memories=context1_rep, x_mask=question_mask,
                                                                            memory_mask=context1_mask)
                reformulated_q = self.sequence_encoder.apply(is_train, question_rep, question_mask)
            else:
                if self.context_to_question_attention is not None:
                    with tf.variable_scope('c2q'):
                        question_rep = self.context_to_question_attention.apply(is_train, x=question_rep,
                                                                                keys=context1_rep,
                                                                                memories=context1_rep,
                                                                                x_mask=question_mask,
                                                                                memory_mask=context1_mask)
                    if self.pre_attention_mapper is not None:
                        with tf.variable_scope("pre_att", reuse=True):
                            question_rep = self.pre_attention_mapper.apply(is_train, question_rep, question_mask)
                with tf.variable_scope('q2c'):
                    context1_rep = self.question_to_context_attention.apply(is_train, x=context1_rep, keys=question_rep,
                                                                            memories=question_rep, x_mask=context1_mask,
                                                                            memory_mask=question_mask)
                reformulated_q = self.sequence_encoder.apply(is_train, context1_rep, context1_mask)
            reformulated_q = tf.identity(reformulated_q, name='reformulated_question')
            tf.add_to_collection(INTERMEDIATE_LAYER_COLLECTION, reformulated_q)

        # Second Iteration
        with tf.variable_scope("context2_relevance"):
            sentence_logits = tf.matmul(context2_sent_rep, tf.expand_dims(reformulated_q, axis=2))
            max_logits = self.max_pool.apply(is_train, sentence_logits, context2_sentence_mask)
            with tf.variable_scope("predictor"):
                context2_pred = self.second_predictor.apply(is_train, max_logits, [answer[1]] + answer[2:])

        return MultipleBinaryPredictions([context1_pred, context2_pred])


class IterativeContextReReadMergeModel(MultipleContextModel):
    """
    Like the reread model, but encode the paragraph and the question, and then merge representations for the
        reformulation.
    """

    def __init__(self,
                 encoder: QuestionsAndParagraphsEncoder,
                 word_embed: Optional[WordEmbedder],
                 char_embed: Optional[CharWordEmbedder],
                 embed_mapper: Optional[Union[SequenceMapper, ElmoWrapper]],
                 sequence_encoder: SequenceEncoder,
                 sentences_encoder: SentencesEncoder,
                 sentence_mapper: Optional[SequenceMapper],
                 merger: FixedMergeLayer,
                 post_merger: Optional[Mapper],
                 context_to_question_attention: Optional[AttentionWithPostMapper],
                 question_to_context_attention: Optional[AttentionWithPostMapper],
                 reread_merger: MergeLayer,
                 multiply_iteration_probs: bool,
                 max_batch_size: Optional[int] = None,
                 elmo_model: Optional[LanguageModel] = None
                 ):
        super().__init__(encoder=encoder, word_embed=word_embed, char_embed=char_embed, max_batch_size=max_batch_size,
                         elmo_model=elmo_model)
        self.embed_mapper = embed_mapper
        self.sequence_encoder = sequence_encoder
        self.sentences_encoder = sentences_encoder
        self.sentence_mapper = sentence_mapper
        self.merger = merger
        self.post_merger = post_merger
        self.question_to_context_attention = question_to_context_attention
        self.context_to_question_attention = context_to_question_attention
        self.reread_merger = reread_merger
        self.multiply_iteration_probs = multiply_iteration_probs
        self.predictor = BinaryNullPredictor()
        self.max_pool = MaxPool(map_layer=None, min_val=VERY_NEGATIVE_NUMBER, regular_reshape=True)
        self.mean_pool = MeanPool()

    def _get_predictions_for(self,
                             is_train,
                             question_embed, question_mask,
                             context_embed, context_mask,
                             answer,
                             question_lm, context_lm, sentence_segments, sentence_mask):
        question_rep, context_rep = question_embed, context_embed
        context1_rep, context2_rep = tf.unstack(context_rep, axis=1, num=2)
        context1_mask, context2_mask = tf.unstack(context_mask, axis=1, num=2)
        context1_sentence_segments, context2_sentence_segments = tf.unstack(sentence_segments, axis=1, num=2)
        context1_sentence_mask, context2_sentence_mask = tf.unstack(sentence_mask, axis=1, num=2)
        q_lm_in, c1_lm_in, c2_lm_in = [], [], []
        if self.use_elmo:
            context1_lm, context2_lm = tf.unstack(context_lm, axis=1, num=2)
            q_lm_in = [question_lm]
            c1_lm_in = [context1_lm]
            c2_lm_in = [context2_lm]
        if self.embed_mapper is not None:
            with tf.variable_scope("map_embed"):
                context1_rep = self.embed_mapper.apply(is_train, context1_rep, context1_mask, *c1_lm_in)
            with tf.variable_scope("map_embed", reuse=True):
                context2_rep = self.embed_mapper.apply(is_train, context2_rep, context2_mask, *c2_lm_in)
                question_rep = self.embed_mapper.apply(is_train, question_rep, question_mask, *q_lm_in)

        with tf.variable_scope("seq_enc"):
            question_enc = self.sequence_encoder.apply(is_train, question_rep, question_mask)
            question_enc = tf.identity(question_enc, name='encode_question')
            tf.add_to_collection(INTERMEDIATE_LAYER_COLLECTION, question_enc)

        def encode_sentences(context, sentence_segs, sentence_mask, rep_name):
            context = self.sentences_encoder.apply(context,
                                                   sentence_segs, sentence_mask)
            if self.sentence_mapper is not None:
                with tf.variable_scope('sentence_mapper'):
                    context = self.sentence_mapper.apply(is_train, context, mask=sentence_mask)
            context = tf.identity(context, name=rep_name)
            tf.add_to_collection(INTERMEDIATE_LAYER_COLLECTION, context)
            return context

        with tf.variable_scope('sentences_enc'):
            context1_sent_rep = encode_sentences(context1_rep, context1_sentence_segments, context1_sentence_mask,
                                                 'encode_context1')
        with tf.variable_scope('sentences_enc', reuse=True):
            context2_sent_rep = encode_sentences(context2_rep, context2_sentence_segments, context2_sentence_mask,
                                                 'encode_context2')

        # First Iteration (same as in the single context model)
        with tf.variable_scope("context1_relevance"):
            c1_q_merged_rep, context1_sentences_logits, context1_pred = \
                merge_weight_predict(is_train=is_train, context_rep=context1_sent_rep, question_rep=question_enc,
                                     context_mask=context1_sentence_mask, merger=self.merger,
                                     post_merger=self.post_merger, max_pool=self.max_pool,
                                     predictor=self.predictor, answer=[answer[0]])

        # Question Reformulation
        with tf.variable_scope("reformulation"):
            with tf.variable_scope('c2q'):
                question_rep = self.context_to_question_attention.apply(is_train, x=question_rep, keys=context1_rep,
                                                                        memories=context1_rep, x_mask=question_mask,
                                                                        memory_mask=context1_mask)
                reread_q_enc = self.sequence_encoder.apply(is_train, question_rep, question_mask)
            with tf.variable_scope('q2c'):
                context1_rep = self.question_to_context_attention.apply(is_train, x=context1_rep, keys=question_rep,
                                                                        memories=question_rep, x_mask=context1_mask,
                                                                        memory_mask=question_mask)
                reread_c1_enc = self.sequence_encoder.apply(is_train, context1_rep, context1_mask)
            with tf.variable_scope('reread_merge'):
                reformulated_q = self.reread_merger.apply(is_train, reread_q_enc, reread_c1_enc)
                reformulated_q = fully_connected(reformulated_q, c1_q_merged_rep.shape.as_list()[-1],
                                                 use_bias=True,
                                                 activation=get_keras_activation('relu'),
                                                 kernel_initializer=get_keras_initialization('glorot_uniform'))
            reformulated_q = tf.identity(reformulated_q, name='reformulated_question')
            tf.add_to_collection(INTERMEDIATE_LAYER_COLLECTION, reformulated_q)

        # Second Iteration
        with tf.variable_scope("context2_relevance"):
            first_iter_probs = None
            if self.multiply_iteration_probs:
                first_iter_probs = tf.expand_dims(context1_pred.get_probs(), axis=1)
            c2_q_merged_rep, context2_sentences_logits, context2_pred = \
                merge_weight_predict(is_train=is_train, context_rep=context2_sent_rep, question_rep=reformulated_q,
                                     context_mask=context2_sentence_mask, merger=self.merger,
                                     post_merger=self.post_merger, max_pool=self.max_pool,
                                     predictor=self.predictor, answer=[answer[1]], multiply_probs=first_iter_probs)

        return MultipleBinaryPredictions([context1_pred, context2_pred])
