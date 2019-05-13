from typing import Union, Optional
import tensorflow as tf
from tensorflow.python.layers.core import fully_connected

from hotpot.elmo.elmo import ElmoWrapper
from hotpot.elmo.lm_model import LanguageModel
from hotpot.encoder import QuestionsAndParagraphsEncoder
from hotpot.models.multiple_context_models import MultipleContextModel
from hotpot.nn.embedder import WordEmbedder, CharWordEmbedder
from hotpot.nn.layers import SequenceMapper, SequenceBiMapper, AttentionMapper, SequencePredictionLayer, \
    AttentionPredictionLayer, SequenceEncoder, get_keras_initialization
from hotpot.nn.sentence_layers import SentencesEncoder
from hotpot.nn.span_prediction import BoundaryAndYesNoPrediction, SpanFromBoundsOrYesNoPredictor


class AttentionQA(MultipleContextModel):
    """
    Model that encodes the question and context, then applies an attention mechanism
    between the two to produce a query-aware context representation, which is used to make a prediction.
    """

    def __init__(self, encoder: QuestionsAndParagraphsEncoder,
                 word_embed: Optional[WordEmbedder],
                 char_embed: Optional[CharWordEmbedder],
                 embed_mapper: Optional[Union[SequenceMapper, ElmoWrapper]],
                 question_mapper: Optional[SequenceMapper],
                 context_mapper: Optional[SequenceMapper],
                 memory_builder: SequenceBiMapper,
                 attention: AttentionMapper,
                 match_encoder: SequenceMapper,
                 predictor: Union[SequencePredictionLayer, AttentionPredictionLayer],
                 max_batch_size: Optional[int] = None,
                 elmo_model: Optional[LanguageModel] = None):
        super().__init__(encoder=encoder, word_embed=word_embed, char_embed=char_embed, max_batch_size=max_batch_size,
                         elmo_model=elmo_model)
        self.embed_mapper = embed_mapper
        self.question_mapper = question_mapper
        self.context_mapper = context_mapper
        self.memory_builder = memory_builder
        self.attention = attention
        self.match_encoder = match_encoder
        self.predictor = predictor

    def _get_predictions_for(self,
                             is_train,
                             question_embed, question_mask,
                             context_embed, context_mask,
                             answer,
                             question_lm=None, context_lm=None, sentence_segments=None, sentence_mask=None):
        question_rep, context_rep = question_embed, context_embed
        context_rep, = tf.unstack(context_rep, axis=1, num=1)
        context_mask, = tf.unstack(context_mask, axis=1, num=1)
        q_lm_in, c_lm_in = [], []
        if self.use_elmo:
            context_lm, = tf.unstack(context_lm, axis=1, num=1)
            q_lm_in = [question_lm]
            c_lm_in = [context_lm]
        if self.embed_mapper is not None:
            with tf.variable_scope("map_embed"):
                context_rep = self.embed_mapper.apply(is_train, context_rep, context_mask, *c_lm_in)
            with tf.variable_scope("map_embed", reuse=True):
                question_rep = self.embed_mapper.apply(is_train, question_rep, question_mask, *q_lm_in)

        if self.question_mapper is not None:
            with tf.variable_scope("map_question"):
                question_rep = self.question_mapper.apply(is_train, question_rep, question_mask)

        if self.context_mapper is not None:
            with tf.variable_scope("map_context"):
                context_rep = self.context_mapper.apply(is_train, context_rep, context_mask)

        with tf.variable_scope("buid_memories"):
            keys, memories = self.memory_builder.apply(is_train, question_rep, question_mask)

        with tf.variable_scope("apply_attention"):
            context_rep = self.attention.apply(is_train, context_rep, keys, memories, context_mask, question_mask)

        if self.match_encoder is not None:
            with tf.variable_scope("process_attention"):
                context_rep = self.match_encoder.apply(is_train, context_rep, context_mask)

        with tf.variable_scope("predict"):
            if isinstance(self.predictor, AttentionPredictionLayer):
                return self.predictor.apply(is_train, context_rep, question_rep, answer, context_mask, question_mask)
            else:
                return self.predictor.apply(is_train, context_rep, answer, context_mask)


class AttentionQAWithYesNo(MultipleContextModel):
    """
    Adds yes/no answer option. The decision on whether to produce a yes/no answer is based solely on the question.
    """

    def __init__(self, encoder: QuestionsAndParagraphsEncoder,
                 word_embed: Optional[WordEmbedder],
                 char_embed: Optional[CharWordEmbedder],
                 embed_mapper: Optional[Union[SequenceMapper, ElmoWrapper]],
                 question_mapper: Optional[SequenceMapper],
                 context_mapper: Optional[SequenceMapper],
                 memory_builder: SequenceBiMapper,
                 attention: AttentionMapper,
                 match_encoder: SequenceMapper,
                 yes_no_question_encoder: SequenceEncoder,
                 yes_no_context_encoder: SequenceEncoder,
                 predictor: SequencePredictionLayer,
                 max_batch_size: Optional[int] = None,
                 elmo_model: Optional[LanguageModel] = None):
        super().__init__(encoder=encoder, word_embed=word_embed, char_embed=char_embed, max_batch_size=max_batch_size,
                         elmo_model=elmo_model)
        self.embed_mapper = embed_mapper
        self.question_mapper = question_mapper
        self.context_mapper = context_mapper
        self.memory_builder = memory_builder
        self.attention = attention
        self.match_encoder = match_encoder
        self.yes_no_question_encoder = yes_no_question_encoder
        self.yes_no_context_encoder = yes_no_context_encoder
        self.predictor = predictor

    def _get_predictions_for(self,
                             is_train,
                             question_embed, question_mask,
                             context_embed, context_mask,
                             answer,
                             question_lm=None, context_lm=None,
                             sentence_segments=None, sentence_mask=None):
        question_rep, context_rep = question_embed, context_embed
        context_rep, = tf.unstack(context_rep, axis=1, num=1)
        context_mask, = tf.unstack(context_mask, axis=1, num=1)
        q_lm_in, c_lm_in = [], []
        if self.use_elmo:
            context_lm, = tf.unstack(context_lm, axis=1, num=1)
            q_lm_in = [question_lm]
            c_lm_in = [context_lm]
        if self.embed_mapper is not None:
            with tf.variable_scope("map_embed"):
                context_rep = self.embed_mapper.apply(is_train, context_rep, context_mask, *c_lm_in)
            with tf.variable_scope("map_embed", reuse=True):
                question_rep = self.embed_mapper.apply(is_train, question_rep, question_mask, *q_lm_in)

        with tf.variable_scope('yes_no_question_prediction'):
            yes_no_q_enc = self.yes_no_question_encoder.apply(is_train, question_rep, question_mask)
            yes_no_choice_logits = fully_connected(yes_no_q_enc, 2,
                                                   use_bias=True,
                                                   activation=None,
                                                   kernel_initializer=get_keras_initialization('glorot_uniform'),
                                                   name='yes_no_choice')

        if self.question_mapper is not None:
            with tf.variable_scope("map_question"):
                question_rep = self.question_mapper.apply(is_train, question_rep, question_mask)

        if self.context_mapper is not None:
            with tf.variable_scope("map_context"):
                context_rep = self.context_mapper.apply(is_train, context_rep, context_mask)

        with tf.variable_scope("buid_memories"):
            keys, memories = self.memory_builder.apply(is_train, question_rep, question_mask)

        with tf.variable_scope("apply_attention"):
            context_rep = self.attention.apply(is_train, context_rep, keys, memories, context_mask, question_mask)

        if self.match_encoder is not None:
            with tf.variable_scope("process_attention"):
                context_rep = self.match_encoder.apply(is_train, context_rep, context_mask)

        with tf.variable_scope('yes_no_answer_prediction'):
            yes_no_c_enc = self.yes_no_context_encoder.apply(is_train, context_rep, context_mask)
            yes_no_answer_logits = fully_connected(yes_no_c_enc, 2,
                                                   use_bias=True,
                                                   activation=None,
                                                   kernel_initializer=get_keras_initialization('glorot_uniform'),
                                                   name='yes_no_answer')

        with tf.variable_scope("predict"):
            return self.predictor.apply(is_train, context_rep, answer, context_mask,
                                        yes_no_choice_logits=yes_no_choice_logits,
                                        yes_no_answer_logits=yes_no_answer_logits)


class AttentionQAFullHotpot(MultipleContextModel):
    """
    Full hotpot-compatible QA model (yes/no, supporting facts)
    """

    def __init__(self, encoder: QuestionsAndParagraphsEncoder,
                 word_embed: Optional[WordEmbedder],
                 char_embed: Optional[CharWordEmbedder],
                 embed_mapper: Optional[Union[SequenceMapper, ElmoWrapper]],
                 question_mapper: Optional[SequenceMapper],
                 context_mapper: Optional[SequenceMapper],
                 memory_builder: SequenceBiMapper,
                 attention: AttentionMapper,
                 match_encoder: SequenceMapper,
                 yes_no_question_encoder: SequenceEncoder,
                 yes_no_context_encoder: SequenceEncoder,
                 pre_sp_mapper: Optional[SequenceMapper],
                 sentences_encoder: SentencesEncoder,
                 sentence_mapper: Optional[SequenceMapper],
                 predictor: SequencePredictionLayer,
                 max_batch_size: Optional[int] = None,
                 elmo_model: Optional[LanguageModel] = None):
        super().__init__(encoder=encoder, word_embed=word_embed, char_embed=char_embed, max_batch_size=max_batch_size,
                         elmo_model=elmo_model)
        self.embed_mapper = embed_mapper
        self.question_mapper = question_mapper
        self.context_mapper = context_mapper
        self.memory_builder = memory_builder
        self.attention = attention
        self.match_encoder = match_encoder
        self.yes_no_question_encoder = yes_no_question_encoder
        self.yes_no_context_encoder = yes_no_context_encoder
        self.pre_sp_mapper = pre_sp_mapper
        self.sentences_encoder = sentences_encoder
        self.sentence_mapper = sentence_mapper
        self.predictor = predictor

    def _get_predictions_for(self,
                             is_train,
                             question_embed, question_mask,
                             context_embed, context_mask,
                             answer,
                             question_lm, context_lm,
                             sentence_segments, sentence_mask):
        question_rep, context_rep = question_embed, context_embed
        context_rep, = tf.unstack(context_rep, axis=1, num=1)
        context_mask, = tf.unstack(context_mask, axis=1, num=1)
        context_sentence_segments, = tf.unstack(sentence_segments, axis=1, num=1)
        context_sentence_mask, = tf.unstack(sentence_mask, axis=1, num=1)
        q_lm_in, c_lm_in = [], []
        if self.use_elmo:
            context_lm, = tf.unstack(context_lm, axis=1, num=1)
            q_lm_in = [question_lm]
            c_lm_in = [context_lm]
        if self.embed_mapper is not None:
            with tf.variable_scope("map_embed"):
                context_rep = self.embed_mapper.apply(is_train, context_rep, context_mask, *c_lm_in)
            with tf.variable_scope("map_embed", reuse=True):
                question_rep = self.embed_mapper.apply(is_train, question_rep, question_mask, *q_lm_in)

        with tf.variable_scope('yes_no_question_prediction'):
            yes_no_q_enc = self.yes_no_question_encoder.apply(is_train, question_rep, question_mask)
            yes_no_choice_logits = fully_connected(yes_no_q_enc, 2,
                                                   use_bias=True,
                                                   activation=None,
                                                   kernel_initializer=get_keras_initialization('glorot_uniform'),
                                                   name='yes_no_choice')

        if self.question_mapper is not None:
            with tf.variable_scope("map_question"):
                question_rep = self.question_mapper.apply(is_train, question_rep, question_mask)

        if self.context_mapper is not None:
            with tf.variable_scope("map_context"):
                context_rep = self.context_mapper.apply(is_train, context_rep, context_mask)

        with tf.variable_scope("buid_memories"):
            keys, memories = self.memory_builder.apply(is_train, question_rep, question_mask)

        with tf.variable_scope("apply_attention"):
            context_rep = self.attention.apply(is_train, context_rep, keys, memories, context_mask, question_mask)

        if self.match_encoder is not None:
            with tf.variable_scope("process_attention"):
                context_rep = self.match_encoder.apply(is_train, context_rep, context_mask)

        with tf.variable_scope('yes_no_answer_prediction'):
            yes_no_c_enc = self.yes_no_context_encoder.apply(is_train, context_rep, context_mask)
            yes_no_answer_logits = fully_connected(yes_no_c_enc, 2,
                                                   use_bias=True,
                                                   activation=None,
                                                   kernel_initializer=get_keras_initialization('glorot_uniform'),
                                                   name='yes_no_answer')

        with tf.variable_scope('supporting_fact_prediction'):
            pre_context_sents = context_rep
            if self.pre_sp_mapper is not None:
                with tf.variable_scope('pre_sp_mapper'):
                    pre_context_sents = self.pre_sp_mapper.apply(is_train, pre_context_sents, context_mask)
            context_sents = self.sentences_encoder.apply(pre_context_sents, context_sentence_segments,
                                                         context_sentence_mask)
            context_sents = tf.identity(context_sents, name='debug')
            if self.sentence_mapper is not None:
                with tf.variable_scope('sentence_mapper'):
                    context_sents = self.sentence_mapper.apply(is_train, context_sents, mask=context_sentence_mask)
            sentences_logits = fully_connected(context_sents, 1,
                                               use_bias=True,
                                               activation=None,
                                               kernel_initializer=get_keras_initialization('glorot_uniform'),
                                               name='supporting_fact_fc')

        with tf.variable_scope("predict"):
            return self.predictor.apply(is_train, context_rep, answer, context_mask,
                                        yes_no_choice_logits=yes_no_choice_logits,
                                        yes_no_answer_logits=yes_no_answer_logits,
                                        sentence_logits=tf.squeeze(sentences_logits, axis=[2]),
                                        sentence_mask=context_sentence_mask)

    def __setstate__(self, state):
        if "pre_sp_mapper" not in state:
            state["pre_sp_mapper"] = None
        super().__setstate__(state)
