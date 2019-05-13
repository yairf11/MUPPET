from typing import Optional, Union
import tensorflow as tf
from tensorflow.python.layers.core import fully_connected

from hotpot.elmo.elmo import ElmoWrapper
from hotpot.elmo.lm_model import LanguageModel
from hotpot.encoder import QuestionsAndParagraphsEncoder
from hotpot.models.multiple_context_models import MultipleContextModel, INTERMEDIATE_LAYER_COLLECTION
from hotpot.nn.embedder import WordEmbedder, CharWordEmbedder
from hotpot.nn.layers import SequenceMapper, SequenceEncoder, MergeLayer, Mapper, SequenceMultiEncoder, \
    MultipleMergeEncode, WeightLayer, FixedMergeLayer, get_keras_initialization, MaxPool, MeanPool
from hotpot.nn.ops import VERY_NEGATIVE_NUMBER
from hotpot.nn.relevance_prediction import BinaryFixedPredictor, BinaryWeightedMultipleFixedPredictor, \
    BinaryNullPredictor
from hotpot.nn.sentence_layers import SentencesEncoder


class SingleContextMultipleEncodingModel(MultipleContextModel):
    """
    Model for a question with a single paragraph, basically expands the basic model by creating multiple fixed size
     representations of the context and the question, and weights each representation depending on the question.
    components are as follows:
    1. embeds each sequence separately
    2. encodes the context and question to multiple fixed size representations
    3. weights each representation, and performs a applies layer on each representation pair
    4. predict by combining all representations
    """

    def __init__(self,
                 encoder: QuestionsAndParagraphsEncoder,
                 word_embed: Optional[WordEmbedder],
                 char_embed: Optional[CharWordEmbedder],
                 embed_mapper: Optional[Union[SequenceMapper, ElmoWrapper]],
                 sequence_multi_encoder: SequenceMultiEncoder,
                 merger_encoder: MultipleMergeEncode,
                 post_merger: Optional[Mapper],
                 predictor: BinaryFixedPredictor,
                 max_batch_size: Optional[int] = None,
                 elmo_model: Optional[LanguageModel] = None):
        super().__init__(encoder=encoder, word_embed=word_embed, char_embed=char_embed, max_batch_size=max_batch_size,
                         elmo_model=elmo_model)
        self.embed_mapper = embed_mapper
        self.sequence_multi_encoder = sequence_multi_encoder
        self.merger_encoder = merger_encoder
        self.post_merger = post_merger
        self.predictor = predictor

    def _get_predictions_for(self,
                             is_train,
                             question_embed, question_mask,
                             context_embed, context_mask,
                             answer,
                             question_lm=None, context_lm=None, sentence_segments=None, sentence_mask=None):
        question_rep, context_rep = question_embed, context_embed
        context1_rep, = tf.unstack(context_rep, axis=1, num=1)
        context1_mask, = tf.unstack(context_mask, axis=1, num=1)
        q_lm_in, c1_lm_in = [], []
        if self.use_elmo:
            context1_lm, = tf.unstack(context_lm, axis=1, num=1)
            q_lm_in = [question_lm]
            c1_lm_in = [context1_lm]
        if self.embed_mapper is not None:
            with tf.variable_scope("map_embed"):
                context1_rep = self.embed_mapper.apply(is_train, context1_rep, context1_mask, *c1_lm_in)
            with tf.variable_scope("map_embed", reuse=True):
                question_rep = self.embed_mapper.apply(is_train, question_rep, question_mask, *q_lm_in)

        with tf.variable_scope("seq_multi_enc"):
            context1_rep = self.sequence_multi_encoder.apply(is_train, context1_rep, context1_mask)
            context1_rep = tf.identity(context1_rep, name='multi_encode_context')
            tf.add_to_collection(INTERMEDIATE_LAYER_COLLECTION, context1_rep)
        with tf.variable_scope("seq_multi_enc", reuse=True):
            question_rep = self.sequence_multi_encoder.apply(is_train, question_rep, question_mask)

        with tf.variable_scope("merger_encoder"):
            merged_rep = self.merger_encoder.apply(is_train, question_rep, context1_rep)

        if self.post_merger is not None:
            with tf.variable_scope("post_merger"):
                merged_rep = self.post_merger.apply(is_train, merged_rep, mask=None)

        with tf.variable_scope("predictor"):
            return self.predictor.apply(is_train, merged_rep, answer)


class SingleContextMultipleEncodingWeightedSoftmaxModel(MultipleContextModel):
    """
    A model very much like SingleContextMultipleEncodingModel above, but for a little but important difference:
        The weighting of the encoding is done after "predicting" with each one of the encodings and performing a
        softmax between each encodings logits. Then a weighted sum is calculated, with gives new probabilities
        for the classes, on which a cross-entropy is applied.
    """

    def __init__(self,
                 encoder: QuestionsAndParagraphsEncoder,
                 word_embed: Optional[WordEmbedder],
                 char_embed: Optional[CharWordEmbedder],
                 embed_mapper: Optional[Union[SequenceMapper, ElmoWrapper]],
                 sequence_multi_encoder: SequenceMultiEncoder,
                 weight_layer: WeightLayer,
                 merger: MergeLayer,
                 post_merger: Optional[Mapper],
                 predictor: BinaryWeightedMultipleFixedPredictor,
                 max_batch_size: Optional[int] = None,
                 elmo_model: Optional[LanguageModel] = None):
        super().__init__(encoder=encoder, word_embed=word_embed, char_embed=char_embed, max_batch_size=max_batch_size,
                         elmo_model=elmo_model)
        self.embed_mapper = embed_mapper
        self.sequence_multi_encoder = sequence_multi_encoder
        self.weight_layer = weight_layer
        self.merger = merger
        self.post_merger = post_merger
        self.predictor = predictor

    def _get_predictions_for(self,
                             is_train,
                             question_embed, question_mask,
                             context_embed, context_mask,
                             answer,
                             question_lm=None, context_lm=None, sentence_segments=None, sentence_mask=None):
        question_rep, context_rep = question_embed, context_embed
        context1_rep, = tf.unstack(context_rep, axis=1, num=1)
        context1_mask, = tf.unstack(context_mask, axis=1, num=1)
        q_lm_in, c1_lm_in = [], []
        if self.use_elmo:
            context1_lm, = tf.unstack(context_lm, axis=1, num=1)
            q_lm_in = [question_lm]
            c1_lm_in = [context1_lm]
        if self.embed_mapper is not None:
            with tf.variable_scope("map_embed"):
                context1_rep = self.embed_mapper.apply(is_train, context1_rep, context1_mask, *c1_lm_in)
            with tf.variable_scope("map_embed", reuse=True):
                question_rep = self.embed_mapper.apply(is_train, question_rep, question_mask, *q_lm_in)

        with tf.variable_scope("seq_multi_enc"):
            context1_rep = self.sequence_multi_encoder.apply(is_train, context1_rep, context1_mask)
            context1_rep = tf.identity(context1_rep, name='multi_encode_context')
            tf.add_to_collection(INTERMEDIATE_LAYER_COLLECTION, context1_rep)
        with tf.variable_scope("seq_multi_enc", reuse=True):
            question_rep = self.sequence_multi_encoder.apply(is_train, question_rep, question_mask)

        with tf.variable_scope('weight_layer'):
            encoding_weights = self.weight_layer.apply(is_train, question_rep, mask=None)
            encoding_weights = tf.identity(encoding_weights, name='encoding_weights')
            tf.add_to_collection(INTERMEDIATE_LAYER_COLLECTION, encoding_weights)

        with tf.variable_scope("merger"):
            merged_rep = self.merger.apply(is_train, question_rep, context1_rep)

        if self.post_merger is not None:
            with tf.variable_scope("post_merger"):
                merged_rep = self.post_merger.apply(is_train, merged_rep, mask=None)

        with tf.variable_scope("predictor"):
            return self.predictor.apply(is_train, merged_rep, weights=encoding_weights, answer=answer)


class SingleContextMaxSentenceModel(MultipleContextModel):
    """
    Model for a question and a single paragraph which takes into account the sentences.
    This model first creates an encoding for each sentence, and then performs a fully connected layer on the
        encodings to get each sentence's prediction. It then gets the maximum value and predicts with it.
    """

    def __init__(self,
                 encoder: QuestionsAndParagraphsEncoder,
                 word_embed: Optional[WordEmbedder],
                 char_embed: Optional[CharWordEmbedder],
                 embed_mapper: Optional[Union[SequenceMapper, ElmoWrapper]],
                 sequence_encoder: SequenceEncoder,
                 sentences_encoder: SentencesEncoder,
                 merger: FixedMergeLayer,
                 post_merger: Optional[Mapper],
                 max_batch_size: Optional[int] = None,
                 elmo_model: Optional[LanguageModel] = None
                 ):
        super().__init__(encoder=encoder, word_embed=word_embed, char_embed=char_embed, max_batch_size=max_batch_size,
                         elmo_model=elmo_model)
        self.embed_mapper = embed_mapper
        self.sequence_encoder = sequence_encoder
        self.sentences_encoder = sentences_encoder
        self.merger = merger
        self.post_merger = post_merger
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
        context1_rep, = tf.unstack(context_rep, axis=1, num=1)
        context1_mask, = tf.unstack(context_mask, axis=1, num=1)
        sentence_segments, = tf.unstack(sentence_segments, axis=1, num=1)
        sentence_mask, = tf.unstack(sentence_mask, axis=1, num=1)
        q_lm_in, c1_lm_in = [], []
        if self.use_elmo:
            context1_lm, = tf.unstack(context_lm, axis=1, num=1)
            q_lm_in = [question_lm]
            c1_lm_in = [context1_lm]
        if self.embed_mapper is not None:
            with tf.variable_scope("map_embed"):
                context1_rep = self.embed_mapper.apply(is_train, context1_rep, context1_mask, *c1_lm_in)
            with tf.variable_scope("map_embed", reuse=True):
                question_rep = self.embed_mapper.apply(is_train, question_rep, question_mask, *q_lm_in)

        with tf.variable_scope("seq_enc"):
            question_rep = self.sequence_encoder.apply(is_train, question_rep, question_mask)

        with tf.variable_scope("sentences_enc"):
            context1_rep = self.sentences_encoder.apply(context1_rep, sentence_segments, sentence_mask)
            context1_rep = tf.identity(context1_rep, name='encode_context')
            tf.add_to_collection(INTERMEDIATE_LAYER_COLLECTION, context1_rep)

        with tf.variable_scope("merger"):
            merged_rep = self.merger.apply(is_train, tensor=context1_rep, fixed_tensor=question_rep, mask=sentence_mask)

        if self.post_merger is not None:
            with tf.variable_scope("post_merger"):
                merged_rep = self.post_merger.apply(is_train, merged_rep, mask=sentence_mask)

        with tf.variable_scope("sentence_level_predictions"):
            sentences_logits = fully_connected(merged_rep, 1,
                                               use_bias=True,
                                               activation=None,
                                               kernel_initializer=get_keras_initialization('glorot_uniform'))
            max_logits = self.max_pool.apply(is_train, sentences_logits, sentence_mask)

        with tf.variable_scope("predictor"):
            return self.predictor.apply(is_train, max_logits, answer)

    def __setstate__(self, state):
        if "post_merger" not in state:
            state["post_merger"] = None
        super().__setstate__(state)
