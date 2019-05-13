from typing import List

import tensorflow as tf
from tensorflow.contrib.layers import fully_connected

from hotpot.model import Prediction
from hotpot.nn.layers import SequencePredictionLayer, SequenceEncoder, FixedPredictionLayer, WeightedPredictionLayer, \
    get_keras_initialization


class BinaryPrediction(Prediction):
    """ Simple binary (0/1) prediction """

    def __init__(self, logits, sigmoid=False):
        self.logits = logits  # (batch_size, 2), unnormalized logits, or (batch_size, )
        self.sigmoid = sigmoid

    def get_probs(self):
        if self.sigmoid:
            return tf.sigmoid(self.logits)
        return tf.nn.softmax(self.logits, axis=1)

    def get_predictions(self):
        if self.sigmoid:
            return tf.cast(self.logits >= 0, dtype=tf.int32)
        return tf.argmax(self.logits, axis=1)


class MultipleBinaryPredictions(Prediction):
    """ Mupltile binary predictions, used for the iterative relevance models """
    def __init__(self, predictions: List[BinaryPrediction]):
        self.predictions = predictions

    def get_probs(self, pred_idx):
        return self.predictions[pred_idx].get_probs()

    def get_predictions(self, pred_idx):
        return self.predictions[pred_idx].get_predictions()


class BinaryFixedPredictor(FixedPredictionLayer):
    """
    Final prediction layer that has an output dimension of 2.
    """
    def __init__(self, fc_init='glorot_uniform', pos_weight=None, sigmoid=False):
        self.fc_init = fc_init
        self.pos_weight = pos_weight
        self.sigmoid = sigmoid
        if sigmoid and pos_weight is not None:
            raise ValueError("Not supporting sigmoid+pos_weight yet")

    def apply(self, is_train, x, answer: List):
        if not self.sigmoid:
            with tf.variable_scope("compute_logits"):
                logits = fully_connected(x, 2, activation_fn=None, weights_initializer=self.fc_init)

            cross_ent = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=answer[0], logits=logits)
            if self.pos_weight is not None:
                cross_ent = cross_ent * tf.to_float((answer[0] * (self.pos_weight-1)) + 1)
        else:
            with tf.variable_scope("compute_logits"):
                logits = fully_connected(x, 1, activation_fn=None, weights_initializer=self.fc_init)
            logits = tf.squeeze(logits, axis=1, name='logits')
            cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.to_float(answer[0]), logits=logits)
        loss = tf.reduce_mean(cross_ent)
        tf.add_to_collection(tf.GraphKeys.LOSSES, loss)
        return BinaryPrediction(logits, sigmoid=self.sigmoid)

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, state):
        if 'pos_weight' not in state:
            state['pos_weight'] = None
        self.__dict__ = state


class BinaryWeightedMultipleFixedPredictor(WeightedPredictionLayer):
    """
    Final prediction layer that has an output dimension of 2, but performs a prediction over multiple vectors,
    and then does a weighted sum over the softmax probabilities by the given vector weights.
    The given weight should sum to 1.
    """
    def __init__(self, fc_init='glorot_uniform', pos_weight=None, ):
        self.fc_init = fc_init
        self.pos_weight = pos_weight

    def apply(self, is_train, x, weights, answer: List):
        init = get_keras_initialization(self.fc_init)
        x_shape = x.shape.as_list()
        with tf.variable_scope("compute_encoding_logits"):
            encoding_pred_weights = tf.get_variable('encoding_weights', shape=[x_shape[1], x_shape[2], 2],
                                                    initializer=init)
            encoding_pred_biases = tf.get_variable('encoding_biases', shape=[x_shape[1], 2],
                                                   initializer=tf.zeros_initializer())
            encoding_logits = tf.einsum('btd,tdx->btx', x, encoding_pred_weights) + encoding_pred_biases
            encoding_softmaxes = tf.nn.softmax(encoding_logits, axis=-1)
            weighted_encoding_softmaxes = encoding_softmaxes * tf.expand_dims(weights, -1)
            weighted_softmax = tf.reduce_sum(weighted_encoding_softmaxes, axis=1)

        with tf.variable_scope("compute_logits"):  # this is only for compatibility with the logits name of old models
            logits = tf.log(weighted_softmax, name='fully_connected/BiasAdd')

        cross_ent = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=answer[0], logits=logits)
        if self.pos_weight is not None:
            cross_ent = cross_ent * tf.to_float((answer[0] * (self.pos_weight-1)) + 1)
        loss = tf.reduce_mean(cross_ent)
        tf.add_to_collection(tf.GraphKeys.LOSSES, loss)
        return BinaryPrediction(logits)


class BinaryNullPredictor(FixedPredictionLayer):
    """
    Final prediction layer that already takes in a value for each sample, just need to create the loss.
    """
    def __init__(self, use_ranking_loss=False, ranking_lambda=1.0, gamma=1.0):
        self.use_ranking_loss = use_ranking_loss
        self.ranking_lambda = ranking_lambda
        self.gamma = gamma

    def apply(self, is_train, x, answer: List):
        if self.use_ranking_loss and len(answer) != 2:
            raise NotImplementedError()
        logits = tf.squeeze(x, axis=1, name='logits')  # for ease of access by tensor name

        cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.to_float(answer[0]), logits=logits)
        loss = tf.reduce_mean(cross_ent)
        tf.add_to_collection(tf.GraphKeys.LOSSES, loss)

        if self.use_ranking_loss:
            with tf.variable_scope('pairwise_ranking'):
                margin_losses = grouped_pairwise_margin_loss(tf.sigmoid(logits), answer[1], answer[0], gamma=self.gamma)
                margin_loss = tf.reduce_mean(margin_losses) * self.ranking_lambda
                tf.add_to_collection(tf.GraphKeys.LOSSES, margin_loss)
        return BinaryPrediction(logits, sigmoid=True)

    def __setstate__(self, state):
        if 'use_ranking_loss' not in state:
            state['use_ranking_loss'] = False
        if 'ranking_lambda' not in state:
            state['ranking_lambda'] = 1.0
        if 'gamma' not in state:
            state['gamma'] = 1.0
        self.__dict__ = state


def grouped_pairwise_margin_loss(similarities, group_ids, labels, gamma=1.0):
    """
    Calculates the pairwise margin ranking loss between paragraph of same question in a batch.
    Assumes at least (and most likely also at most) 2 paragraphs, one positive and one negative, for each question.
    """
    _, group_segments = tf.unique(group_ids)
    num_segments = tf.reduce_max(group_segments) + 1
    # we take advantage of the fact that negative segment ids get dropped
    # note: if label==1, 2*label-1 == 1, but if label==0, 2*label-1==-1
    positive_ranks = tf.unsorted_segment_mean(similarities, (group_segments + 1) * (2*labels - 1) - 1,
                                              num_segments=num_segments)
    negative_ranks = tf.unsorted_segment_mean(similarities, (group_segments + 1) * (1 - 2*labels) - 1,
                                              num_segments=num_segments)
    return tf.maximum(gamma - positive_ranks + negative_ranks, 0.)
