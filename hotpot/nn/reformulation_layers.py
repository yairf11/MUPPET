""" Layers used for iterative relevance prediction, when reformulating the question for the second retrieval """
from typing import Union, Optional

import tensorflow as tf
from tensorflow.python.layers.core import fully_connected

from hotpot.configurable import Configurable
from hotpot.nn.layers import get_keras_initialization, get_keras_activation, SequenceEncoder, Mapper, SequenceMapper
from hotpot.nn.ops import exp_mask


class ReformulationLayer(Configurable):
    """ Basic reformulation layer """

    def apply(self, is_train, sentences_rep, iteration1_logits, mask):
        raise NotImplementedError()


class WeightedSumThenProjectReformulation(ReformulationLayer):
    def __init__(self, n_project: int, activation: Union[str, None] = 'relu', init='glorot_uniform'):
        self.n_project = n_project
        self.activation = activation
        self.init = init

    def apply(self, is_train, sentences_rep, iteration1_logits, mask):
        init = get_keras_initialization(self.init)
        masked_logits1 = exp_mask(tf.squeeze(iteration1_logits, axis=[2]), mask)
        weights = tf.nn.softmax(masked_logits1)
        weighted_rep = tf.reduce_sum(tf.expand_dims(weights, axis=2) * sentences_rep, axis=1)
        return fully_connected(weighted_rep, units=self.n_project, kernel_initializer=init,
                               use_bias=True, activation=get_keras_activation(self.activation))


class ProjectThenWeightedSumReformulation(ReformulationLayer):
    def __init__(self, n_project: int, activation: Union[str, None] = 'relu', init='glorot_uniform'):
        self.n_project = n_project
        self.activation = activation
        self.init = init

    def apply(self, is_train, sentences_rep, iteration1_logits, mask):
        init = get_keras_initialization(self.init)
        projected = fully_connected(sentences_rep, units=self.n_project, kernel_initializer=init,
                                    use_bias=True, activation=get_keras_activation(self.activation))
        masked_logits1 = exp_mask(tf.squeeze(iteration1_logits, axis=[2]), mask)
        weights = tf.nn.softmax(masked_logits1)
        return tf.reduce_sum(tf.expand_dims(weights, axis=2) * projected, axis=1)


class ProjectMapEncodeReformulation(ReformulationLayer):
    def __init__(self, project_layer: Optional[Mapper], sequence_mapper: Optional[SequenceMapper],
                 encoder: SequenceEncoder):
        self.project_layer = project_layer
        self.sequence_mapper = sequence_mapper
        self.encoder = encoder

    def apply(self, is_train, sentences_rep, iteration1_logits, mask):
        if self.project_layer is not None:
            sentences_rep = self.project_layer.apply(is_train, sentences_rep, mask)
        if self.sequence_mapper is not None:
            sentences_rep = self.sequence_mapper.apply(is_train, sentences_rep, mask)
        return self.encoder.apply(is_train, sentences_rep, mask)


