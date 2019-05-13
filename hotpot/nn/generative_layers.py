"""
Layers that generate sequences from the input, usually a single vector but maybe more
"""
from typing import Optional

import tensorflow as tf
from tensorflow.python.ops.rnn_cell_impl import RNNCell

from hotpot.configurable import Configurable
from hotpot.nn.layers import Mapper


class VecToSeq(Configurable):
    """ (batch, in_dim) -> (batch, time, out_dim) """
    def apply(self, is_train, x):
        raise NotImplementedError()


class GenerativeRNN(VecToSeq):
    def __init__(self, rnn_cell: RNNCell, output_layer: Optional[Mapper], vec_to_in: Optional[Mapper],
                 seq_len: int, include_original_vec: bool):
        self.rnn_cell = rnn_cell
        self.output_layer = output_layer
        self.vec_to_in = vec_to_in
        self.seq_len = seq_len
        self.include_original_vec = include_original_vec

    def apply(self, is_train, x):
        cell_state = self.rnn_cell.zero_state(tf.shape(x)[0], dtype=tf.float32)
        cell_outputs = x
        if self.vec_to_in is not None:
            with tf.variable_scope('vec_to_in_layer'):
                x = self.vec_to_in.apply(is_train, x)
        outputs = [x] if self.include_original_vec else []
        for time in range(self.seq_len):
            cell_outputs, cell_state = self.rnn_cell(inputs=cell_outputs, state=cell_state)
            if self.output_layer is not None:
                with tf.variable_scope('output_layer'):
                    cell_outputs = self.output_layer.apply(is_train, cell_outputs)
            outputs.append(cell_outputs)
        return tf.stack(outputs, axis=1)

