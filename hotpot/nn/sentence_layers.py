import tensorflow as tf

from hotpot.configurable import Configurable


class SentencesEncoder(Configurable):
    """
    Encodes a sequence into sentence representations
    (batch, time, dim1) -> (batch, num_sents, dim2)
    """
    def apply(self, x, segments, num_sentences_mask):
        raise NotImplementedError()


class SentenceMaxEncoder(SentencesEncoder):
    def apply(self, x, segments, num_sentences_mask):
        num_segments = tf.shape(num_sentences_mask)[0] * tf.reduce_max(num_sentences_mask)
        segmented = tf.unsorted_segment_max(x, segments, num_segments=num_segments)
        reshaped = tf.reshape(segmented, (tf.shape(x)[0], -1, x.shape.as_list()[2]))
        mask = tf.expand_dims(tf.cast(tf.sequence_mask(num_sentences_mask, tf.shape(reshaped)[1]), tf.float32), 2)
        reshaped *= mask
        return reshaped
