import json
from os.path import join
from typing import Optional

import h5py
import numpy as np
import tensorflow as tf

from hotpot.config import LM_DIR
from hotpot.configurable import Configurable
from .data import UnicodeCharsVocabulary, Batcher

DTYPE = 'float32'
DTYPE_INT = 'int64'


class LanguageModel(Configurable):
    """ Pointer to the source files needed to use a pre-trained language model """

    def __init__(self, lm_vocab_file, options_file, weight_file, embed_weights_file: Optional[str]):
        self.lm_vocab_file = lm_vocab_file
        self.options_file = options_file
        self.weight_file = weight_file
        self.embed_weights_file = embed_weights_file


class SquadContextConcatSkip(LanguageModel):
    """ Model re-trained on SQuAD that uses skip connections"""

    def __init__(self):
        basedir = join(LM_DIR, "squad-context-concat-skip")
        super().__init__(
            join(basedir, "squad_train_dev_all_unique_tokens.txt"),
            join(basedir, "options_squad_lm_2x4096_512_2048cnn_2xhighway_skip.json"),
            join(basedir, "squad_context_concat_lm_2x4096_512_2048cnn_2xhighway_skip.hdf5"),
            join(basedir, "squad_train_dev_all_unique_tokens_context_concat_lm_2x4096_512_2048cnn_2xhighway_skip.hdf5"),
        )


class OriginalElmoModel(LanguageModel):
    """ The 5.5B ELMO model"""
    def __init__(self, vocab_file, embeddings_file, options_file=None):
        basedir = join(LM_DIR, "5.5B")
        super().__init__(
            vocab_file,
            join(basedir,
                 options_file if options_file is not None else "elmo_2x4096_512_2048cnn_2xhighway_5.5B_options.json"),
            join(basedir, "elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5"),
            embeddings_file,
        )


class BidirectionalLanguageModel(object):
    def __init__(
            self,
            options_file: str,
            weight_file: str,
            ids_placeholder,
            use_character_inputs=True,
            embedding_weight_file=None,
            max_batch_size=128, token_embeds_as_placeholders=False
    ):
        '''
        Creates the language model computational graph and loads weights
        Two options for input type:
            (1) To use character inputs (paired with Batcher)
                pass use_character_inputs=True, and ids_placeholder
                of shape (None, None, max_characters_per_token)
            (2) To use token ids as input (paired with TokenBatcher),
                pass use_character_inputs=False and ids_placeholder
                of shape (None, None).  In this case, embedding_weight_file
                is also required input
        options_file: location of the json formatted file with
                      LM hyperparameters
        weight_file: location of the hdf5 file with LM weights
        ids_placeholder: a tf.placeholder of type int32.
            If use_character_inputs=True, it is shape
                (None, None, max_characters_per_token) and holds the input
                character ids for a batch
            If use_character_input=False, it is shape (None, None) and
                holds the input token ids for a batch
        use_character_inputs: if True, then use character ids as input,
            otherwise use token ids
        max_batch_size: the maximum allowable batch size
        '''
        with open(options_file, 'r') as fin:
            options = json.load(fin)

        if not use_character_inputs:
            if embedding_weight_file is None:
                raise ValueError(
                    "embedding_weight_file is required input with "
                    "not use_character_inputs"
                )

        self._lm_graph = BidirectionalLanguageModelGraph(
            options,
            weight_file,
            ids_placeholder,
            embedding_weight_file=embedding_weight_file,
            use_character_inputs=use_character_inputs,
            max_batch_size=max_batch_size,
            token_embeds_as_placeholders=token_embeds_as_placeholders)

        self._ops = None

    def get_ops(self):
        '''
        returns a dictionary with tensorflow ops:
            {'lm_embeddings': embedding_op,
             'lengths': sequence_lengths_op,
             'mask': op to compute mask}
        embedding_op computes the LM embeddings and is shape
            (None, 3, None, 1024)
        lengths_op computes the sequence lengths and is shape (None, )
        mask computes the sequence mask and is shape (None, None)
        '''
        if self._ops is None:
            self._build_ops()
        return self._ops

    def _build_ops(self):
        with tf.control_dependencies([self._lm_graph.update_state_op]):
            # get the LM embeddings
            token_embeddings = self._lm_graph.embedding
            layers = [
                tf.concat([token_embeddings, token_embeddings], axis=2)
            ]

            n_lm_layers = len(self._lm_graph.lstm_outputs['forward'])
            for i in range(n_lm_layers):
                layers.append(
                    tf.concat(
                        [self._lm_graph.lstm_outputs['forward'][i],
                         self._lm_graph.lstm_outputs['backward'][i]],
                        axis=-1
                    )
                )

            # The layers include the BOS/EOS tokens.  Remove them
            sequence_length_wo_bos_eos = self._lm_graph.sequence_lengths - 2
            layers_without_bos_eos = []
            for layer in layers:
                layer_wo_bos_eos = layer[:, 1:, :]
                layer_wo_bos_eos = tf.reverse_sequence(
                    layer_wo_bos_eos,
                    self._lm_graph.sequence_lengths - 1,
                    seq_axis=1,
                    batch_axis=0,
                )
                layer_wo_bos_eos = layer_wo_bos_eos[:, 1:, :]
                layer_wo_bos_eos = tf.reverse_sequence(
                    layer_wo_bos_eos,
                    sequence_length_wo_bos_eos,
                    seq_axis=1,
                    batch_axis=0,
                )
                layers_without_bos_eos.append(layer_wo_bos_eos)

            # concatenate the layers
            lm_embeddings = tf.concat(
                [tf.expand_dims(t, axis=1) for t in layers_without_bos_eos],
                axis=1
            )

            # get the mask op without bos/eos.
            # tf doesn't support reversing boolean tensors, so cast
            # to int then back
            mask_wo_bos_eos = tf.cast(self._lm_graph.mask[:, 1:], 'int32')
            mask_wo_bos_eos = tf.reverse_sequence(
                mask_wo_bos_eos,
                self._lm_graph.sequence_lengths - 1,
                seq_axis=1,
                batch_axis=0,
            )
            mask_wo_bos_eos = mask_wo_bos_eos[:, 1:]
            mask_wo_bos_eos = tf.reverse_sequence(
                mask_wo_bos_eos,
                sequence_length_wo_bos_eos,
                seq_axis=1,
                batch_axis=0,
            )
            mask_wo_bos_eos = tf.cast(mask_wo_bos_eos, 'bool')

        self._ops = {
            'lm_embeddings': lm_embeddings,
            'lengths': sequence_length_wo_bos_eos,
            'token_embeddings': self._lm_graph.embedding,
            'mask': mask_wo_bos_eos,
        }


def load_elmo_pretrained_token_embeddings(embedding_weight_file):
    with h5py.File(embedding_weight_file, 'r') as fin:
        # Have added a special 0 index for padding not present
        # in the original model.
        embed_weights = fin['embedding'][...]
        weights = np.zeros(
            (embed_weights.shape[0] + 1, embed_weights.shape[1]),
            dtype=DTYPE
        )
        weights[1:, :] = embed_weights
    return weights


def _pretrained_initializer(varname, weight_file, embedding_weight_file=None):
    '''
    We'll stub out all the initializers in the pretrained LM with
    a function that loads the weights from the file
    '''
    weight_name_map = {}
    for i in range(2):
        for j in range(8):  # if we decide to add more layers
            root = 'RNN_{}/RNN/MultiRNNCell/Cell{}'.format(i, j)
            weight_name_map[root + '/rnn/lstm_cell/kernel'] = \
                root + '/LSTMCell/W_0'
            weight_name_map[root + '/rnn/lstm_cell/bias'] = \
                root + '/LSTMCell/B'
            weight_name_map[root + '/rnn/lstm_cell/projection/kernel'] = \
                root + '/LSTMCell/W_P_0'

    # convert the graph name to that in the checkpoint
    varname_in_file = varname[5:]
    if varname_in_file.startswith('RNN'):
        varname_in_file = weight_name_map[varname_in_file]

    if varname_in_file == 'embedding':
        with h5py.File(embedding_weight_file, 'r') as fin:  # TODO: move this to someplace it can be fed as placeholder
            # Have added a special 0 index for padding not present
            # in the original model.
            embed_weights = fin[varname_in_file][...]
            weights = np.zeros(
                (embed_weights.shape[0] + 1, embed_weights.shape[1]),
                dtype=DTYPE
            )
            weights[1:, :] = embed_weights
    else:
        with h5py.File(weight_file, 'r') as fin:
            if varname_in_file == 'char_embed':
                # Have added a special 0 index for padding not present
                # in the original model.
                char_embed_weights = fin[varname_in_file][...]
                weights = np.zeros(
                    (char_embed_weights.shape[0] + 1,
                     char_embed_weights.shape[1]),
                    dtype=DTYPE
                )
                weights[1:, :] = char_embed_weights
            else:
                weights = fin[varname_in_file][...]

    # Tensorflow initializers are callables that accept a shape parameter
    # and some optional kwargs
    def ret(shape, **kwargs):
        if list(shape) != list(weights.shape):
            raise ValueError(
                "Invalid shape initializing {0}, got {1}, expected {2}".format(
                    varname_in_file, shape, weights.shape)
            )
        return weights

    return ret


class BidirectionalLanguageModelGraph(object):
    '''
    Creates the computational graph and holds the ops necessary for runnint
    a bidirectional language model
    '''

    def __init__(self, options, weight_file, ids_placeholder,
                 use_character_inputs=True, embedding_weight_file=None,
                 max_batch_size=128, token_embeds_as_placeholders=False):

        self.options = options
        self._max_batch_size = max_batch_size
        self.ids_placeholder = ids_placeholder
        self.use_character_inputs = use_character_inputs
        self.token_embeds_as_placeholders = token_embeds_as_placeholders

        # this custom_getter will make all variables not trainable and
        # override the default initializer
        def custom_getter(getter, name, *args, **kwargs):
            kwargs['trainable'] = False
            kwargs['initializer'] = _pretrained_initializer(
                name, weight_file, embedding_weight_file
            )
            return getter(name, *args, **kwargs)

        if self.token_embeds_as_placeholders and not self.use_character_inputs:
            with tf.variable_scope('bilm', reuse=tf.AUTO_REUSE):
                self._build_word_embeddings()
        with tf.variable_scope('bilm', custom_getter=custom_getter):
            self._build()

    def _build(self):
        if self.use_character_inputs:
            self._build_word_char_embeddings()
        elif not self.token_embeds_as_placeholders:
            self._build_word_embeddings()
        self._build_lstms()

    def _build_word_char_embeddings(self):
        '''
        options contains key 'char_cnn': {
        'n_characters': 60,
        # includes the start / end characters
        'max_characters_per_token': 17,
        'filters': [
            [1, 32],
            [2, 32],
            [3, 64],
            [4, 128],
            [5, 256],
            [6, 512],
            [7, 512]
        ],
        'activation': 'tanh',
        # for the character embedding
        'embedding': {'dim': 16}
        # for highway layers
        # if omitted, then no highway layers
        'n_highway': 2,
        }
        '''
        projection_dim = self.options['lstm']['projection_dim']

        cnn_options = self.options['char_cnn']
        filters = cnn_options['filters']
        n_filters = sum(f[1] for f in filters)
        max_chars = cnn_options['max_characters_per_token']
        char_embed_dim = cnn_options['embedding']['dim']
        n_chars = cnn_options['n_characters']
        if cnn_options['activation'] == 'tanh':
            activation = tf.nn.tanh
        elif cnn_options['activation'] == 'relu':
            activation = tf.nn.relu

        # the character embeddings
        with tf.device("/cpu:0"):
            self.embedding_weights = tf.get_variable(
                "char_embed", [n_chars, char_embed_dim],
                dtype=DTYPE,
                initializer=tf.random_uniform_initializer(-1.0, 1.0)
            )
            # shape (batch_size, unroll_steps, max_chars, embed_dim)
            self.char_embedding = tf.nn.embedding_lookup(self.embedding_weights,
                                                         self.ids_placeholder)

        # the convolutions
        def make_convolutions(inp):
            with tf.variable_scope('CNN') as scope:
                convolutions = []
                for i, (width, num) in enumerate(filters):
                    if cnn_options['activation'] == 'relu':
                        # He initialization for ReLU activation
                        # with char embeddings init between -1 and 1
                        # w_init = tf.random_normal_initializer(
                        #    mean=0.0,
                        #    stddev=np.sqrt(2.0 / (width * char_embed_dim))
                        # )

                        # Kim et al 2015, +/- 0.05
                        w_init = tf.random_uniform_initializer(
                            minval=-0.05, maxval=0.05)
                    elif cnn_options['activation'] == 'tanh':
                        # glorot init
                        w_init = tf.random_normal_initializer(
                            mean=0.0,
                            stddev=np.sqrt(1.0 / (width * char_embed_dim))
                        )
                    w = tf.get_variable(
                        "W_cnn_%s" % i,
                        [1, width, char_embed_dim, num],
                        initializer=w_init,
                        dtype=DTYPE)
                    b = tf.get_variable(
                        "b_cnn_%s" % i, [num], dtype=DTYPE,
                        initializer=tf.constant_initializer(0.0))

                    conv = tf.nn.conv2d(
                        inp, w,
                        strides=[1, 1, 1, 1],
                        padding="VALID") + b
                    # now max pool
                    conv = tf.nn.max_pool(
                        conv, [1, 1, max_chars - width + 1, 1],
                        [1, 1, 1, 1], 'VALID')

                    # activation
                    conv = activation(conv)
                    conv = tf.squeeze(conv, axis=[2])

                    convolutions.append(conv)

            return tf.concat(convolutions, 2)

        embedding = make_convolutions(self.char_embedding)

        # for highway and projection layers
        n_highway = cnn_options.get('n_highway')
        use_highway = n_highway is not None and n_highway > 0
        use_proj = n_filters != projection_dim

        if use_highway or use_proj:
            #   reshape from (batch_size, n_tokens, dim) to (-1, dim)
            batch_size_n_tokens = tf.shape(embedding)[0:2]
            embedding = tf.reshape(embedding, [-1, n_filters])

        # set up weights for projection
        if use_proj:
            assert n_filters > projection_dim
            with tf.variable_scope('CNN_proj') as scope:
                W_proj_cnn = tf.get_variable(
                    "W_proj", [n_filters, projection_dim],
                    initializer=tf.random_normal_initializer(
                        mean=0.0, stddev=np.sqrt(1.0 / n_filters)),
                    dtype=DTYPE)
                b_proj_cnn = tf.get_variable(
                    "b_proj", [projection_dim],
                    initializer=tf.constant_initializer(0.0),
                    dtype=DTYPE)

        # apply highways layers
        def high(x, ww_carry, bb_carry, ww_tr, bb_tr):
            carry_gate = tf.nn.sigmoid(tf.matmul(x, ww_carry) + bb_carry)
            transform_gate = tf.nn.relu(tf.matmul(x, ww_tr) + bb_tr)
            return carry_gate * transform_gate + (1.0 - carry_gate) * x

        if use_highway:
            highway_dim = n_filters

            for i in range(n_highway):
                with tf.variable_scope('CNN_high_%s' % i) as scope:
                    W_carry = tf.get_variable(
                        'W_carry', [highway_dim, highway_dim],
                        # glorit init
                        initializer=tf.random_normal_initializer(
                            mean=0.0, stddev=np.sqrt(1.0 / highway_dim)),
                        dtype=DTYPE)
                    b_carry = tf.get_variable(
                        'b_carry', [highway_dim],
                        initializer=tf.constant_initializer(-2.0),
                        dtype=DTYPE)
                    W_transform = tf.get_variable(
                        'W_transform', [highway_dim, highway_dim],
                        initializer=tf.random_normal_initializer(
                            mean=0.0, stddev=np.sqrt(1.0 / highway_dim)),
                        dtype=DTYPE)
                    b_transform = tf.get_variable(
                        'b_transform', [highway_dim],
                        initializer=tf.constant_initializer(0.0),
                        dtype=DTYPE)

                embedding = high(embedding, W_carry, b_carry,
                                 W_transform, b_transform)

        # finally project down if needed
        if use_proj:
            embedding = tf.matmul(embedding, W_proj_cnn) + b_proj_cnn

        # reshape back to (batch_size, tokens, dim)
        if use_highway or use_proj:
            shp = tf.concat([batch_size_n_tokens, [projection_dim]], axis=0)
            embedding = tf.reshape(embedding, shp)

        # at last assign attributes for remainder of the model
        self.embedding = embedding

    @staticmethod
    def get_word_embedding_init_placeholder_and_op(options_file: str, ):
        with open(options_file, 'r') as fin:
            options = json.load(fin)
        n_tokens_vocab = options['n_tokens_vocab']  # TODO: hard coded vocab size, make it depend on the vocab file instead.
        projection_dim = options['lstm']['projection_dim']

        with tf.variable_scope('bilm', reuse=tf.AUTO_REUSE):
            with tf.device("/cpu:0"):
                embedding_weights = tf.get_variable("embedding",
                                                    [n_tokens_vocab, projection_dim],
                                                    dtype=DTYPE, trainable=False,
                                                    initializer=tf.constant_initializer(0.0))

        embedding_placeholder = tf.placeholder(DTYPE, [n_tokens_vocab, projection_dim])
        embedding_init = embedding_weights.assign(embedding_placeholder)
        return embedding_placeholder, embedding_init

    def _build_word_embeddings(self):
        n_tokens_vocab = self.options['n_tokens_vocab']

        projection_dim = self.options['lstm']['projection_dim']

        # the word embeddings
        with tf.device("/cpu:0"):
            # self.embedding_weights = tf.get_variable("embedding",
            #                                          [n_tokens_vocab, projection_dim],
            #                                          dtype=DTYPE, trainable=False,
            #                                          initializer=tf.constant_initializer(0.0))
            #
            # embedding_placeholder = tf.placeholder(DTYPE, [n_tokens_vocab, projection_dim])
            # embedding_init = self.embedding_weights.assign(embedding_placeholder)

            # Old code - maybe problematic computational graph as it's too big for checkpointing ( maybe now not!)
            self.embedding_weights = tf.get_variable(
                "embedding", [n_tokens_vocab, projection_dim],
                dtype=DTYPE, trainable=False, initializer=tf.constant_initializer(0.0)
            )
            self.embedding = tf.nn.embedding_lookup(self.embedding_weights,
                                                    self.ids_placeholder)

    def _build_lstms(self):
        # now the LSTMs
        # these will collect the initial states for the forward
        #   (and reverse LSTMs if we are doing bidirectional)

        # parse the options
        lstm_dim = self.options['lstm']['dim']
        projection_dim = self.options['lstm']['projection_dim']
        n_lstm_layers = self.options['lstm'].get('n_layers', 1)
        cell_clip = self.options['lstm'].get('cell_clip')
        proj_clip = self.options['lstm'].get('proj_clip')
        use_skip_connections = self.options['lstm'].get(
            'use_skip_connections')
        if use_skip_connections:
            print("USING SKIP CONNECTIONS")

        # the sequence lengths from input mask
        if self.use_character_inputs:
            mask = tf.reduce_any(self.ids_placeholder > 0, axis=2)
        else:
            mask = self.ids_placeholder > 0
        sequence_lengths = tf.reduce_sum(tf.cast(mask, tf.int32), axis=1)
        batch_size = tf.shape(sequence_lengths)[0]

        # for each direction, we'll store tensors for each layer
        self.lstm_outputs = {'forward': [], 'backward': []}
        self.lstm_state_sizes = {'forward': [], 'backward': []}
        self.lstm_init_states = {'forward': [], 'backward': []}
        self.lstm_final_states = {'forward': [], 'backward': []}

        update_ops = []
        for direction in ['forward', 'backward']:
            if direction == 'forward':
                layer_input = self.embedding
            else:
                layer_input = tf.reverse_sequence(
                    self.embedding,
                    sequence_lengths,
                    seq_axis=1,
                    batch_axis=0
                )

            for i in range(n_lstm_layers):
                if projection_dim < lstm_dim:
                    # are projecting down output
                    lstm_cell = tf.nn.rnn_cell.LSTMCell(
                        lstm_dim, num_proj=projection_dim,
                        cell_clip=cell_clip, proj_clip=proj_clip)
                else:
                    lstm_cell = tf.nn.rnn_cell.LSTMCell(
                        lstm_dim,
                        cell_clip=cell_clip, proj_clip=proj_clip)

                if use_skip_connections:
                    # ResidualWrapper adds inputs to outputs
                    if i == 0:
                        # don't add skip connection from token embedding to
                        # 1st layer output
                        pass
                    else:
                        # add a skip connection
                        lstm_cell = tf.nn.rnn_cell.ResidualWrapper(lstm_cell)

                # collect the input state, run the dynamic rnn, collect
                # the output
                state_size = lstm_cell.state_size
                # the LSTMs are stateful.  To support multiple batch sizes,
                # we'll allocate size for states up to max_batch_size,
                # then use the first batch_size entries for each batch
                init_states = [
                    tf.Variable(
                        tf.zeros([self._max_batch_size, dim]),
                        trainable=False
                    )
                    for dim in lstm_cell.state_size
                ]
                batch_init_states = [
                    state[:batch_size, :] for state in init_states
                ]

                if direction == 'forward':
                    i_direction = 0
                else:
                    i_direction = 1
                variable_scope_name = 'RNN_{0}/RNN/MultiRNNCell/Cell{1}'.format(
                    i_direction, i)
                with tf.variable_scope(variable_scope_name):
                    layer_output, final_state = tf.nn.dynamic_rnn(
                        lstm_cell,
                        layer_input,
                        sequence_length=sequence_lengths,
                        initial_state=tf.nn.rnn_cell.LSTMStateTuple(
                            *batch_init_states),
                    )

                self.lstm_state_sizes[direction].append(lstm_cell.state_size)
                self.lstm_init_states[direction].append(init_states)
                self.lstm_final_states[direction].append(final_state)
                if direction == 'forward':
                    self.lstm_outputs[direction].append(layer_output)
                else:
                    self.lstm_outputs[direction].append(
                        tf.reverse_sequence(
                            layer_output,
                            sequence_lengths,
                            seq_axis=1,
                            batch_axis=0
                        )
                    )

                with tf.control_dependencies([layer_output]):
                    # update the initial states
                    for i in range(2):
                        new_state = tf.concat(
                            [final_state[i][:batch_size, :],
                             init_states[i][batch_size:, :]], axis=0)
                        state_update_op = tf.assign(init_states[i], new_state)
                        update_ops.append(state_update_op)

                layer_input = layer_output

        self.mask = mask
        self.sequence_lengths = sequence_lengths
        self.update_state_op = tf.group(*update_ops)


def dump_token_embeddings(vocab_file, options_file, weight_file, outfile):
    '''
    Given an input vocabulary file, dump all the token embeddings to the
    outfile.  The result can be used as the embedding_weight_file when
    constructing a BidirectionalLanguageModel.
    '''
    with open(options_file, 'r') as fin:
        options = json.load(fin)
    max_word_length = options['char_cnn']['max_characters_per_token']

    vocab = UnicodeCharsVocabulary(vocab_file, max_word_length)
    batcher = Batcher(vocab_file, max_word_length)

    ids_placeholder = tf.placeholder('int32',
                                     shape=(None, None, max_word_length)
                                     )
    model = BidirectionalLanguageModel(
        options_file, weight_file, ids_placeholder
    )

    embedding_op = model.get_ops()['token_embeddings']

    n_tokens = vocab.size
    embed_dim = int(embedding_op.shape[2])

    embeddings = np.zeros((n_tokens, embed_dim), dtype=DTYPE)

    config = tf.ConfigProto(allow_soft_placement=True)
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        for k in range(n_tokens):
            token = vocab.id_to_word(k)
            char_ids = batcher.batch_sentences([[token]])[0, 1, :].reshape(
                1, 1, -1)
            embeddings[k, :] = sess.run(
                embedding_op, feed_dict={ids_placeholder: char_ids}
            )

    with h5py.File(outfile, 'w') as fout:
        ds = fout.create_dataset(
            'embedding', embeddings.shape, dtype='float32', data=embeddings
        )
