from typing import Optional, Dict, List, Union, Set

from tensorflow import Tensor
import tensorflow as tf

from hotpot.data_handling.dataset import QuestionAndParagraphsSpec, QuestionAndParagraphsDataset
from hotpot.elmo.data import TokenBatcher, Batcher
from hotpot.elmo.elmo import ElmoWrapper
from hotpot.elmo.lm_model import LanguageModel, BidirectionalLanguageModelGraph, BidirectionalLanguageModel
from hotpot.encoder import QuestionsAndParagraphsEncoder
from hotpot.model import Model, Prediction
from hotpot.nn.attention import AttentionWithPostMapper
from hotpot.nn.embedder import WordEmbedder, CharWordEmbedder
from hotpot.nn.generative_layers import VecToSeq
from hotpot.nn.layers import SequenceMapper, AttentionMapper, SequenceEncoder, \
    MultipleMergeLayer, MaxMerge, ConcatWithProduct, MergeLayer, Mapper, FixedMergeLayer
from hotpot.nn.relevance_prediction import BinaryFixedPredictor
from hotpot.utils import ResourceLoader

INTERMEDIATE_LAYER_COLLECTION = 'intermediate'


class MultipleContextModel(Model):
    """
    A basic model that handles a question with mutltiple contexts.
    This base model should implement the initialization and encoding, but leaving the computational model to
      its subclasses.
    """

    def __init__(self,
                 encoder: QuestionsAndParagraphsEncoder,
                 word_embed: Optional[WordEmbedder],
                 char_embed: Optional[CharWordEmbedder] = None,
                 max_batch_size: Optional[int] = None,
                 elmo_model: Optional[LanguageModel] = None):
        if word_embed is None and char_embed is None:
            raise ValueError("Must have at least one kind of word/char embedder")
        self.encoder = encoder
        self.word_embed = word_embed
        self.char_embed = char_embed
        self.max_batch_size = max_batch_size
        self.lm_model = elmo_model

        # legacy from original elmo code
        self._max_num_sentences = self.max_batch_size
        self._batcher = None
        self._max_word_size = None

        self.per_sentence = False
        # placeholders
        self._is_train_placeholder = None
        # placeholders for lm
        self._batch_len_placeholders = None
        self._question_char_ids_placeholder = None
        self._context_char_ids_placeholder = None
        # we don't allow per-sentence option yet, so this is probably useless
        self._context_sentence_ixs = None

        # For creating the lm token embeddings as placeholders so they will not affect the checkpoint files
        self._elmo_token_embed_placeholder = None
        self._elmo_token_embed_init_op = None

    @property
    def use_elmo(self):
        """
        Are we using elmo, or is this just a regular model?
        """
        return self.lm_model is not None

    @property
    def token_lookup(self):
        """
        Are we using pre-computed word vectors, or running the LM's CNN to dynmacially derive
        word vectors from characters.
        """
        if not self.use_elmo:  # TODO: this is not pretty
            raise RuntimeError("Trying to access ELMO, but no ELMO in this model.")
        return self.lm_model.embed_weights_file is not None

    def init(self, train_word_counts, resource_loader: ResourceLoader):
        if self.word_embed is not None:
            self.word_embed.set_vocab(train_word_counts, resource_loader, None)
        if self.char_embed is not None:
            self.char_embed.embeder.set_vocab(train_word_counts)

    def set_inputs(self, datasets: Optional[List[QuestionAndParagraphsDataset]], word_vec_loader: ResourceLoader = None,
                   input_spec: Optional[QuestionAndParagraphsSpec]=None, voc: Optional[Set[str]]=None):
        if datasets and (input_spec is not None or voc is not None):
            raise ValueError("If datasets are given, don't specify input_spec and voc.")
        elif not datasets and (input_spec is None or voc is None):
            raise ValueError("If datasets are not given, must specify input_spec and voc.")

        if voc is None:
            voc = set()
            for dataset in datasets:
                voc.update(dataset.get_vocab())

        if input_spec is None:
            input_spec = datasets[0].get_spec()
            for dataset in datasets[1:]:
                input_spec += dataset.get_spec()

        if word_vec_loader is None:
            word_vec_loader = ResourceLoader()
        if self.word_embed is not None:
            self.word_embed.init(word_vec_loader, voc)
        if self.char_embed is not None:
            self.char_embed.embeder.init(word_vec_loader, voc)
        self.encoder.init(input_spec, len_opt=True, word_emb=self.word_embed,
                          char_emb=None if self.char_embed is None else self.char_embed.embeder, num_contexts_opt=False)
        self._is_train_placeholder = tf.placeholder(tf.bool, ())

        if self.use_elmo:
            batch_size = input_spec.batch_size
            num_contexts = self.encoder.max_num_contexts
            if self.token_lookup:
                self._batcher = TokenBatcher(self.lm_model.lm_vocab_file)
                self._question_char_ids_placeholder = tf.placeholder(tf.int32, (batch_size, None))
                self._context_char_ids_placeholder = tf.placeholder(tf.int32, (batch_size, num_contexts, None))
                self._max_word_size = input_spec.max_word_size
                self._context_sentence_ixs = None
                self._elmo_token_embed_placeholder, self._elmo_token_embed_init_op = \
                    BidirectionalLanguageModelGraph.get_word_embedding_init_placeholder_and_op(
                        self.lm_model.options_file)
            else:
                input_spec.max_word_size = 50  # TODO hack, harded coded from the lm model
                self._batcher = Batcher(self.lm_model.lm_vocab_file, 50)
                self._max_word_size = input_spec.max_word_size
                self._question_char_ids_placeholder = tf.placeholder(tf.int32,
                                                                     (batch_size, None, self._max_word_size))
                if self.per_sentence:
                    raise NotImplementedError()
                    # self._context_char_ids_placeholder = tf.placeholder(tf.int32,
                    #                                                     (None, None, self._max_word_size))
                    # self._context_sentence_ixs = tf.placeholder(tf.int32, (batch_size, 3, None, 3))
                else:
                    self._context_char_ids_placeholder = tf.placeholder(tf.int32,
                                                                        (batch_size, num_contexts,
                                                                         None, self._max_word_size))
                    self._context_sentence_ixs = None

    def get_placeholders(self) -> List[Tensor]:
        if not self.use_elmo:
            return self.encoder.get_placeholders() + [self._is_train_placeholder]
        else:
            return self.encoder.get_placeholders() + [
                self._is_train_placeholder,
                self._question_char_ids_placeholder,
                self._context_char_ids_placeholder
            ] + ([self._context_sentence_ixs] if (self._context_sentence_ixs is not None) else [])

    def get_elmo_token_embed_ph_and_op(self):
        if not (self.use_elmo and self.token_lookup):
            raise NotImplementedError("Need to use elmo with token lookup for this method")
        return self._elmo_token_embed_placeholder, self._elmo_token_embed_init_op

    def get_predictions_for(self, input_tensors: Dict[Tensor, Tensor]) -> Prediction:
        is_train = input_tensors[self._is_train_placeholder]
        enc = self.encoder

        if self.use_elmo:
            q_lm_model = BidirectionalLanguageModel(self.lm_model.options_file, self.lm_model.weight_file,
                                                    input_tensors[self._question_char_ids_placeholder],
                                                    embedding_weight_file=self.lm_model.embed_weights_file,
                                                    use_character_inputs=not self.token_lookup,
                                                    max_batch_size=self.max_batch_size,
                                                    token_embeds_as_placeholders=True)
            q_lm_encoding = q_lm_model.get_ops()["lm_embeddings"]

            with tf.variable_scope(tf.get_variable_scope(), reuse=True):
                original_shape = tf.shape(input_tensors[self._context_char_ids_placeholder])
                if self.token_lookup:
                    flattened_contexts = tf.reshape(input_tensors[self._context_char_ids_placeholder],
                                                    [original_shape[0] * original_shape[1], original_shape[2]])
                else:
                    flattened_contexts = tf.reshape(input_tensors[self._context_char_ids_placeholder],
                                                    [original_shape[0] * original_shape[1],
                                                     original_shape[2], original_shape[3]])
                c_lm_model = BidirectionalLanguageModel(self.lm_model.options_file, self.lm_model.weight_file,
                                                        flattened_contexts,
                                                        embedding_weight_file=self.lm_model.embed_weights_file,
                                                        use_character_inputs=not self.token_lookup,
                                                        max_batch_size=self.max_batch_size,
                                                        token_embeds_as_placeholders=True)
                c_lm_encoding = c_lm_model.get_ops()["lm_embeddings"]
                lm_shape = c_lm_encoding.get_shape()
                c_lm_encoding = tf.reshape(c_lm_encoding, (original_shape[0], original_shape[1], lm_shape[1],
                                                           original_shape[2] - 2, lm_shape[3]))
        else:
            q_lm_encoding = None
            c_lm_encoding = None

        contexts_mask = input_tensors[enc.num_contexts]
        q_word_mask = input_tensors[enc.question_len]
        c_word_mask = input_tensors[enc.context_len_multiple]
        c_word = input_tensors[enc.context_words_multiple]
        q_char_mask = input_tensors[enc.question_word_len]
        c_char_mask = input_tensors[enc.context_word_len_multiple]
        c_char = input_tensors[enc.context_chars_multiple]

        if enc.use_sentence_segments:
            sentence_mask = input_tensors[enc.context_num_sentences]
            sentence_segments = input_tensors[enc.context_sentence_segments_multiple]
        else:
            sentence_mask = None
            sentence_segments = None

        c_char_list = tf.unstack(c_char, axis=1)
        c_char_mask_list = tf.unstack(c_char_mask,
                                      axis=1)  # FIXME assumes more than one context, might not be always the case

        c_word_list = tf.unstack(c_word, axis=1)
        c_word_mask_list = tf.unstack(c_word_mask, axis=1)

        q_embed = []
        c_embed = []

        if enc.question_chars in input_tensors:
            with tf.variable_scope("char-embed"):
                out = self.char_embed.embed(is_train,
                                            (input_tensors[enc.question_chars], q_char_mask),
                                            *zip(c_char_list, c_char_mask_list))
            q_embed.append(out[0])
            c_embed.append(tf.stack(out[1:], axis=1))

        if enc.question_words in input_tensors:
            with tf.variable_scope("word-embed"):
                out = self.word_embed.embed(is_train,
                                            (input_tensors[enc.question_words], q_word_mask),
                                            *zip(c_word_list, c_word_mask_list))
            q_embed.append(out[0])
            c_embed.append(tf.stack(out[1:], axis=1))

        q_embed = tf.concat(q_embed, axis=2)
        c_embed = tf.concat(c_embed, axis=3)  # axis=3 as contexts have an extra dimension over the questions

        answer = [input_tensors[x] for x in enc.answer_encoder.get_placeholders()]
        return self._get_predictions_for(is_train, q_embed, q_word_mask, c_embed,
                                         c_word_mask, answer,
                                         question_lm=q_lm_encoding, context_lm=c_lm_encoding,
                                         sentence_segments=sentence_segments, sentence_mask=sentence_mask)
        # TODO: what about context masks? NOT DONE YET, but for now lets leave it.

    def _get_predictions_for(self,
                             is_train,
                             question_embed, question_mask,
                             context_embed, context_mask,
                             answer,
                             question_lm, context_lm, sentence_segments, sentence_mask) -> Prediction:
        raise NotImplemented()

    def encode(self, batch: List, is_train: bool) -> Dict[Tensor, object]:
        if self.max_batch_size is not None:
            if len(batch) > self.max_batch_size:
                raise ValueError("The model can only use a batch <= %d, but got %d" %
                                 (self.max_batch_size, len(batch)))
        data = self.encoder.encode(batch, is_train)
        data[self._is_train_placeholder] = is_train

        if self.use_elmo:
            data[self._question_char_ids_placeholder] = self._batcher.batch_sentences([q.question for q in batch])
            flattened_contexts = [par for x in batch for par in x.paragraphs]
            batched_contexts = self._batcher.batch_sentences(flattened_contexts)
            if self.token_lookup:
                data[self._context_char_ids_placeholder] = batched_contexts.reshape((len(batch),
                                                                                     self.encoder.max_num_contexts,
                                                                                     -1))
            else:
                data[self._context_char_ids_placeholder] = batched_contexts.reshape((len(batch),
                                                                                     self.encoder.max_num_contexts,
                                                                                     -1,
                                                                                     self._max_word_size))
        return data

    def __getstate__(self):
        state = super().__getstate__()
        state["_is_train_placeholder"] = None
        return state

    def __setstate__(self, state):
        if 'lm_model' not in state:
            state['lm_model'] = None
        super().__setstate__(state)


class ContextPairRelevanceModel(MultipleContextModel):
    """
    Model that handles a triplet of (question, context1, context2) and predicts the relevancy of the contexts
    to the question.

    Currently, our method is like so:
    1. Given a sequence of word embeddings for each context (and question)
    2. Possibly process each one independently (RNN)
    3. Possibly use attention between the contexts and the question
    4. Create a fixed-size encoding for each context
    5. Merge the contexts to a single vector (concatenate, max-pool etc.)
    6. Predict relevancy using a fully connected layer
    """

    def __init__(self,
                 encoder: QuestionsAndParagraphsEncoder,
                 word_embed: Optional[WordEmbedder],
                 char_embed: Optional[CharWordEmbedder],
                 embed_mapper: Optional[SequenceMapper],
                 question_to_context_attention: Optional[AttentionMapper],
                 context_to_question_attention: Optional[AttentionMapper],
                 context_to_context_attention: Optional[AttentionMapper],
                 sequence_encoder: SequenceEncoder,
                 merger: MultipleMergeLayer,  # TODO: not sure actually what layer will come here
                 predictor: BinaryFixedPredictor,
                 max_batch_size: Optional[int] = None
                 ):
        super().__init__(encoder=encoder, word_embed=word_embed, char_embed=char_embed, max_batch_size=max_batch_size)
        self.embed_mapper = embed_mapper
        self.question_to_context_attention = question_to_context_attention
        self.context_to_context_attention = context_to_context_attention
        self.context_to_question_attention = context_to_question_attention
        self.sequence_encoder = sequence_encoder
        self.merger = merger
        self.predictor = predictor

    def _get_predictions_for(self,
                             is_train,
                             question_embed, question_mask,
                             context_embed, context_mask,  # TODO: change to multiple inputs (embed,mask)
                             answer,
                             question_lm=None, context_lm=None, sentence_segments=None, sentence_mask=None):
        question_rep, context_rep = question_embed, context_embed
        context1_rep, context2_rep = tf.unstack(context_rep, axis=1, num=2)
        context1_mask, context2_mask = tf.unstack(context_mask, axis=1, num=2)
        if self.embed_mapper is not None:
            with tf.variable_scope("map_embed"):
                context1_rep = self.embed_mapper.apply(is_train, context1_rep, context1_mask)
            with tf.variable_scope("map_embed", reuse=True):
                context2_rep = self.embed_mapper.apply(is_train, context2_rep, context2_mask)
                question_rep = self.embed_mapper.apply(is_train, question_rep, question_mask)

        # TODO: come up with a better way to call these attention layers ? this is kind of ugly
        if self.question_to_context_attention is not None:
            with tf.variable_scope("q2c"):
                q_to_c1 = self.question_to_context_attention.apply(is_train, x=context1_rep, keys=question_rep,
                                                                   memories=question_rep, x_mask=context1_mask,
                                                                   memory_mask=question_mask)
            with tf.variable_scope("q2c", reuse=True):
                q_to_c2 = self.question_to_context_attention.apply(is_train, x=context2_rep, keys=question_rep,
                                                                   memories=question_rep, x_mask=context2_mask,
                                                                   memory_mask=question_mask)

        if self.context_to_question_attention is not None:
            with tf.variable_scope("c2q"):
                c1_to_q = self.context_to_question_attention.apply(is_train, x=question_rep, keys=context1_rep,
                                                                   memories=context1_rep, x_mask=question_mask,
                                                                   memory_mask=context1_mask)
            with tf.variable_scope("c2q", reuse=True):
                c2_to_q = self.context_to_question_attention.apply(is_train, x=question_rep, keys=context2_rep,
                                                                   memories=context2_rep, x_mask=question_mask,
                                                                   memory_mask=context2_mask)

        if self.context_to_context_attention is not None:
            with tf.variable_scope("c2c"):
                c1_to_c2 = self.context_to_context_attention.apply(is_train, x=context2_rep, keys=context1_rep,
                                                                   memories=context1_rep, x_mask=context2_mask,
                                                                   memory_mask=context1_mask)
            with tf.variable_scope("c2c", reuse=True):
                c2_to_c1 = self.context_to_context_attention.apply(is_train, x=context1_rep, keys=context2_rep,
                                                                   memories=context2_rep, x_mask=context1_mask,
                                                                   memory_mask=context2_mask)

        with tf.variable_scope("seq_enc"):
            context1_rep = self.sequence_encoder.apply(is_train, context1_rep, context1_mask)
        with tf.variable_scope("seq_enc", reuse=True):
            context2_rep = self.sequence_encoder.apply(is_train, context2_rep, context2_mask)
            question_rep = self.sequence_encoder.apply(is_train, question_rep, question_mask)

        with tf.variable_scope("merger"):
            merged_rep = self.merger.apply(is_train, question_rep, context1_rep, context2_rep)

        with tf.variable_scope("predictor"):
            return self.predictor.apply(is_train, merged_rep, answer)


class ContextsToQuestionModel(MultipleContextModel):  # todo: make this look better
    def __init__(self,
                 encoder: QuestionsAndParagraphsEncoder,
                 word_embed: Optional[WordEmbedder],
                 char_embed: Optional[CharWordEmbedder],
                 embed_mapper: Optional[SequenceMapper],
                 context_to_question_attention: AttentionMapper,
                 attention_merger: MaxMerge,  # fixme hard-coded type, for now
                 sequence_encoder: SequenceEncoder,
                 predictor: BinaryFixedPredictor,
                 max_batch_size: Optional[int] = None
                 ):
        super().__init__(encoder=encoder, word_embed=word_embed, char_embed=char_embed, max_batch_size=max_batch_size)
        self.embed_mapper = embed_mapper
        self.context_to_question_attention = context_to_question_attention
        self.sequence_encoder = sequence_encoder
        self.predictor = predictor
        self.attention_merger = attention_merger

    def _get_predictions_for(self,
                             is_train,
                             question_embed, question_mask,
                             context_embed, context_mask,  # TODO: change to multiple inputs (embed,mask)
                             answer,
                             question_lm=None, context_lm=None, sentence_segments=None, sentence_mask=None):
        question_rep, context_rep = question_embed, context_embed
        context1_rep, context2_rep = tf.unstack(context_rep, axis=1, num=2)
        context1_mask, context2_mask = tf.unstack(context_mask, axis=1, num=2)
        if self.embed_mapper is not None:
            with tf.variable_scope("map_embed"):
                context1_rep = self.embed_mapper.apply(is_train, context1_rep, context1_mask)
            with tf.variable_scope("map_embed", reuse=True):
                context2_rep = self.embed_mapper.apply(is_train, context2_rep, context2_mask)
                question_rep = self.embed_mapper.apply(is_train, question_rep, question_mask)

        with tf.variable_scope("c2q"):
            c1_to_q = self.context_to_question_attention.apply(is_train, x=question_rep, keys=context1_rep,
                                                               memories=context1_rep, x_mask=question_mask,
                                                               memory_mask=context1_mask)
        with tf.variable_scope("c2q", reuse=True):
            c2_to_q = self.context_to_question_attention.apply(is_train, x=question_rep, keys=context2_rep,
                                                               memories=context2_rep, x_mask=question_mask,
                                                               memory_mask=context2_mask)

        with tf.variable_scope("attention_merger"):
            attended_rep = self.attention_merger.apply(is_train, c1_to_q, c2_to_q, question_mask, question_mask)

        with tf.variable_scope("seq_enc"):
            fixed_rep = self.sequence_encoder.apply(is_train, attended_rep, question_mask)

        with tf.variable_scope("predictor"):
            return self.predictor.apply(is_train, fixed_rep, answer)


class MultiHopContextsToQuestionModel(MultipleContextModel):  # todo: make this look better
    """
    More complicated model, does the following:
    1. embeds each sequence separately
    2. creates question-aware attended representations of the contexts (optional)
    3. creates context-aware context representations (optional)
    4. creates fully-aware question representations
    5. merges the representations, encodes them, and predicts


    """

    def __init__(self,
                 encoder: QuestionsAndParagraphsEncoder,
                 word_embed: Optional[WordEmbedder],
                 char_embed: Optional[CharWordEmbedder],
                 embed_mapper: Optional[SequenceMapper],
                 context_to_question_attention: AttentionMapper,
                 context_to_context_attention: Optional[AttentionWithPostMapper],
                 question_to_context_attention: Optional[AttentionWithPostMapper],
                 attention_merger: MaxMerge,  # fixme hard-coded type, for now
                 sequence_encoder: SequenceEncoder,
                 predictor: BinaryFixedPredictor,
                 max_batch_size: Optional[int] = None,
                 c2c_hops: int = 1):
        super().__init__(encoder=encoder, word_embed=word_embed, char_embed=char_embed, max_batch_size=max_batch_size)
        self.embed_mapper = embed_mapper
        self.context_to_question_attention = context_to_question_attention
        self.question_to_context_attention = question_to_context_attention
        self.context_to_context_attention = context_to_context_attention
        self.sequence_encoder = sequence_encoder
        self.predictor = predictor
        self.attention_merger = attention_merger
        self.c2c_hops = c2c_hops
        # self.intermediate_reps = {}

    def _get_predictions_for(self,
                             is_train,
                             question_embed, question_mask,
                             context_embed, context_mask,  # TODO: change to multiple inputs (embed,mask)
                             answer,
                             question_lm=None, context_lm=None, sentence_segments=None, sentence_mask=None):
        question_rep, context_rep = question_embed, context_embed
        context1_rep, context2_rep = tf.unstack(context_rep, axis=1, num=2)
        context1_mask, context2_mask = tf.unstack(context_mask, axis=1, num=2)
        # dummy_var = tf.Variable(0.0, dtype=tf.float32, trainable=False)
        if self.embed_mapper is not None:
            with tf.variable_scope("map_embed"):
                context1_rep = self.embed_mapper.apply(is_train, context1_rep, context1_mask)
                # TODO: This is how we are going to allow intermediate inputs. Do this in every place
                # we might want to inject a new input and discard the old layers of the network.
                # For now, leaving this here as a reference, and will use it with the relevant models.
                context1_rep = tf.identity(context1_rep, name='map_embed_context1')
                tf.add_to_collection('intermediate', context1_rep)
            with tf.variable_scope("map_embed", reuse=True):
                context2_rep = self.embed_mapper.apply(is_train, context2_rep, context2_mask)
                question_rep = self.embed_mapper.apply(is_train, question_rep, question_mask)

        if self.question_to_context_attention is not None:
            with tf.variable_scope("q2c"):
                context1_rep = self.question_to_context_attention.apply(is_train, x=context1_rep, keys=question_rep,
                                                                        memories=question_rep, x_mask=context1_mask,
                                                                        memory_mask=question_mask)
            with tf.variable_scope("q2c", reuse=True):
                context2_rep = self.question_to_context_attention.apply(is_train, x=context2_rep, keys=question_rep,
                                                                        memories=question_rep, x_mask=context2_mask,
                                                                        memory_mask=question_mask)

        if self.context_to_context_attention is not None:
            """ At first, we used regular bidirectional attention, maybe multiple times. 
                turns out, it is more beneficial to run the bidirectional attention in turns. 
                For now we leave the supposedly original code as a comment. """
            # for hop in range(self.c2c_hops):
            #     with tf.variable_scope(f"c2c_hop_{hop}"):
            #         c1_to_c2 = self.context_to_context_attention.apply(is_train, x=context2_rep, keys=context1_rep,
            #                                                            memories=context1_rep, x_mask=context2_mask,
            #                                                            memory_mask=context1_mask)
            #     with tf.variable_scope(f"c2c_hop_{hop}", reuse=True):
            #         c2_to_c1 = self.context_to_context_attention.apply(is_train, x=context1_rep, keys=context2_rep,
            #                                                            memories=context2_rep, x_mask=context1_mask,
            #                                                            memory_mask=context2_mask)
            #     context2_rep = c1_to_c2
            #     context1_rep = c2_to_c1
            for hop in range(self.c2c_hops):
                with tf.variable_scope(f"c2c_hop_{hop}"):
                    c1_to_c2 = self.context_to_context_attention.apply(is_train, x=context2_rep, keys=context1_rep,
                                                                       memories=context1_rep, x_mask=context2_mask,
                                                                       memory_mask=context1_mask)
                with tf.variable_scope(f"c2c_hop_{hop}", reuse=True):
                    c1_to_c2_to_c1 = self.context_to_context_attention.apply(is_train, x=context1_rep, keys=c1_to_c2,
                                                                             memories=c1_to_c2, x_mask=context1_mask,
                                                                             memory_mask=context2_mask)
                    c2_to_c1 = self.context_to_context_attention.apply(is_train, x=context1_rep, keys=context2_rep,
                                                                       memories=context2_rep, x_mask=context1_mask,
                                                                       memory_mask=context2_mask)
                    c2_to_c1_to_c2 = self.context_to_context_attention.apply(is_train, x=context2_rep, keys=c2_to_c1,
                                                                             memories=c2_to_c1, x_mask=context2_mask,
                                                                             memory_mask=context1_mask)
                context2_rep = tf.add(c2_to_c1_to_c2, c1_to_c2)
                context1_rep = tf.add(c1_to_c2_to_c1, c2_to_c1)

        with tf.variable_scope("c2q"):
            c1_to_q = self.context_to_question_attention.apply(is_train, x=question_rep, keys=context1_rep,
                                                               memories=context1_rep, x_mask=question_mask,
                                                               memory_mask=context1_mask)
        with tf.variable_scope("c2q", reuse=True):
            c2_to_q = self.context_to_question_attention.apply(is_train, x=question_rep, keys=context2_rep,
                                                               memories=context2_rep, x_mask=question_mask,
                                                               memory_mask=context2_mask)

        with tf.variable_scope("attention_merger"):
            attended_rep = self.attention_merger.apply(is_train, c1_to_q, c2_to_q, question_mask, question_mask)

        with tf.variable_scope("seq_enc"):
            fixed_rep = self.sequence_encoder.apply(is_train, attended_rep, question_mask)

        with tf.variable_scope("predictor"):
            return self.predictor.apply(is_train, fixed_rep, answer)


class MultiHopContextsOnlyModel(MultipleContextModel):  # todo: make this look better
    """
    A model that takes into account only the contexts without the question
    """

    def __init__(self,
                 encoder: QuestionsAndParagraphsEncoder,
                 word_embed: Optional[WordEmbedder],
                 char_embed: Optional[CharWordEmbedder],
                 embed_mapper: Optional[SequenceMapper],
                 context_to_context_attention: Optional[AttentionWithPostMapper],
                 sequence_encoder: SequenceEncoder,
                 predictor: BinaryFixedPredictor,
                 max_batch_size: Optional[int] = None,
                 c2c_hops: int = 1):
        super().__init__(encoder=encoder, word_embed=word_embed, char_embed=char_embed, max_batch_size=max_batch_size)
        self.embed_mapper = embed_mapper
        self.context_to_context_attention = context_to_context_attention
        self.sequence_encoder = sequence_encoder
        self.predictor = predictor
        self.c2c_hops = c2c_hops
        self.context_fixed_merge = ConcatWithProduct()
        # self.intermediate_reps = {}

    def _get_predictions_for(self,
                             is_train,
                             question_embed, question_mask,
                             context_embed, context_mask,  # TODO: change to multiple inputs (embed,mask)
                             answer,
                             question_lm=None, context_lm=None, sentence_segments=None, sentence_mask=None):
        question_rep, context_rep = question_embed, context_embed
        context1_rep, context2_rep = tf.unstack(context_rep, axis=1, num=2)
        context1_mask, context2_mask = tf.unstack(context_mask, axis=1, num=2)
        # dummy_var = tf.Variable(0.0, dtype=tf.float32, trainable=False)
        if self.embed_mapper is not None:
            with tf.variable_scope("map_embed"):
                context1_rep = self.embed_mapper.apply(is_train, context1_rep, context1_mask)
            with tf.variable_scope("map_embed", reuse=True):
                context2_rep = self.embed_mapper.apply(is_train, context2_rep, context2_mask)

        if self.context_to_context_attention is not None:
            for hop in range(self.c2c_hops):
                with tf.variable_scope(f"c2c_hop_{hop}"):
                    c1_to_c2 = self.context_to_context_attention.apply(is_train, x=context2_rep, keys=context1_rep,
                                                                       memories=context1_rep, x_mask=context2_mask,
                                                                       memory_mask=context1_mask)
                with tf.variable_scope(f"c2c_hop_{hop}", reuse=True):
                    c1_to_c2_to_c1 = self.context_to_context_attention.apply(is_train, x=context1_rep, keys=c1_to_c2,
                                                                             memories=c1_to_c2, x_mask=context1_mask,
                                                                             memory_mask=context2_mask)
                    c2_to_c1 = self.context_to_context_attention.apply(is_train, x=context1_rep, keys=context2_rep,
                                                                       memories=context2_rep, x_mask=context1_mask,
                                                                       memory_mask=context2_mask)
                    c2_to_c1_to_c2 = self.context_to_context_attention.apply(is_train, x=context2_rep, keys=c2_to_c1,
                                                                             memories=c2_to_c1, x_mask=context2_mask,
                                                                             memory_mask=context1_mask)
                context2_rep = tf.add(c2_to_c1_to_c2, c1_to_c2)
                context1_rep = tf.add(c1_to_c2_to_c1, c2_to_c1)

        with tf.variable_scope("seq_enc"):
            fixed_rep1 = self.sequence_encoder.apply(is_train, context1_rep, context1_mask)
        with tf.variable_scope("seq_enc", reuse=True):
            fixed_rep2 = self.sequence_encoder.apply(is_train, context2_rep, context2_mask)

        with tf.variable_scope("merge"):
            fixed_rep = self.context_fixed_merge.apply(is_train, fixed_rep1, fixed_rep2)

        with tf.variable_scope("predictor"):
            return self.predictor.apply(is_train, fixed_rep, answer)


class BasicSingleContextAndQuestionIndependentModel(MultipleContextModel):
    """
    Model for a question and a single paragraph.
    Processes the question and context independently, gets a fixed size representation, and predicts.
    This model's purpose is to serve as a baseline for the task, as it is expected to perform poorly.
    """

    def __init__(self,
                 encoder: QuestionsAndParagraphsEncoder,
                 word_embed: Optional[WordEmbedder],
                 char_embed: Optional[CharWordEmbedder],
                 embed_mapper: Optional[Union[SequenceMapper, ElmoWrapper]],
                 sequence_encoder: SequenceEncoder,
                 merger: MergeLayer,
                 post_merger: Optional[Mapper],
                 predictor: BinaryFixedPredictor,
                 max_batch_size: Optional[int] = None,
                 elmo_model: Optional[LanguageModel] = None):
        super().__init__(encoder=encoder, word_embed=word_embed, char_embed=char_embed, max_batch_size=max_batch_size,
                         elmo_model=elmo_model)
        self.embed_mapper = embed_mapper
        self.sequence_encoder = sequence_encoder
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

        with tf.variable_scope("seq_enc"):
            context1_rep = self.sequence_encoder.apply(is_train, context1_rep, context1_mask)
            context1_rep = tf.identity(context1_rep, name='encode_context')
            tf.add_to_collection(INTERMEDIATE_LAYER_COLLECTION, context1_rep)
        with tf.variable_scope("seq_enc", reuse=True):
            question_rep = self.sequence_encoder.apply(is_train, question_rep, question_mask)

        with tf.variable_scope("merger"):
            merged_rep = self.merger.apply(is_train, question_rep, context1_rep)

        if self.post_merger is not None:
            with tf.variable_scope("post_merger"):
                merged_rep = self.post_merger.apply(is_train, merged_rep, mask=None)

        with tf.variable_scope("predictor"):
            return self.predictor.apply(is_train, merged_rep, answer)

    def __setstate__(self, state):
        if "post_merger" not in state:
            state["post_merger"] = None
        super().__setstate__(state)


class SingleFixedContextToQuestionModel(MultipleContextModel):
    """
    Model that encodes the context to a fixed size representation, then uses the question and that representation
    to get a prediction.
    """

    def __init__(self,
                 encoder: QuestionsAndParagraphsEncoder,
                 word_embed: Optional[WordEmbedder],
                 char_embed: Optional[CharWordEmbedder],
                 embed_mapper: Optional[SequenceMapper],
                 context_mapper: Optional[SequenceMapper],
                 context_encoder: SequenceEncoder,
                 question_mapper: Optional[SequenceMapper],
                 merger: FixedMergeLayer,
                 post_merger: Optional[SequenceMapper],
                 final_encoder: SequenceEncoder,
                 predictor: BinaryFixedPredictor,
                 max_batch_size: Optional[int] = None):
        super().__init__(encoder=encoder, word_embed=word_embed, char_embed=char_embed, max_batch_size=max_batch_size)
        self.embed_mapper = embed_mapper
        self.context_mapper = context_mapper
        self.context_encoder = context_encoder
        self.question_mapper = question_mapper
        self.merger = merger
        self.post_merger = post_merger
        self.final_encoder = final_encoder
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
        if self.embed_mapper is not None:
            with tf.variable_scope("map_embed"):
                context1_rep = self.embed_mapper.apply(is_train, context1_rep, context1_mask)
            with tf.variable_scope("map_embed", reuse=True):
                question_rep = self.embed_mapper.apply(is_train, question_rep, question_mask)

        if self.context_mapper is not None:
            with tf.variable_scope("context_mapper"):
                context1_rep = self.context_mapper.apply(is_train, context1_rep, context1_mask)

        with tf.variable_scope("context_enc"):
            context1_rep = self.context_encoder.apply(is_train, context1_rep, context1_mask)
            context1_rep = tf.identity(context1_rep, name='encode_context')
            tf.add_to_collection(INTERMEDIATE_LAYER_COLLECTION, context1_rep)

        if self.question_mapper is not None:
            with tf.variable_scope("question_mapper"):
                question_rep = self.question_mapper.apply(is_train, question_rep, question_mask)

        with tf.variable_scope("merger"):
            merged_rep = self.merger.apply(is_train, question_rep, fixed_tensor=context1_rep, mask=question_mask)

        if self.post_merger is not None:
            with tf.variable_scope("post_merger"):
                merged_rep = self.post_merger.apply(is_train, merged_rep, mask=question_mask)

        with tf.variable_scope("final_enc"):
            merged_rep = self.final_encoder.apply(is_train, merged_rep, mask=question_mask)

        with tf.variable_scope("predictor"):
            return self.predictor.apply(is_train, merged_rep, answer)


class SingleContextToQuestionModel(MultipleContextModel):
    """
    Model for a question with a single paragraph. supposed ot be similar to the multiple context ones.
    components are as follows:
    1. embeds each sequence separately
    2. creates question-aware attended representations of the context (optional)
    3. creates a context-aware question representation
    4. encodes the representation, and predicts
    """

    def __init__(self,
                 encoder: QuestionsAndParagraphsEncoder,
                 word_embed: Optional[WordEmbedder],
                 char_embed: Optional[CharWordEmbedder],
                 embed_mapper: Optional[SequenceMapper],
                 context_to_question_attention: AttentionMapper,
                 question_to_context_attention: Optional[AttentionWithPostMapper],
                 sequence_encoder: SequenceEncoder,
                 predictor: BinaryFixedPredictor,
                 max_batch_size: Optional[int] = None):
        super().__init__(encoder=encoder, word_embed=word_embed, char_embed=char_embed, max_batch_size=max_batch_size)
        self.embed_mapper = embed_mapper
        self.context_to_question_attention = context_to_question_attention
        self.question_to_context_attention = question_to_context_attention
        self.sequence_encoder = sequence_encoder
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
        if self.embed_mapper is not None:
            with tf.variable_scope("map_embed"):
                context1_rep = self.embed_mapper.apply(is_train, context1_rep, context1_mask)
                # TODO: This is how we are going to allow intermediate inputs. Do this in every place
                # we might want to inject a new input and discard the old layers of the network.
                # For now, leaving this here as a reference, and will use it with the relevant models.
                context1_rep = tf.identity(context1_rep, name='map_embed_context1')
                tf.add_to_collection('intermediate', context1_rep)
            with tf.variable_scope("map_embed", reuse=True):
                question_rep = self.embed_mapper.apply(is_train, question_rep, question_mask)

        if self.question_to_context_attention is not None:
            with tf.variable_scope("q2c"):
                context1_rep = self.question_to_context_attention.apply(is_train, x=context1_rep, keys=question_rep,
                                                                        memories=question_rep, x_mask=context1_mask,
                                                                        memory_mask=question_mask)

        with tf.variable_scope("c2q"):
            attended_rep = self.context_to_question_attention.apply(is_train, x=question_rep, keys=context1_rep,
                                                                    memories=context1_rep, x_mask=question_mask,
                                                                    memory_mask=context1_mask)

        with tf.variable_scope("seq_enc"):
            fixed_rep = self.sequence_encoder.apply(is_train, attended_rep, question_mask)

        with tf.variable_scope("predictor"):
            return self.predictor.apply(is_train, fixed_rep, answer)


class SingleContextWithBottleneckToQuestionModel(MultipleContextModel):
    """
    Model for a question with a single paragraph, but gets a fixed size representation of the context
    and uses it for prediction.
    components are as follows:
    1. embeds each sequence separately
    2. encodes the context to a fixed size representation
    2. creates question-aware attended representations of the context (optional)
    3. creates a context-aware question representation
    4. encodes the representation, concatenates the context rep from stage 2, and predicts
    """

    def __init__(self,
                 encoder: QuestionsAndParagraphsEncoder,
                 word_embed: Optional[WordEmbedder],
                 char_embed: Optional[CharWordEmbedder],
                 embed_mapper: Optional[SequenceMapper],
                 context_to_question_attention: AttentionMapper,
                 question_to_context_attention: Optional[AttentionWithPostMapper],
                 sequence_encoder: SequenceEncoder,
                 rep_merge: MergeLayer,
                 predictor: BinaryFixedPredictor,
                 max_batch_size: Optional[int] = None):
        super().__init__(encoder=encoder, word_embed=word_embed, char_embed=char_embed, max_batch_size=max_batch_size)
        self.embed_mapper = embed_mapper
        self.context_to_question_attention = context_to_question_attention
        self.question_to_context_attention = question_to_context_attention
        self.sequence_encoder = sequence_encoder
        self.rep_merge = rep_merge
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
        if self.embed_mapper is not None:
            with tf.variable_scope("map_embed"):
                context1_rep = self.embed_mapper.apply(is_train, context1_rep, context1_mask)
                context1_rep = tf.identity(context1_rep, name='sequence')
                tf.add_to_collection(INTERMEDIATE_LAYER_COLLECTION, context1_rep)
            with tf.variable_scope("map_embed", reuse=True):
                question_rep = self.embed_mapper.apply(is_train, question_rep, question_mask)
                question_rep = tf.identity(question_rep, name='question_sequence')
                tf.add_to_collection(INTERMEDIATE_LAYER_COLLECTION, question_rep)

        with tf.variable_scope('context_encode'):
            fixed_context = self.sequence_encoder.apply(is_train, context1_rep, context1_mask)
            fixed_context = tf.identity(fixed_context, name='fixed')
            tf.add_to_collection(INTERMEDIATE_LAYER_COLLECTION, fixed_context)

        if self.question_to_context_attention is not None:
            with tf.variable_scope("q2c"):
                context1_rep = self.question_to_context_attention.apply(is_train, x=context1_rep, keys=question_rep,
                                                                        memories=question_rep, x_mask=context1_mask,
                                                                        memory_mask=question_mask)

        with tf.variable_scope("c2q"):
            attended_rep = self.context_to_question_attention.apply(is_train, x=question_rep, keys=context1_rep,
                                                                    memories=context1_rep, x_mask=question_mask,
                                                                    memory_mask=context1_mask)

        with tf.variable_scope("seq_enc"):
            fixed_rep = self.sequence_encoder.apply(is_train, attended_rep, question_mask)

        with tf.variable_scope("rep_merge"):
            fixed_rep = self.rep_merge.apply(is_train, fixed_rep, fixed_context)

        with tf.variable_scope("predictor"):
            return self.predictor.apply(is_train, fixed_rep, answer)


class SingleContextBottleneckToSeqQuestionModel(MultipleContextModel):
    """
    Model for a question with a single paragraph, but gets a fixed size representation of the context
    and uses it to generate a new sequence which in turn will be regarded as the context in the classic models.
    components are as follows:
    1. embeds each sequence separately
    2. encodes the context to a fixed size representation
    3. generates a sequence from the fixed size representation
    TODO: allow attending on the question for creating the sequence
    2. creates question-aware attended representations of the context (optional)
    3. creates a context-aware question representation
    4. encodes the representation, concatenates the context rep from stage 2, and predicts
    """

    def __init__(self,
                 encoder: QuestionsAndParagraphsEncoder,
                 word_embed: Optional[WordEmbedder],
                 char_embed: Optional[CharWordEmbedder],
                 embed_mapper: Optional[SequenceMapper],
                 sequence_encoder: SequenceEncoder,
                 sequence_generator: VecToSeq,
                 pre_attention: Optional[SequenceMapper],
                 context_to_question_attention: AttentionMapper,
                 question_to_context_attention: Optional[AttentionWithPostMapper],
                 predictor: BinaryFixedPredictor,
                 max_batch_size: Optional[int] = None):
        super().__init__(encoder=encoder, word_embed=word_embed, char_embed=char_embed, max_batch_size=max_batch_size)
        self.embed_mapper = embed_mapper
        self.sequence_generator = sequence_generator
        self.pre_attention = pre_attention
        self.context_to_question_attention = context_to_question_attention
        self.question_to_context_attention = question_to_context_attention
        self.sequence_encoder = sequence_encoder
        self.predictor = predictor

    def _get_predictions_for(self,
                             is_train,
                             question_embed, question_mask,
                             context_embed, context_mask,
                             answer,
                             question_lm=None, context_lm=None, sentence_segments=None, sentence_mask=None):  # TODO: add elmo
        question_rep, context_rep = question_embed, context_embed
        context1_rep, = tf.unstack(context_rep, axis=1, num=1)
        context1_mask, = tf.unstack(context_mask, axis=1, num=1)
        if self.embed_mapper is not None:
            with tf.variable_scope("map_embed"):
                context1_rep = self.embed_mapper.apply(is_train, context1_rep, context1_mask)
                context1_rep = tf.identity(context1_rep, name='sequence')
                tf.add_to_collection(INTERMEDIATE_LAYER_COLLECTION, context1_rep)
            with tf.variable_scope("map_embed", reuse=True):
                question_rep = self.embed_mapper.apply(is_train, question_rep, question_mask)
                question_rep = tf.identity(question_rep, name='question_sequence')
                tf.add_to_collection(INTERMEDIATE_LAYER_COLLECTION, question_rep)

        with tf.variable_scope('context_encode'):
            fixed_context = self.sequence_encoder.apply(is_train, context1_rep, context1_mask)
            fixed_context = tf.identity(fixed_context, name='fixed')
            tf.add_to_collection(INTERMEDIATE_LAYER_COLLECTION, fixed_context)

        with tf.variable_scope('bottleneck_to_seq', reuse=tf.AUTO_REUSE):
            generated_context_rep = self.sequence_generator.apply(is_train, fixed_context)

        with tf.variable_scope('pre_att_question'):
            question_rep = self.pre_attention.apply(is_train, question_rep, question_mask)
        with tf.variable_scope('pre_att_context'):
            generated_context_rep = self.pre_attention.apply(is_train, generated_context_rep, mask=None)

        if self.question_to_context_attention is not None:
            with tf.variable_scope("q2c"):
                generated_context_rep = self.question_to_context_attention.apply(is_train, x=generated_context_rep,
                                                                                 keys=question_rep,
                                                                                 memories=question_rep, x_mask=None,
                                                                                 memory_mask=question_mask)

        with tf.variable_scope("c2q"):
            attended_rep = self.context_to_question_attention.apply(is_train, x=question_rep,
                                                                    keys=generated_context_rep,
                                                                    memories=generated_context_rep,
                                                                    x_mask=question_mask,
                                                                    memory_mask=None)

        with tf.variable_scope("seq_enc"):
            fixed_rep = self.sequence_encoder.apply(is_train, attended_rep, question_mask)

        with tf.variable_scope("predictor"):
            return self.predictor.apply(is_train, fixed_rep, answer)