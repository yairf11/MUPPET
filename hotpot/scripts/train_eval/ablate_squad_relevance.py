import argparse
from datetime import datetime
from typing import Optional, List

from keras.initializers import TruncatedNormal
import tensorflow as tf

from hotpot import model_dir, trainer
from hotpot.config import SQUAD_ELMO_VOCAB, SQUAD_ELMO_EMBEDDINGS, SQUAD_ELMO_FINETUNED_OPTIONS, \
    SQUAD_ELMO_FINETUNED_WEIGHTS, HOTPOT_WIKI_ELMO_VOCAB, HOTPOT_WIKI_ELMO_EMBEDDINGS, HOTPOT_WIKI_ELMO_OPTIONS, \
    SQUAD_ELMO_FINETUNED_WIKI_EMBEDDINGS
from hotpot.data_handling.dataset import ClusteredBatcher, multiple_contexts_len
from hotpot.data_handling.squad.squad_data import SquadRelevanceCorpus
from hotpot.data_handling.squad.squad_relevance_training_data import SquadBinaryRelevanceTrainingData, \
    SquadTextLengthPreprocessor
from hotpot.elmo.elmo import ElmoLayer, ElmoWrapper
from hotpot.elmo.lm_model import OriginalElmoModel, LanguageModel
from hotpot.encoder import QuestionsAndParagraphsEncoder, BinaryAnswerEncoder
from hotpot.evaluator import LossEvaluator, BinaryClassificationEvaluator
from hotpot.models.multiple_context_models import MultipleContextModel, BasicSingleContextAndQuestionIndependentModel, \
    SingleFixedContextToQuestionModel, SingleContextToQuestionModel, SingleContextWithBottleneckToQuestionModel, \
    SingleContextBottleneckToSeqQuestionModel
from hotpot.models.single_context_models import SingleContextMultipleEncodingModel, \
    SingleContextMultipleEncodingWeightedSoftmaxModel, SingleContextMaxSentenceModel
from hotpot.nn.attention import StaticAttentionSelf, BiAttention, AttentionWithPostMapper
from hotpot.nn.embedder import LearnedCharEmbedder, CharWordEmbedder, FixedWordEmbedder, WordEmbedder
from hotpot.nn.generative_layers import GenerativeRNN
from hotpot.nn.layers import ConcatWithProduct, MaxPool, VariationalDropoutLayer, SequenceMapperSeq, Conv1d, \
    FullyConnected, ResidualLayer, SequenceMapper, AttentionMapper, SequenceEncoder, DropoutLayer, MapperSeq, \
    WithConcatOptions, ConcatLayer, ConcatWithProductSub, MultiMapThenEncode, MultiMergeWeightedEncode, \
    MultiEncodingWeights
from hotpot.nn.ops import VERY_NEGATIVE_NUMBER
from hotpot.nn.recurrent_layers import CudnnGru, CudnnGruEncoder, CudnnLstm
from hotpot.nn.relevance_prediction import BinaryFixedPredictor, BinaryWeightedMultipleFixedPredictor, \
    BinaryNullPredictor
from hotpot.nn.sentence_layers import SentenceMaxEncoder
from hotpot.nn.similarity_layers import TriLinear
from hotpot.trainer import TrainParams, SerializableOptimizer, resume_training_with, \
    ExponentialDecayWrapper, ReduceLROnPlateau


def get_squad_elmo():
    return LanguageModel(lm_vocab_file=HOTPOT_WIKI_ELMO_VOCAB, embed_weights_file=SQUAD_ELMO_FINETUNED_WIKI_EMBEDDINGS,
                         options_file=HOTPOT_WIKI_ELMO_OPTIONS, weight_file=SQUAD_ELMO_FINETUNED_WEIGHTS)
    # return LanguageModel(lm_vocab_file=SQUAD_ELMO_VOCAB, embed_weights_file=SQUAD_ELMO_EMBEDDINGS,
    #                      options_file=SQUAD_ELMO_FINETUNED_OPTIONS, weight_file=SQUAD_ELMO_FINETUNED_WEIGHTS)
    # return OriginalElmoModel(vocab_file=SQUAD_ELMO_VOCAB, embeddings_file=SQUAD_ELMO_EMBEDDINGS)


def get_wiki_elmo():
    return OriginalElmoModel(vocab_file=HOTPOT_WIKI_ELMO_VOCAB, embeddings_file=HOTPOT_WIKI_ELMO_EMBEDDINGS,
                             options_file=HOTPOT_WIKI_ELMO_OPTIONS)


def get_mlp(layer_sizes: List[int], dropout=0.5, activation='relu'):
    layers = []
    for layer_size in layer_sizes:
        layers.append(FullyConnected(layer_size, activation=activation))
        layers.append(DropoutLayer(dropout))
    return MapperSeq(*layers)


def get_basic_model(rnn_dim, post_merger_params: Optional[dict] = None, use_elmo=False, keep_rate=0.8):
    recurrent_layer = CudnnGru(rnn_dim, w_init=TruncatedNormal(stddev=0.05))
    answer_encoder = BinaryAnswerEncoder()

    embed_mapper = SequenceMapperSeq(VariationalDropoutLayer(keep_rate), recurrent_layer)
    # embed_mapper = SequenceMapperSeq(
    #         SequenceMapperSeq(VariationalDropoutLayer(keep_rate), recurrent_layer),
    #         ResidualLayer(SequenceMapperSeq(VariationalDropoutLayer(keep_rate), recurrent_layer)),
    #         ResidualLayer(SequenceMapperSeq(VariationalDropoutLayer(keep_rate), recurrent_layer))
    #     )

    elmo_model = None
    if use_elmo:
        print("Using Elmo!")
        elmo_model = get_squad_elmo()
        lm_reduce = MapperSeq(
            ElmoLayer(0, layer_norm=False, top_layer_only=False),
            DropoutLayer(0.5),
        )
        embed_mapper = ElmoWrapper(input_append=True, output_append=True, rnn_layer=embed_mapper, lm_reduce=lm_reduce)

    post_merger = None if post_merger_params is None else get_mlp(**post_merger_params)

    return BasicSingleContextAndQuestionIndependentModel(
        encoder=QuestionsAndParagraphsEncoder(answer_encoder),
        word_embed=FixedWordEmbedder(vec_name="glove.840B.300d", word_vec_init_scale=0, learn_unk=False, cpu=True),
        char_embed=CharWordEmbedder(
            LearnedCharEmbedder(word_size_th=14, char_th=50, char_dim=20, init_scale=0.05, force_cpu=True),
            MaxPool(Conv1d(100, 5, 0.8)),
            shared_parameters=True
        ),
        elmo_model=elmo_model,
        embed_mapper=embed_mapper,
        sequence_encoder=MaxPool(map_layer=None, min_val=0, regular_reshape=True),
        merger=ConcatWithProductSub(),
        post_merger=post_merger,
        predictor=BinaryFixedPredictor(sigmoid=True),
        max_batch_size=128
    )


def get_res_fc_seq_fc(model_rnn_dim, rnn: bool, self_att: bool):
    seq_mapper = []
    if not rnn and not self_att:
        raise NotImplementedError()
    if rnn:
        seq_mapper.extend([VariationalDropoutLayer(0.8), CudnnGru(model_rnn_dim, w_init=TruncatedNormal(stddev=0.05))])
    if self_att:
        seq_mapper.extend([VariationalDropoutLayer(0.8),
                           StaticAttentionSelf(TriLinear(bias=True), ConcatWithProduct())])
    seq_mapper.append(FullyConnected(model_rnn_dim * 2, activation="relu"))
    return SequenceMapperSeq(FullyConnected(model_rnn_dim * 2, activation="relu"),
                             ResidualLayer(SequenceMapperSeq(
                                 *seq_mapper
                             )))


def get_context_to_question_model(rnn_dim: int, q2c: bool, res_rnn: bool, res_self_att: bool):
    recurrent_layer = CudnnGru(rnn_dim, w_init=TruncatedNormal(stddev=0.05))
    answer_encoder = BinaryAnswerEncoder()

    res_model = get_res_fc_seq_fc(model_rnn_dim=rnn_dim, rnn=res_rnn, self_att=res_self_att)

    question_to_context = \
        AttentionWithPostMapper(BiAttention(TriLinear(bias=True), True), post_mapper=res_model) if q2c else None
    context_to_question = AttentionWithPostMapper(BiAttention(TriLinear(bias=True), True), post_mapper=res_model)

    return SingleContextToQuestionModel(
        encoder=QuestionsAndParagraphsEncoder(answer_encoder),
        word_embed=FixedWordEmbedder(vec_name="glove.840B.300d", word_vec_init_scale=0, learn_unk=False, cpu=True),
        char_embed=CharWordEmbedder(
            LearnedCharEmbedder(word_size_th=14, char_th=50, char_dim=20, init_scale=0.05, force_cpu=True),
            MaxPool(Conv1d(100, 5, 0.8)),
            shared_parameters=True
        ),
        embed_mapper=SequenceMapperSeq(
            VariationalDropoutLayer(0.8),
            recurrent_layer,
            VariationalDropoutLayer(0.8),
        ),
        question_to_context_attention=question_to_context,
        context_to_question_attention=context_to_question,
        sequence_encoder=MaxPool(map_layer=None, min_val=0, regular_reshape=True),
        predictor=BinaryFixedPredictor()
    )


def get_context_with_bottleneck_to_question_model(rnn_dim: int, q2c: bool, res_rnn: bool, res_self_att: bool):
    recurrent_layer = CudnnGru(rnn_dim, w_init=TruncatedNormal(stddev=0.05))
    answer_encoder = BinaryAnswerEncoder()

    res_model = get_res_fc_seq_fc(model_rnn_dim=rnn_dim, rnn=res_rnn, self_att=res_self_att)

    question_to_context = \
        AttentionWithPostMapper(BiAttention(TriLinear(bias=True), True), post_mapper=res_model) if q2c else None
    context_to_question = AttentionWithPostMapper(BiAttention(TriLinear(bias=True), True), post_mapper=res_model)

    return SingleContextWithBottleneckToQuestionModel(
        encoder=QuestionsAndParagraphsEncoder(answer_encoder),
        word_embed=FixedWordEmbedder(vec_name="glove.840B.300d", word_vec_init_scale=0, learn_unk=False, cpu=True),
        char_embed=CharWordEmbedder(
            LearnedCharEmbedder(word_size_th=14, char_th=50, char_dim=20, init_scale=0.05, force_cpu=True),
            MaxPool(Conv1d(100, 5, 0.8)),
            shared_parameters=True
        ),
        embed_mapper=SequenceMapperSeq(
            VariationalDropoutLayer(0.8),
            recurrent_layer,
            VariationalDropoutLayer(0.8),
        ),
        question_to_context_attention=question_to_context,
        context_to_question_attention=context_to_question,
        sequence_encoder=CudnnGruEncoder(rnn_dim, w_init=TruncatedNormal(stddev=0.05)),
        # MaxPool(map_layer=None, min_val=0, regular_reshape=True),
        rep_merge=ConcatLayer(),
        predictor=BinaryFixedPredictor()
    )


def get_fixed_context_to_question(rnn_dim):
    recurrent_layer = CudnnGru(rnn_dim, w_init=TruncatedNormal(stddev=0.05))
    answer_encoder = BinaryAnswerEncoder()

    return SingleFixedContextToQuestionModel(
        encoder=QuestionsAndParagraphsEncoder(answer_encoder),
        word_embed=FixedWordEmbedder(vec_name="glove.840B.300d", word_vec_init_scale=0, learn_unk=False, cpu=True),
        char_embed=CharWordEmbedder(
            LearnedCharEmbedder(word_size_th=14, char_th=50, char_dim=20, init_scale=0.05, force_cpu=True),
            MaxPool(Conv1d(100, 5, 0.8)),
            shared_parameters=True
        ),
        embed_mapper=SequenceMapperSeq(
            SequenceMapperSeq(VariationalDropoutLayer(0.8), recurrent_layer),
            ResidualLayer(SequenceMapperSeq(VariationalDropoutLayer(0.8), recurrent_layer)),
            ResidualLayer(SequenceMapperSeq(VariationalDropoutLayer(0.8), recurrent_layer))
        ),
        context_mapper=None,
        # ResidualLayer(
        #     SequenceMapperSeq(
        #         VariationalDropoutLayer(0.8),
        #         StaticAttentionSelf(TriLinear(bias=True), ConcatWithProduct()),
        #         FullyConnected(rnn_dim*2, activation=None))),
        context_encoder=MaxPool(map_layer=None, min_val=0, regular_reshape=True),
        question_mapper=None,
        # ResidualLayer(
        #     SequenceMapperSeq(
        #         VariationalDropoutLayer(0.8),
        #         StaticAttentionSelf(TriLinear(bias=True), ConcatWithProduct()),
        #         FullyConnected(rnn_dim*2, activation=None))),
        merger=WithConcatOptions(dot=True, sub=True, hadamard=True, raw=True, project=False),
        post_merger=SequenceMapperSeq(
            FullyConnected(rnn_dim * 2, activation='relu'),
            ResidualLayer(SequenceMapperSeq(VariationalDropoutLayer(0.8), recurrent_layer,
                                            FullyConnected(rnn_dim * 2, activation='relu')))
        ),
        final_encoder=MaxPool(map_layer=None, min_val=0, regular_reshape=True),
        predictor=BinaryFixedPredictor()
    )


class CustomLayer(SequenceMapper):
    def __init__(self):
        pass

    def apply(self, is_train, x, mask=None):
        pass


def get_bottleneck_to_seq_model(rnn_dim, q2c: bool, res_rnn: bool, res_self_att: bool, seq_len=50):
    recurrent_layer = CudnnGru(rnn_dim, w_init=TruncatedNormal(stddev=0.05))
    answer_encoder = BinaryAnswerEncoder()

    res_model = get_res_fc_seq_fc(model_rnn_dim=rnn_dim, rnn=res_rnn, self_att=res_self_att)

    question_to_context = \
        AttentionWithPostMapper(BiAttention(TriLinear(bias=True), True), post_mapper=res_model) if q2c else None
    context_to_question = AttentionWithPostMapper(BiAttention(TriLinear(bias=True), True), post_mapper=res_model)

    sequence_generator = GenerativeRNN(tf.contrib.rnn.LSTMCell(num_units=rnn_dim,
                                                               initializer=tf.initializers.truncated_normal(
                                                                   stddev=0.05)),
                                       output_layer=FullyConnected(rnn_dim * 2, activation='relu'),
                                       vec_to_in=FullyConnected(rnn_dim * 2, activation='relu'),
                                       seq_len=seq_len, include_original_vec=False)

    return SingleContextBottleneckToSeqQuestionModel(
        encoder=QuestionsAndParagraphsEncoder(answer_encoder),
        word_embed=FixedWordEmbedder(vec_name="glove.840B.300d", word_vec_init_scale=0, learn_unk=False, cpu=True),
        char_embed=CharWordEmbedder(
            LearnedCharEmbedder(word_size_th=14, char_th=50, char_dim=20, init_scale=0.05, force_cpu=True),
            MaxPool(Conv1d(100, 5, 0.8)),
            shared_parameters=True
        ),
        embed_mapper=SequenceMapperSeq(
            VariationalDropoutLayer(0.8),
            recurrent_layer,
        ),
        sequence_generator=sequence_generator,
        pre_attention=VariationalDropoutLayer(0.8),
        question_to_context_attention=question_to_context,
        context_to_question_attention=context_to_question,
        sequence_encoder=MaxPool(map_layer=None, min_val=0, regular_reshape=True),
        predictor=BinaryFixedPredictor()
    )


def get_multi_encode_model(rnn_dim, multi_rnn_dim, num_encodings, keep_rate=0.8, map_embed=True):
    recurrent_layer = CudnnGru(rnn_dim, w_init=TruncatedNormal(stddev=0.05))
    multi_recurrent_layer = CudnnGru(multi_rnn_dim, w_init=TruncatedNormal(stddev=0.05))
    answer_encoder = BinaryAnswerEncoder()

    return SingleContextMultipleEncodingModel(
        encoder=QuestionsAndParagraphsEncoder(answer_encoder),
        word_embed=FixedWordEmbedder(vec_name="glove.840B.300d", word_vec_init_scale=0, learn_unk=False, cpu=True),
        char_embed=CharWordEmbedder(
            LearnedCharEmbedder(word_size_th=14, char_th=50, char_dim=20, init_scale=0.05, force_cpu=True),
            MaxPool(Conv1d(100, 5, 0.8)),
            shared_parameters=True
        ),
        embed_mapper=SequenceMapperSeq(
            VariationalDropoutLayer(keep_rate),
            recurrent_layer,
        ) if map_embed else None,
        sequence_multi_encoder=MultiMapThenEncode(
            mapper=SequenceMapperSeq(VariationalDropoutLayer(keep_rate), multi_recurrent_layer),
            encoder=MaxPool(map_layer=None, min_val=0, regular_reshape=True),
            num_encodings=num_encodings
        ),
        merger_encoder=MultiMergeWeightedEncode(
            merge=ConcatWithProduct(), weight_mode='fully_connected', weight_context=False, encode='concat'
        ),
        post_merger=None,
        predictor=BinaryFixedPredictor()
    )


def get_multi_encode_softmax_weighting_model(rnn_dim, multi_rnn_dim, num_encodings, keep_rate=0.8, map_embed=True):
    recurrent_layer = CudnnGru(rnn_dim, w_init=TruncatedNormal(stddev=0.05))
    multi_recurrent_layer = CudnnGru(multi_rnn_dim, w_init=TruncatedNormal(stddev=0.05))
    answer_encoder = BinaryAnswerEncoder()

    return SingleContextMultipleEncodingWeightedSoftmaxModel(
        encoder=QuestionsAndParagraphsEncoder(answer_encoder),
        word_embed=FixedWordEmbedder(vec_name="glove.840B.300d", word_vec_init_scale=0, learn_unk=False, cpu=True),
        char_embed=CharWordEmbedder(
            LearnedCharEmbedder(word_size_th=14, char_th=50, char_dim=20, init_scale=0.05, force_cpu=True),
            MaxPool(Conv1d(100, 5, 0.8)),
            shared_parameters=True
        ),
        embed_mapper=SequenceMapperSeq(
            VariationalDropoutLayer(keep_rate),
            recurrent_layer,
        ) if map_embed else None,
        sequence_multi_encoder=MultiMapThenEncode(
            mapper=SequenceMapperSeq(VariationalDropoutLayer(keep_rate), multi_recurrent_layer),
            encoder=MaxPool(map_layer=None, min_val=0, regular_reshape=True),
            num_encodings=num_encodings
        ),
        weight_layer=MultiEncodingWeights(weight_mode='mlp'),
        merger=ConcatWithProduct(),
        post_merger=None,
        predictor=BinaryWeightedMultipleFixedPredictor()
    )


def get_sentences_model(rnn_dim, use_elmo, keep_rate=0.8):
    recurrent_layer = CudnnGru(rnn_dim, w_init=TruncatedNormal(stddev=0.05))
    answer_encoder = BinaryAnswerEncoder()

    embed_mapper = SequenceMapperSeq(VariationalDropoutLayer(keep_rate), recurrent_layer)

    elmo_model = None
    if use_elmo:
        print("Using Elmo!")
        elmo_model = get_wiki_elmo()
        lm_reduce = MapperSeq(
            ElmoLayer(0, layer_norm=False, top_layer_only=False),
            DropoutLayer(0.5),
        )
        embed_mapper = ElmoWrapper(input_append=True, output_append=False, rnn_layer=embed_mapper, lm_reduce=lm_reduce)

    return SingleContextMaxSentenceModel(
        encoder=QuestionsAndParagraphsEncoder(answer_encoder, use_sentence_segments=True, paragraph_as_sentence=True),
        word_embed=FixedWordEmbedder(vec_name="glove.840B.300d", word_vec_init_scale=0, learn_unk=False, cpu=True),
        char_embed=CharWordEmbedder(
            LearnedCharEmbedder(word_size_th=14, char_th=50, char_dim=20, init_scale=0.05, force_cpu=True),
            MaxPool(Conv1d(100, 5, 0.8)),
            shared_parameters=True
        ),
        elmo_model=elmo_model,
        embed_mapper=embed_mapper,
        sequence_encoder=MaxPool(map_layer=None, min_val=VERY_NEGATIVE_NUMBER, regular_reshape=True),
        sentences_encoder=SentenceMaxEncoder(),
        post_merger=None,
        merger=WithConcatOptions(sub=False, hadamard=True, dot=True, raw=True),
        max_batch_size=256
    )


def main():
    parser = argparse.ArgumentParser(description='Train a model on the Squad relevance dataset')
    parser.add_argument("name", help="Where to store the model")
    parser.add_argument("--elmo", action='store_true', help="Whether to use elmo or not")
    parser.add_argument("-c", "--continue_model", action='store_true', help="Whether to start a new run or "
                                                                            "continue an existing one")
    args = parser.parse_args()

    with open(__file__, "r") as f:
        notes = f.read()

    continue_existing_run = args.continue_model
    # save_preprocessed = args.save
    if continue_existing_run:
        print("We will continue an existing run!")
    else:
        print("We will start a new run!")

    out = args.name + "-" + datetime.now().strftime("%m%d-%H%M%S")
    if continue_existing_run:
        out = args.name

    # model = get_basic_model(500, post_merger_params=None, use_elmo=args.elmo, keep_rate=0.8)
    # model = get_context_to_question_model(rnn_dim=150, q2c=False, res_rnn=True, res_self_att=False)
    # model = get_context_with_bottleneck_to_question_model(rnn_dim=500, q2c=False, res_rnn=True, res_self_att=False)
    # model = get_ablate_model()
    # model = get_fixed_context_to_question(150)
    # model = get_bottleneck_to_seq_model(500, q2c=False, res_rnn=True, res_self_att=False, seq_len=50)
    # model = get_multi_encode_model(0, 200, num_encodings=5, map_embed=False)
    # model = get_multi_encode_softmax_weighting_model(0, 400, num_encodings=5, map_embed=False)
    model = get_sentences_model(512, use_elmo=args.elmo, keep_rate=0.8)

    corpus = SquadRelevanceCorpus()
    train_batcher = ClusteredBatcher(45, multiple_contexts_len, truncate_batches=True)
    dev_batcher = ClusteredBatcher(128, multiple_contexts_len, truncate_batches=True)
    data = SquadBinaryRelevanceTrainingData(corpus=corpus, train_batcher=train_batcher, dev_batcher=dev_batcher,
                                            sample_filter=None, preprocessor=SquadTextLengthPreprocessor(600),
                                            sample_train=None, sample_dev=None, sample_seed=18)

    eval = [LossEvaluator(), BinaryClassificationEvaluator()]

    n_epochs = 80

    adadelta = SerializableOptimizer("Adadelta", dict(learning_rate=1.0))
    momentum = SerializableOptimizer("Momentum", dict(learning_rate=0.01, momentum=0.9, use_nesterov=True))
    adam = SerializableOptimizer("Adam", dict(learning_rate=1e-4))

    reduce_lr_on_plateau = ReduceLROnPlateau(dev_name='dev', scalar_name='loss', factor=0.2,
                                             patience=8, verbose=1, mode='min', terminate_th=1e-5)

    params = TrainParams(
        adadelta,
        num_epochs=n_epochs, ema=0.999, max_checkpoints_to_keep=2,
        async_encoding=8, log_period=30, eval_period=1800, save_period=1800,
        eval_samples=dict(dev=None, train=3000), best_weights=('dev', 'binary-relevance/average_precision'),
        monitor_gradients=True, clip_norm=None, regularization_lambda=None, reduce_lr_on_plateau=None
    )

    if not continue_existing_run:
        trainer.start_training(data, model, params, eval, model_dir.ModelDir(out), notes, save_graph=False)
    else:
        resume_training_with(data=data, out=model_dir.ModelDir(out),
                             train_params=params, evaluators=eval, notes=notes, start_eval=True)


if __name__ == "__main__":
    # import os
    #
    # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
    # os.environ["CUDA_VISIBLE_DEVICES"] = "6"
    main()
