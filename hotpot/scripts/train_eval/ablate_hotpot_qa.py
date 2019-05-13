import argparse

from datetime import datetime

from tensorflow.python import TruncatedNormal

from hotpot import model_dir, trainer
from hotpot.config import HOTPOT_ELMO_VOCAB, HOTPOT_ELMO_EMBEDDINGS, HOTPOT_ELMO_OPTIONS
from hotpot.data_handling.dataset import ClusteredBatcher, multiple_contexts_len
from hotpot.data_handling.hotpot.hotpot_data import HotpotQuestions
from hotpot.data_handling.hotpot.hotpot_qa_training_data import HotpotQATrainingData, HotpotQuestionFilterWithSpans, \
    HotpotTextLengthPreprocessorWithSpans
from hotpot.elmo.elmo import ElmoWrapper, ElmoLayer
from hotpot.elmo.lm_model import OriginalElmoModel
from hotpot.encoder import QuestionsAndParagraphsEncoder, GroupedSpanAnswerEncoder, GroupedSpanAnswerEncoderWithYesNo, \
    GroupedSpanAnswerEncoderFullHotpot
from hotpot.evaluator import LossEvaluator, MultiParagraphSpanEvaluator
from hotpot.models.single_context_qa_models import AttentionQA, AttentionQAWithYesNo, AttentionQAFullHotpot
from hotpot.nn.attention import StaticAttentionSelf, AttentionWithPostMapper, BiAttention
from hotpot.nn.embedder import FixedWordEmbedder, CharWordEmbedder, LearnedCharEmbedder
from hotpot.nn.layers import SequenceMapperSeq, VariationalDropoutLayer, DropoutLayer, MapperSeq, MaxPool, Conv1d, \
    WithConcatOptions, ResidualLayer, FullyConnected, ConcatWithProduct, NullBiMapper, ChainBiMapper, EncodeMap, \
    ActivationLayer
from hotpot.nn.ops import VERY_NEGATIVE_NUMBER
from hotpot.nn.recurrent_layers import CudnnGru
from hotpot.nn.sentence_layers import SentenceMaxEncoder
from hotpot.nn.similarity_layers import TriLinear
from hotpot.nn.span_prediction import BoundsPredictor, IndependentBoundsGrouped, IndependentBoundsGroupedWithYesNo
from hotpot.trainer import TrainParams, SerializableOptimizer, resume_training_with


def get_hotpot_elmo():
    return OriginalElmoModel(vocab_file=HOTPOT_ELMO_VOCAB, embeddings_file=HOTPOT_ELMO_EMBEDDINGS,
                             options_file=HOTPOT_ELMO_OPTIONS)


def get_model(rnn_dim: int, use_elmo):
    recurrent_layer = CudnnGru(rnn_dim, w_init=TruncatedNormal(stddev=0.05))

    embed_mapper = SequenceMapperSeq(
        VariationalDropoutLayer(0.8),
        recurrent_layer,
        VariationalDropoutLayer(0.8),
    )

    elmo_model = None
    if use_elmo:
        print("Using Elmo!")
        elmo_model = get_hotpot_elmo()
        lm_reduce = MapperSeq(
            ElmoLayer(0, layer_norm=False, top_layer_only=False),
            DropoutLayer(0.5),
        )
        embed_mapper = ElmoWrapper(input_append=True, output_append=False, rnn_layer=embed_mapper, lm_reduce=lm_reduce)

    answer_encoder = GroupedSpanAnswerEncoder(group=True)
    predictor = BoundsPredictor(
        ChainBiMapper(
            first_layer=recurrent_layer,
            second_layer=recurrent_layer
        ),
        span_predictor=IndependentBoundsGrouped(aggregate="sum")
    )

    return AttentionQA(
        encoder=QuestionsAndParagraphsEncoder(answer_encoder, use_sentence_segments=False),
        word_embed=FixedWordEmbedder(vec_name="glove.840B.300d", word_vec_init_scale=0, learn_unk=False, cpu=True),
        char_embed=CharWordEmbedder(
            LearnedCharEmbedder(word_size_th=14, char_th=50, char_dim=20, init_scale=0.05, force_cpu=True),
            MaxPool(Conv1d(100, 5, 0.8)),
            shared_parameters=True
        ),
        elmo_model=elmo_model,
        embed_mapper=embed_mapper,
        question_mapper=None,
        context_mapper=None,
        memory_builder=NullBiMapper(),
        attention=BiAttention(TriLinear(bias=True), True),
        match_encoder=SequenceMapperSeq(FullyConnected(rnn_dim * 2, activation="relu"),
                                        ResidualLayer(SequenceMapperSeq(
                                            VariationalDropoutLayer(0.8),
                                            recurrent_layer,
                                            VariationalDropoutLayer(0.8),
                                            StaticAttentionSelf(TriLinear(bias=True), ConcatWithProduct()),
                                            FullyConnected(rnn_dim * 2, activation="relu"),
                                        )),
                                        VariationalDropoutLayer(0.8)),
        predictor=predictor
    )


def get_model_with_yes_no(rnn_dim: int, use_elmo, keep_rate=0.8):
    recurrent_layer = CudnnGru(rnn_dim, w_init=TruncatedNormal(stddev=0.05))

    embed_mapper = SequenceMapperSeq(
        VariationalDropoutLayer(keep_rate),
        recurrent_layer,
        VariationalDropoutLayer(keep_rate),
    )

    elmo_model = None
    if use_elmo:
        print("Using Elmo!")
        elmo_model = get_hotpot_elmo()
        lm_reduce = MapperSeq(
            ElmoLayer(0, layer_norm=False, top_layer_only=False),
            DropoutLayer(0.5),
        )
        embed_mapper = ElmoWrapper(input_append=True, output_append=False, rnn_layer=embed_mapper, lm_reduce=lm_reduce)

    answer_encoder = GroupedSpanAnswerEncoderWithYesNo(group=True)
    predictor = BoundsPredictor(
        ChainBiMapper(
            first_layer=recurrent_layer,
            second_layer=recurrent_layer
        ),
        span_predictor=IndependentBoundsGroupedWithYesNo()
    )

    return AttentionQAWithYesNo(
        encoder=QuestionsAndParagraphsEncoder(answer_encoder, use_sentence_segments=False),
        word_embed=FixedWordEmbedder(vec_name="glove.840B.300d", word_vec_init_scale=0, learn_unk=False, cpu=True),
        char_embed=CharWordEmbedder(
            LearnedCharEmbedder(word_size_th=14, char_th=50, char_dim=20, init_scale=0.05, force_cpu=True),
            MaxPool(Conv1d(100, 5, 0.8)),
            shared_parameters=True
        ),
        elmo_model=elmo_model,
        embed_mapper=embed_mapper,
        question_mapper=None,
        context_mapper=None,
        memory_builder=NullBiMapper(),
        attention=BiAttention(TriLinear(bias=True), True),
        match_encoder=SequenceMapperSeq(FullyConnected(rnn_dim * 2, activation="relu"),
                                        ResidualLayer(SequenceMapperSeq(
                                            VariationalDropoutLayer(keep_rate),
                                            recurrent_layer,
                                            VariationalDropoutLayer(keep_rate),
                                            StaticAttentionSelf(TriLinear(bias=True), ConcatWithProduct()),
                                            FullyConnected(rnn_dim * 2, activation="relu"),
                                        )),
                                        VariationalDropoutLayer(keep_rate)),
        predictor=predictor,
        yes_no_question_encoder=MaxPool(map_layer=None, min_val=0, regular_reshape=True),
        yes_no_context_encoder=MaxPool(map_layer=None, min_val=0, regular_reshape=True)
    )


def get_full_hotpot_model(rnn_dim: int, use_elmo, keep_rate=0.8):
    recurrent_layer = CudnnGru(rnn_dim, w_init=TruncatedNormal(stddev=0.05))

    embed_mapper = SequenceMapperSeq(
        VariationalDropoutLayer(keep_rate),
        recurrent_layer,
    )

    elmo_model = None
    if use_elmo:
        print("Using Elmo!")
        elmo_model = get_hotpot_elmo()
        lm_reduce = MapperSeq(
            ElmoLayer(0, layer_norm=False, top_layer_only=False),
            DropoutLayer(0.5),
        )
        embed_mapper = ElmoWrapper(input_append=True, output_append=False, rnn_layer=embed_mapper, lm_reduce=lm_reduce)

    answer_encoder = GroupedSpanAnswerEncoderFullHotpot(group=True)
    predictor = BoundsPredictor(
        ChainBiMapper(
            first_layer=SequenceMapperSeq(VariationalDropoutLayer(keep_rate), recurrent_layer),
            second_layer=SequenceMapperSeq(VariationalDropoutLayer(keep_rate), recurrent_layer)
        ),
        span_predictor=IndependentBoundsGroupedWithYesNo()
    )

    return AttentionQAFullHotpot(
        encoder=QuestionsAndParagraphsEncoder(answer_encoder, use_sentence_segments=True,
                                              force_precomputed_sentences=True),
        word_embed=FixedWordEmbedder(vec_name="glove.840B.300d", word_vec_init_scale=0, learn_unk=False, cpu=True),
        char_embed=CharWordEmbedder(
            LearnedCharEmbedder(word_size_th=14, char_th=50, char_dim=20, init_scale=0.05, force_cpu=True),
            MaxPool(Conv1d(100, 5, 0.8)),
            shared_parameters=True
        ),
        elmo_model=elmo_model,
        embed_mapper=embed_mapper,
        question_mapper=VariationalDropoutLayer(keep_rate),
        context_mapper=VariationalDropoutLayer(keep_rate),
        memory_builder=NullBiMapper(),
        attention=BiAttention(TriLinear(bias=True), True),
        match_encoder=SequenceMapperSeq(FullyConnected(rnn_dim * 2, activation="relu"),
                                        ResidualLayer(SequenceMapperSeq(
                                            VariationalDropoutLayer(keep_rate),
                                            recurrent_layer,
                                            VariationalDropoutLayer(keep_rate),
                                            StaticAttentionSelf(TriLinear(bias=True), ConcatWithProduct()),
                                            FullyConnected(rnn_dim * 2, activation="relu"),
                                        ))),
        predictor=predictor,
        yes_no_question_encoder=EncodeMap(encoder=MaxPool(map_layer=None, min_val=VERY_NEGATIVE_NUMBER,
                                                          regular_reshape=True),
                                          mapper=None),
        yes_no_context_encoder=EncodeMap(encoder=MaxPool(map_layer=None, min_val=VERY_NEGATIVE_NUMBER,
                                                         regular_reshape=True),
                                         mapper=None),
        pre_sp_mapper=SequenceMapperSeq(VariationalDropoutLayer(keep_rate), recurrent_layer),
        sentences_encoder=SentenceMaxEncoder(),
        sentence_mapper=FullyConnected(rnn_dim, activation='relu')
    )


def main():
    parser = argparse.ArgumentParser(description='Train a model on the Hotpot QA dataset')
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

    # model = get_model_with_yes_no(rnn_dim=150, use_elmo=args.elmo, keep_rate=0.7)
    model = get_full_hotpot_model(rnn_dim=150, use_elmo=args.elmo, keep_rate=0.8)

    corpus = HotpotQuestions()
    train_batcher = ClusteredBatcher(25, multiple_contexts_len, truncate_batches=True)
    dev_batcher = ClusteredBatcher(90, multiple_contexts_len, truncate_batches=True)
    data = HotpotQATrainingData(corpus=corpus, train_batcher=train_batcher, dev_batcher=dev_batcher,
                                sample_filter=HotpotQuestionFilterWithSpans(1, keep_yes_no=True),
                                preprocessor=HotpotTextLengthPreprocessorWithSpans(600),
                                sample_train=None, sample_dev=None, sample_seed=18,
                                group_pairs_in_batches=True, distractor_pairs=2)

    eval = [LossEvaluator(), MultiParagraphSpanEvaluator(8, "hotpot", yes_no_option=True, supporting_facts_option=True)]

    n_epochs = 80

    eval_samples = dict(dev=None, train=3000)

    params = TrainParams(
        SerializableOptimizer("Adadelta", dict(learning_rate=1.0)),
        num_epochs=n_epochs, ema=0.999, max_checkpoints_to_keep=2,
        async_encoding=8, log_period=30, eval_period=1200, save_period=1200,
        eval_samples=eval_samples, best_weights=('dev', 'b8/question-text-f1'),
        monitor_gradients=True, clip_norm=None
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
    # os.environ["CUDA_VISIBLE_DEVICES"] = "7"
    main()
