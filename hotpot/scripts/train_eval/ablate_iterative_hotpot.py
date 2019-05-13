import argparse

from datetime import datetime

from tensorflow.python import TruncatedNormal

from hotpot import model_dir, trainer
from hotpot.config import HOTPOT_WIKI_ELMO_VOCAB, \
    HOTPOT_WIKI_ELMO_EMBEDDINGS, HOTPOT_WIKI_ELMO_OPTIONS
from hotpot.data_handling.dataset import ClusteredBatcher, multiple_contexts_len
from hotpot.data_handling.hotpot.hotpot_data import HotpotQuestions
from hotpot.data_handling.hotpot.hotpot_relevance_training_data import HotpotTextLengthPreprocessor, \
    HotpotQuestionFilter, HotpotIterativeRelevanceTrainingData
from hotpot.elmo.elmo import ElmoWrapper, ElmoLayer
from hotpot.elmo.lm_model import OriginalElmoModel
from hotpot.encoder import IterativeAnswerEncoder, QuestionsAndParagraphsEncoder
from hotpot.evaluator import LossEvaluator, IterativeRelevanceEvaluator
from hotpot.models.iterative_context_models import IterativeContextMaxSentenceModel, IterativeContextReReadModel, \
    IterativeContextReReadMergeModel, IterativeContextReReadSimpleScoreModel
from hotpot.nn.attention import StaticAttentionSelf, AttentionWithPostMapper, BiAttention
from hotpot.nn.embedder import FixedWordEmbedder, CharWordEmbedder, LearnedCharEmbedder
from hotpot.nn.layers import SequenceMapperSeq, VariationalDropoutLayer, DropoutLayer, MapperSeq, MaxPool, Conv1d, \
    WithConcatOptions, ResidualLayer, FullyConnected, ConcatWithProduct, ActivationLayer, EncodeMap
from hotpot.nn.ops import VERY_NEGATIVE_NUMBER
from hotpot.nn.recurrent_layers import CudnnGru, CudnnGruEncoder
from hotpot.nn.reformulation_layers import WeightedSumThenProjectReformulation, ProjectMapEncodeReformulation, \
    ProjectThenWeightedSumReformulation
from hotpot.nn.relevance_prediction import BinaryNullPredictor
from hotpot.nn.sentence_layers import SentenceMaxEncoder
from hotpot.nn.similarity_layers import TriLinear
from hotpot.trainer import TrainParams, SerializableOptimizer, resume_training_with


def get_hotpot_elmo():
    return OriginalElmoModel(vocab_file=HOTPOT_WIKI_ELMO_VOCAB, embeddings_file=HOTPOT_WIKI_ELMO_EMBEDDINGS,
                             options_file=HOTPOT_WIKI_ELMO_OPTIONS)


def get_model(rnn_dim, use_elmo, keep_rate=0.8):
    recurrent_layer = CudnnGru(rnn_dim, w_init=TruncatedNormal(stddev=0.05))
    answer_encoder = IterativeAnswerEncoder()

    embed_mapper = SequenceMapperSeq(VariationalDropoutLayer(keep_rate), recurrent_layer)

    elmo_model = None
    if use_elmo:
        print("Using Elmo!")
        elmo_model = get_hotpot_elmo()
        lm_reduce = MapperSeq(
            ElmoLayer(0, layer_norm=False, top_layer_only=False),
            DropoutLayer(0.5),
        )
        embed_mapper = ElmoWrapper(input_append=True, output_append=False, rnn_layer=embed_mapper, lm_reduce=lm_reduce)

    reformulation = ProjectMapEncodeReformulation(project_layer=None,
                                                  sequence_mapper=None,
                                                  encoder=CudnnGruEncoder(rnn_dim, w_init=TruncatedNormal(stddev=0.05)))
    # reformulation = WeightedSumThenProjectReformulation(rnn_dim*2, activation='relu')
    # reformulation = ProjectThenWeightedSumReformulation(rnn_dim*2, activation='relu')

    return IterativeContextMaxSentenceModel(
        encoder=QuestionsAndParagraphsEncoder(answer_encoder, use_sentence_segments=True),
        word_embed=FixedWordEmbedder(vec_name="glove.840B.300d", word_vec_init_scale=0, learn_unk=False, cpu=True),
        char_embed=CharWordEmbedder(
            LearnedCharEmbedder(word_size_th=14, char_th=50, char_dim=20, init_scale=0.05, force_cpu=True),
            MaxPool(Conv1d(100, 5, 0.8)),
            shared_parameters=True
        ),
        elmo_model=elmo_model,
        embed_mapper=embed_mapper,
        sequence_encoder=MaxPool(map_layer=None, min_val=0, regular_reshape=True),
        sentences_encoder=SentenceMaxEncoder(),
        sentence_mapper=recurrent_layer,
        post_merger=None,
        merger=WithConcatOptions(sub=False, hadamard=True, dot=True, raw=True),
        reformulation_layer=reformulation,
        max_batch_size=128
    )


def get_res_fc_seq_fc(model_rnn_dim, rnn: bool, self_att: bool, keep_rate=0.8):
    seq_mapper = []
    if not rnn and not self_att:
        raise NotImplementedError()
    if rnn:
        seq_mapper.extend([VariationalDropoutLayer(keep_rate),
                           CudnnGru(model_rnn_dim, w_init=TruncatedNormal(stddev=0.05))])
    if self_att:
        seq_mapper.extend([VariationalDropoutLayer(keep_rate),
                           StaticAttentionSelf(TriLinear(bias=True), ConcatWithProduct())])
    seq_mapper.append(FullyConnected(model_rnn_dim * 2, activation="relu"))
    return SequenceMapperSeq(FullyConnected(model_rnn_dim * 2, activation="relu"),
                             ResidualLayer(SequenceMapperSeq(
                                 *seq_mapper
                             )))


def get_reread_model(rnn_dim, use_elmo, encoder_keep_rate=0.8, reread_keep_rate=0.8,
                     two_phase_att=False, res_rnn=True, res_self_att=False,
                     multiply_iteration_probs=False, reformulate_by_context=False,
                     rank_first=False, rank_second=False, reread_rnn_dim=None,
                     first_rank_lambda=1.0, second_rank_lambda=1.0,
                     ranking_gamma=1.0):
    recurrent_layer = CudnnGru(rnn_dim, w_init=TruncatedNormal(stddev=0.05))
    answer_encoder = IterativeAnswerEncoder(group=rank_first or rank_second)

    embed_mapper = SequenceMapperSeq(VariationalDropoutLayer(encoder_keep_rate), recurrent_layer)

    if res_rnn or res_self_att:
        res_model = get_res_fc_seq_fc(model_rnn_dim=rnn_dim, rnn=res_rnn, self_att=res_self_att,
                                      keep_rate=reread_keep_rate)
    else:
        res_model = FullyConnected(rnn_dim * 2, activation="relu")
    attention = AttentionWithPostMapper(BiAttention(TriLinear(bias=True), True),
                                        post_mapper=res_model)
    use_c2q = two_phase_att or not reformulate_by_context
    use_q2c = two_phase_att or reformulate_by_context

    elmo_model = None
    if use_elmo:
        print("Using Elmo!")
        elmo_model = get_hotpot_elmo()
        lm_reduce = MapperSeq(
            ElmoLayer(0, layer_norm=False, top_layer_only=False),
            DropoutLayer(0.5),
        )
        embed_mapper = ElmoWrapper(input_append=True, output_append=False, rnn_layer=embed_mapper, lm_reduce=lm_reduce)

    return IterativeContextReReadModel(
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
        sentence_mapper=None,
        post_merger=None,
        merger=WithConcatOptions(sub=False, hadamard=True, dot=True, raw=True),
        reread_mapper=None if reread_rnn_dim is None else CudnnGru(reread_rnn_dim, w_init=TruncatedNormal(stddev=0.05)),
        pre_attention_mapper=None,  # VariationalDropoutLayer(reread_keep_rate),
        context_to_question_attention=attention if use_c2q else None,
        question_to_context_attention=attention if use_q2c else None,
        reformulate_by_context=reformulate_by_context,
        multiply_iteration_probs=multiply_iteration_probs,
        first_predictor=BinaryNullPredictor(rank_first, ranking_lambda=first_rank_lambda, gamma=ranking_gamma),
        second_predictor=BinaryNullPredictor(rank_second, ranking_lambda=second_rank_lambda, gamma=ranking_gamma),
        max_batch_size=512
    )


def get_reread_simple_score(rnn_dim, use_elmo, keep_rate=0.8, two_phase_att=False, res_rnn=True, res_self_att=False,
                            reformulate_by_context=False, rank_first=False, rank_second=False, reread_rnn_dim=None):
    recurrent_layer = CudnnGru(rnn_dim, w_init=TruncatedNormal(stddev=0.05))
    answer_encoder = IterativeAnswerEncoder(group=rank_first or rank_second)

    embed_mapper = SequenceMapperSeq(VariationalDropoutLayer(keep_rate), recurrent_layer)

    if res_rnn or res_self_att:
        res_model = get_res_fc_seq_fc(model_rnn_dim=rnn_dim, rnn=res_rnn, self_att=res_self_att, keep_rate=keep_rate)
    else:
        res_model = FullyConnected(rnn_dim * 2, activation="relu")
    attention = AttentionWithPostMapper(BiAttention(TriLinear(bias=True), True),
                                        post_mapper=res_model)
    use_c2q = two_phase_att or not reformulate_by_context
    use_q2c = two_phase_att or reformulate_by_context

    elmo_model = None
    if use_elmo:
        print("Using Elmo!")
        elmo_model = get_hotpot_elmo()
        lm_reduce = MapperSeq(
            ElmoLayer(0, layer_norm=False, top_layer_only=False),
            DropoutLayer(0.5),
        )
        embed_mapper = ElmoWrapper(input_append=True, output_append=False, rnn_layer=embed_mapper, lm_reduce=lm_reduce)

    return IterativeContextReReadSimpleScoreModel(
        encoder=QuestionsAndParagraphsEncoder(answer_encoder, use_sentence_segments=True),
        word_embed=FixedWordEmbedder(vec_name="glove.840B.300d", word_vec_init_scale=0, learn_unk=False, cpu=True),
        char_embed=CharWordEmbedder(
            LearnedCharEmbedder(word_size_th=14, char_th=50, char_dim=20, init_scale=0.05, force_cpu=True),
            MaxPool(Conv1d(100, 5, 0.8)),
            shared_parameters=True
        ),
        elmo_model=elmo_model,
        embed_mapper=embed_mapper,
        sequence_encoder=EncodeMap(encoder=MaxPool(map_layer=None, min_val=VERY_NEGATIVE_NUMBER, regular_reshape=True),
                                   mapper=FullyConnected(rnn_dim * 2, activation=None)),
        sentences_encoder=SentenceMaxEncoder(),
        sentence_mapper=FullyConnected(rnn_dim * 2, activation=None),
        reread_mapper=None if reread_rnn_dim is None else CudnnGru(reread_rnn_dim, w_init=TruncatedNormal(stddev=0.05)),
        pre_attention_mapper=VariationalDropoutLayer(keep_rate),
        context_to_question_attention=attention if use_c2q else None,
        question_to_context_attention=attention if use_q2c else None,
        reformulate_by_context=reformulate_by_context,
        first_predictor=BinaryNullPredictor(rank_first),
        second_predictor=BinaryNullPredictor(rank_second),
        max_batch_size=512
    )


def get_reread_merge_model(rnn_dim, use_elmo, keep_rate=0.8, res_rnn=True, res_self_att=False,
                           multiply_iteration_probs=False):
    recurrent_layer = CudnnGru(rnn_dim, w_init=TruncatedNormal(stddev=0.05))
    answer_encoder = IterativeAnswerEncoder()

    embed_mapper = SequenceMapperSeq(VariationalDropoutLayer(keep_rate), recurrent_layer)

    if res_rnn or res_self_att:
        res_model = get_res_fc_seq_fc(model_rnn_dim=rnn_dim, rnn=res_rnn, self_att=res_self_att)
    else:
        res_model = FullyConnected(rnn_dim * 2, activation="relu")
    attention = AttentionWithPostMapper(BiAttention(TriLinear(bias=True), True), post_mapper=res_model)

    elmo_model = None
    if use_elmo:
        print("Using Elmo!")
        elmo_model = get_hotpot_elmo()
        lm_reduce = MapperSeq(
            ElmoLayer(0, layer_norm=False, top_layer_only=False),
            DropoutLayer(0.5),
        )
        embed_mapper = ElmoWrapper(input_append=True, output_append=False, rnn_layer=embed_mapper, lm_reduce=lm_reduce)

    return IterativeContextReReadMergeModel(
        encoder=QuestionsAndParagraphsEncoder(answer_encoder, use_sentence_segments=True),
        word_embed=FixedWordEmbedder(vec_name="glove.840B.300d", word_vec_init_scale=0, learn_unk=False, cpu=True),
        char_embed=CharWordEmbedder(
            LearnedCharEmbedder(word_size_th=14, char_th=50, char_dim=20, init_scale=0.05, force_cpu=True),
            MaxPool(Conv1d(100, 5, 0.8)),
            shared_parameters=True
        ),
        elmo_model=elmo_model,
        embed_mapper=embed_mapper,
        sequence_encoder=MaxPool(map_layer=None, min_val=0, regular_reshape=True),
        sentences_encoder=SentenceMaxEncoder(),
        sentence_mapper=None,
        post_merger=None,
        merger=WithConcatOptions(sub=False, hadamard=True, dot=False, raw=True),
        context_to_question_attention=attention,
        question_to_context_attention=attention,
        reread_merger=ConcatWithProduct(),
        multiply_iteration_probs=multiply_iteration_probs,
        max_batch_size=128
    )


def main():
    parser = argparse.ArgumentParser(description='Train a model on the Hotpot iterative relevance dataset')
    parser.add_argument("name", help="Where to store the model")
    parser.add_argument("--elmo", action='store_true', help="Whether to use elmo or not")
    parser.add_argument("--label-method", choices=["br-as-cp", "span", 'tfidf'], default="tfidf")
    parser.add_argument("--rank", action='store_true', help="Whether to use ranking loss or not")
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

    print(f"Labeling method: {args.label_method}")

    # model = get_model(rnn_dim=500, use_elmo=args.elmo, keep_rate=0.8)
    model = get_reread_model(rnn_dim=512, use_elmo=args.elmo, encoder_keep_rate=0.8, reread_keep_rate=0.8,
                             two_phase_att=False, res_rnn=True, res_self_att=False,
                             multiply_iteration_probs=False, reformulate_by_context=False,
                             rank_first=True, first_rank_lambda=1.0,
                             rank_second=True, second_rank_lambda=1.0,
                             reread_rnn_dim=None, ranking_gamma=1.0)
    # model = get_reread_simple_score(rnn_dim=512, use_elmo=args.elmo, keep_rate=0.8,
    #                                 two_phase_att=False, res_rnn=True, res_self_att=False, reformulate_by_context=False,
    #                                 rank_first=True,
    #                                 rank_second=True,
    #                                 reread_rnn_dim=None)
    # model = get_reread_merge_model(rnn_dim=512, use_elmo=args.elmo, keep_rate=0.8,
    #                                res_rnn=True, res_self_att=False,
    #                                multiply_iteration_probs=False)

    corpus = HotpotQuestions()
    if not args.rank:
        train_batcher = ClusteredBatcher(45, multiple_contexts_len, truncate_batches=True)
        dev_batcher = ClusteredBatcher(90, multiple_contexts_len, truncate_batches=True)
        data = HotpotIterativeRelevanceTrainingData(corpus=corpus, train_batcher=train_batcher, dev_batcher=dev_batcher,
                                                    sample_filter=HotpotQuestionFilter(2),
                                                    preprocessor=HotpotTextLengthPreprocessor(600),
                                                    sample_train=None, sample_dev=None, sample_seed=18,
                                                    bridge_as_comparison=args.label_method == 'br-as-cp',
                                                    label_by_span=args.label_method == 'span')
    else:
        train_batcher = ClusteredBatcher(25, multiple_contexts_len, truncate_batches=True)
        dev_batcher = ClusteredBatcher(75, multiple_contexts_len, truncate_batches=True)
        data = HotpotIterativeRelevanceTrainingData(corpus=corpus, train_batcher=train_batcher, dev_batcher=dev_batcher,
                                                    sample_filter=HotpotQuestionFilter(2),
                                                    preprocessor=HotpotTextLengthPreprocessor(600),
                                                    sample_train=None, sample_dev=None, sample_seed=18,
                                                    bridge_as_comparison=args.label_method == 'br-as-cp',
                                                    group_pairs_in_batches=True,
                                                    label_by_span=args.label_method == 'span',
                                                    num_distractors_in_batch=2,
                                                    max_batch_size=model.max_batch_size)

    eval = [LossEvaluator(), IterativeRelevanceEvaluator()]

    n_epochs = 80

    eval_samples = dict(dev=None, train=1500)
    if args.rank:
        eval_samples.update(dict(dev_grouped=1500, train_grouped=1500))

    params = TrainParams(
        SerializableOptimizer("Adadelta", dict(learning_rate=1.0)),
        num_epochs=n_epochs, ema=0.999, max_checkpoints_to_keep=2,
        async_encoding=8, log_period=30, eval_period=1800, save_period=1800,
        eval_samples=eval_samples, best_weights=('dev', 'iterative-relevance/second/average_precision'),
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
    # os.environ["CUDA_VISIBLE_DEVICES"] = "5"
    main()
