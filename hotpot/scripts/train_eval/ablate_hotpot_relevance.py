import argparse

from datetime import datetime

from tensorflow.python import TruncatedNormal

from hotpot import model_dir, trainer
from hotpot.data_handling.dataset import ClusteredBatcher, multiple_contexts_len
from hotpot.data_handling.hotpot.hotpot_data import HotpotQuestions
from hotpot.data_handling.hotpot.hotpot_relevance_training_data import HotpotTextLengthPreprocessor, \
    HotpotQuestionFilter, HotpotBinaryRelevanceTrainingData
from hotpot.encoder import BinaryAnswerEncoder, QuestionsAndParagraphsEncoder
from hotpot.evaluator import LossEvaluator, BinaryClassificationEvaluator
from hotpot.models.multiple_context_models import ContextPairRelevanceModel, ContextsToQuestionModel, \
    MultiHopContextsToQuestionModel, MultiHopContextsOnlyModel
from hotpot.nn.attention import BiAttention, StaticAttentionSelf, AttentionWithPostMapper
from hotpot.nn.embedder import FixedWordEmbedder, CharWordEmbedder, LearnedCharEmbedder
from hotpot.nn.layers import MaxPool, Conv1d, SequenceMapperSeq, VariationalDropoutLayer, MergeTwoContextsConcatQuestion, MaxMerge, FullyConnected, ResidualLayer, ConcatWithProduct, WeightedMerge
from hotpot.nn.recurrent_layers import CudnnGru, CudnnGruEncoder
from hotpot.nn.relevance_prediction import BinaryFixedPredictor
from hotpot.nn.similarity_layers import TriLinear
from hotpot.trainer import TrainParams, SerializableOptimizer, resume_training_with


def get_model(rnn_dim):
    recurrent_layer = CudnnGru(rnn_dim, w_init=TruncatedNormal(stddev=0.05))
    answer_encoder = BinaryAnswerEncoder()

    return ContextPairRelevanceModel(
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
            # VariationalDropoutLayer(0.8),  # fixme probably doesn't belong here
        ),
        question_to_context_attention=None,
        context_to_context_attention=None,
        context_to_question_attention=None,
        sequence_encoder=MaxPool(map_layer=None, min_val=0, regular_reshape=True),
        merger=MergeTwoContextsConcatQuestion(),
        predictor=BinaryFixedPredictor()
    )


def get_contexts_to_question_model(rnn_dim, post_merge):
    recurrent_layer = CudnnGru(rnn_dim, w_init=TruncatedNormal(stddev=0.05))
    answer_encoder = BinaryAnswerEncoder()

    if post_merge == 'res_rnn_self_att':
        post_map_layer = SequenceMapperSeq(FullyConnected(rnn_dim * 2, activation="relu"),
                                           ResidualLayer(SequenceMapperSeq(
                                               VariationalDropoutLayer(0.8),
                                               recurrent_layer,
                                               VariationalDropoutLayer(0.8),
                                               StaticAttentionSelf(TriLinear(bias=True), ConcatWithProduct()),
                                               FullyConnected(rnn_dim * 2, activation="relu"),
                                           )))
    elif post_merge == 'res_rnn':
        post_map_layer = SequenceMapperSeq(FullyConnected(rnn_dim * 2, activation="relu"),
                                           ResidualLayer(SequenceMapperSeq(
                                               VariationalDropoutLayer(0.8),
                                               recurrent_layer,
                                               FullyConnected(rnn_dim * 2, activation="relu"),
                                           )))
    elif post_merge == 'res_self_att':
        post_map_layer = SequenceMapperSeq(FullyConnected(rnn_dim * 2, activation="relu"),
                                           ResidualLayer(SequenceMapperSeq(
                                               VariationalDropoutLayer(0.8),
                                               StaticAttentionSelf(TriLinear(bias=True), ConcatWithProduct()),
                                               FullyConnected(rnn_dim * 2, activation="relu"),
                                           )))
    else:
        raise NotImplementedError()

    return ContextsToQuestionModel(
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
        attention_merger=MaxMerge(pre_map_layer=None,
                                  post_map_layer=post_map_layer),
        context_to_question_attention=BiAttention(TriLinear(bias=True), True),
        sequence_encoder=MaxPool(map_layer=None, min_val=0, regular_reshape=True),
        predictor=BinaryFixedPredictor()
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


def get_multi_hop_model(rnn_dim, c2c: bool, q2c: bool, res_rnn: bool, res_self_att: bool, post_merge: bool,
                        encoder: str, merge_type: str, num_c2c_hops: int):
    recurrent_layer = CudnnGru(rnn_dim, w_init=TruncatedNormal(stddev=0.05))
    answer_encoder = BinaryAnswerEncoder()

    res_model = get_res_fc_seq_fc(model_rnn_dim=rnn_dim, rnn=res_rnn, self_att=res_self_att)
    context_to_context = \
        AttentionWithPostMapper(BiAttention(TriLinear(bias=True), True), post_mapper=res_model) if c2c else None
    question_to_context = \
        AttentionWithPostMapper(BiAttention(TriLinear(bias=True), True), post_mapper=res_model) if q2c else None

    if encoder == 'max':
        sequence_encoder = MaxPool(map_layer=None, min_val=0, regular_reshape=True)
    elif encoder == 'rnn':
        sequence_encoder = CudnnGruEncoder(rnn_dim, w_init=TruncatedNormal(stddev=0.05))
    else:
        raise NotImplementedError()

    if merge_type == 'max':
        attention_merger = MaxMerge(pre_map_layer=None, post_map_layer=(res_model if post_merge else None))
    else:
        attention_merger = WeightedMerge(pre_map_layer=None, post_map_layer=(res_model if post_merge else None),
                                         weight_type=merge_type)

    return MultiHopContextsToQuestionModel(
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
        context_to_context_attention=context_to_context,
        c2c_hops=num_c2c_hops,
        context_to_question_attention=BiAttention(TriLinear(bias=True), True),
        attention_merger=attention_merger,
        sequence_encoder=sequence_encoder,
        predictor=BinaryFixedPredictor()
    )


def get_context_only_model(rnn_dim, res_rnn: bool, res_self_att: bool, encoder: str, num_c2c_hops: int):
    recurrent_layer = CudnnGru(rnn_dim, w_init=TruncatedNormal(stddev=0.05))
    answer_encoder = BinaryAnswerEncoder()

    res_model = get_res_fc_seq_fc(model_rnn_dim=rnn_dim, rnn=res_rnn, self_att=res_self_att)
    context_to_context = \
        AttentionWithPostMapper(BiAttention(TriLinear(bias=True), True), post_mapper=res_model)

    if encoder == 'max':
        sequence_encoder = MaxPool(map_layer=None, min_val=0, regular_reshape=True)
    elif encoder == 'rnn':
        sequence_encoder = CudnnGruEncoder(rnn_dim, w_init=TruncatedNormal(stddev=0.05))
    else:
        raise NotImplementedError()
    return MultiHopContextsOnlyModel(
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
        context_to_context_attention=context_to_context,
        c2c_hops=num_c2c_hops,
        sequence_encoder=sequence_encoder,
        predictor=BinaryFixedPredictor()
    )


def main():
    parser = argparse.ArgumentParser(description='Train a model on the Hotpot pairwise relevance dataset')
    parser.add_argument("name", help="Where to store the model")
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

    # model = get_model(rnn_dim=150)
    # model = get_contexts_to_question_model(rnn_dim=150, post_merge='res_self_att')
    model = get_multi_hop_model(rnn_dim=150, c2c=True, q2c=False, res_rnn=True, res_self_att=False, post_merge=True,
                                encoder='max', merge_type='max', num_c2c_hops=1)
    # model = get_context_only_model(rnn_dim=150, res_rnn=True, res_self_att=False,
    #                                encoder='max', num_c2c_hops=1)

    corpus = HotpotQuestions()
    train_batcher = ClusteredBatcher(45, multiple_contexts_len, truncate_batches=True)
    dev_batcher = ClusteredBatcher(45, multiple_contexts_len, truncate_batches=True)
    data = HotpotBinaryRelevanceTrainingData(corpus=corpus, train_batcher=train_batcher, dev_batcher=dev_batcher,
                                             sample_filter=HotpotQuestionFilter(2), preprocessor=HotpotTextLengthPreprocessor(600),
                                             sample_train=None, sample_dev=None, sample_seed=18,
                                             add_gold_distractors=True)

    eval = [LossEvaluator(), BinaryClassificationEvaluator()]

    n_epochs = 80

    params = TrainParams(
        SerializableOptimizer("Adadelta", dict(learning_rate=1.0)),
        num_epochs=n_epochs, ema=0.999, max_checkpoints_to_keep=2,
        async_encoding=8, log_period=30, eval_period=1800, save_period=1800,
        eval_samples=dict(dev=None, train=3000), best_weights=('dev', 'binary-relevance/f1_score'),
        monitor_gradients=True
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
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    main()
