from typing import List, Dict

import tensorflow as tf
import itertools
from time import time
import numpy as np
from os.path import join
from collections import namedtuple

from sklearn.preprocessing import normalize
from tqdm import tqdm

from hotpot.data_handling.squad.squad_data import SquadRelevanceCorpus
from hotpot.data_handling.squad.squad_relevance_training_data import SquadTextLengthPreprocessor, \
    SquadFullQuestionParagraphPairsDataset
from hotpot.data_handling.dataset import ClusteredBatcher, multiple_contexts_len
from hotpot.data_handling.hotpot.hotpot_data import HotpotQuestions
from hotpot.data_handling.hotpot.hotpot_relevance_training_data import HotpotTextLengthPreprocessor, \
    HotpotFullQuestionParagraphPairsDataset
from hotpot.models.multiple_context_models import INTERMEDIATE_LAYER_COLLECTION
from hotpot.data_handling.dataset import TrainingData, Dataset
from hotpot.model_dir import ModelDir

InjectVarParams = namedtuple('InjectVarParams', ['original_name', 'new_name', 'optimize', 'shape'])


class GraphModifier(object):
    """
    A class for loading and modifying trained models.
    """

    def __init__(self, model_dir_path: str):
        self.model_dir = ModelDir(model_dir_path)
        self.checkpoint = self.model_dir.get_best_weights()
        self.model = self.model_dir.get_model()

        self.original_graph_session = None
        self.meta_graph_def = None
        self.modified_graph_session = None
        self.input_map_original_names = None
        self.input_map_new_names = None
        self.opt_vars = None
        self.original_feed_dict = None
        self.modified_feed_dict = None
        self.loss = None
        self.softmax = None
        self.minimize_op = None
        self.original_softmax = None
        self.cur_batch = None

    def init_original_model(self, data: Dataset, loader):
        self.original_graph_session = tf.Session(config=tf.ConfigProto(allow_soft_placement=True), graph=tf.Graph())
        self.meta_graph_def = self._build_restore_export(data, loader)
        self.original_softmax = tf.nn.softmax(  # hard coded tensor name because they all have the same predictor
            self.original_graph_session.graph.get_tensor_by_name('predictor/compute_logits/fully_connected/BiasAdd:0'),
            name='softmax')

    def _build_restore_export(self, data: Dataset, loader):
        with self.original_graph_session.graph.as_default():
            self.model.set_inputs([data], loader)
            inputs = self.model.get_placeholders()
            input_dict = {p: x for p, x in zip(self.model.get_placeholders(), inputs)}
            with self.original_graph_session.as_default():
                with self.original_graph_session.graph.as_default():  # Not sure if this is needed
                    _ = self.model.get_predictions_for(input_dict)  # for building the model
            saver = tf.train.Saver()
            saver.restore(self.original_graph_session, self.checkpoint)
            meta_def = saver.export_meta_graph()
        return meta_def

    def build_modified_model(self, inject_vars: List[InjectVarParams], optimizer: tf.train.Optimizer,
                             collection_key=INTERMEDIATE_LAYER_COLLECTION):
        intermediate_names = [x.name for x in self.original_graph_session.graph.get_collection(collection_key)]
        for x in inject_vars:
            if x.original_name not in intermediate_names:
                raise ValueError(f"{x.original_name} not in {collection_key} collection")
        if self.modified_graph_session is not None:
            self.modified_graph_session.close()
        self.modified_graph_session = tf.Session(config=tf.ConfigProto(allow_soft_placement=True), graph=tf.Graph())
        with self.modified_graph_session.graph.as_default():
            with self.modified_graph_session.as_default():
                input_map = {x.original_name: tf.get_variable(
                    initializer=tf.zeros(self.get_original_intermediate(x.original_name).shape),
                    dtype=tf.float32, validate_shape=True, name=x.new_name)
                    for x in inject_vars}
                saver = tf.train.import_meta_graph(self.meta_graph_def,
                                                   input_map={name: tf.convert_to_tensor(var)
                                                              for name, var in input_map.items()})
                saver.restore(self.modified_graph_session, self.checkpoint)
        self.input_map_original_names = input_map
        self.input_map_new_names = {x.new_name: self.input_map_original_names[x.original_name] for x in inject_vars}
        self.opt_vars = [self.input_map_new_names[x.new_name] for x in inject_vars if x.optimize]

        with self.modified_graph_session.graph.as_default():
            self.softmax = tf.nn.softmax(  # hard coded tensor name because they all have the same predictor
                self.modified_graph_session.graph.get_tensor_by_name(
                    'predictor/compute_logits/fully_connected/BiasAdd:0'),
                name='softmax')
            loss = tf.get_collection(tf.GraphKeys.LOSSES)
            self.loss = tf.add_n(loss, name="loss")
            self.minimize_op = optimizer.minimize(self.loss, var_list=self.opt_vars, name='optimize_inputs')

        initialize_uninitialized(self.modified_graph_session)
        self.modified_feed_dict = switch_dict_keys(self.original_feed_dict, self.modified_graph_session.graph)

    def assign_intermediate_vals(self, val_map: Dict, use_new_names=True):
        """
        Assign values to the new input layers.
        :param val_map: tensor_name (str) -> new value (np array etc.)
        :param use_new_names: Whether the mapping is from the new tensor names or from the original ones
        :return: None
        """
        input_map = self.input_map_new_names if use_new_names else self.input_map_original_names
        with self.modified_graph_session.graph.as_default():
            assigns = [tf.assign(input_map[k], v, validate_shape=True) for k, v in val_map.items()]
            self.modified_graph_session.run(assigns)

    def set_input(self, batch: List):
        self.cur_batch = batch
        self.original_feed_dict = self.model.encode(batch, False)
        # self.modified_feed_dict = switch_dict_keys(self.original_feed_dict, self.modified_graph_session.graph)

    def get_original_intermediate(self, tensor_name):
        """ The tensor name should be in the original graph """
        return self.original_graph_session.run(tensor_name, feed_dict=self.original_feed_dict)

    def get_original_intermediate_for_batch(self, tensor_name, batch, restore_feed_dict=False):
        res = self.original_graph_session.run(tensor_name, feed_dict=self.model.encode(batch, False))
        if restore_feed_dict:
            self.original_feed_dict = self.model.encode(self.cur_batch, False)  # because the dict is changing
        return res

    def get_optimized_intermediate(self, tensor_name, use_new_names=True):
        input_map = self.input_map_new_names if use_new_names else self.input_map_original_names
        return self.modified_graph_session.run(input_map[tensor_name], feed_dict=self.modified_feed_dict)

    def get_loss(self, return_softmax=False):
        if not return_softmax:
            return self.modified_graph_session.run(self.loss, feed_dict=self.modified_feed_dict)
        return self.modified_graph_session.run([self.loss, self.softmax], feed_dict=self.modified_feed_dict)

    def optimize(self, num_steps=1):
        for _ in range(num_steps):
            self.modified_graph_session.run(self.minimize_op, feed_dict=self.modified_feed_dict)

    def get_original_loss(self, return_softmax=False):
        if not return_softmax:
            return self.original_graph_session.run('predictor/Mean:0', feed_dict=self.original_feed_dict)
        return self.original_graph_session.run(['predictor/Mean:0', self.original_softmax],
                                               feed_dict=self.original_feed_dict)

    def get_original_loss_for_batch(self, batch, return_softmax=False, restore_feed_dict=False):
        if not return_softmax:
            res = self.original_graph_session.run('predictor/Mean:0', feed_dict=self.model.encode(batch, False))
        else:
            res = self.original_graph_session.run(['predictor/Mean:0', self.original_softmax],
                                                  feed_dict=self.model.encode(batch, False))
        if restore_feed_dict:
            self.original_feed_dict = self.model.encode(self.cur_batch, False)  # because the dict is changing
        return res


def merge_if_none(list1, list2):
    return list2 if list1 is None else [x2 if (x1 is None) else x1 for x1, x2 in zip(list1, list2)]


class GraphModifierWithSequence(GraphModifier):
    """
    A class for models that have a bottleneck layer but also use the full context for predicting.
    This means we need to optimize a sequence, and use the fixed representation only for dot-products.
    """
    def __init__(self, model_dir_path: str):
        super().__init__(model_dir_path=model_dir_path)
        self.no_opt_vars = None

    def build_modified_model(self, inject_vars: List[InjectVarParams], optimizer: tf.train.Optimizer,
                             collection_key=INTERMEDIATE_LAYER_COLLECTION):
        intermediate_names = [x.name for x in self.original_graph_session.graph.get_collection(collection_key)]
        for x in inject_vars:
            if x.original_name not in intermediate_names:
                raise ValueError(f"{x.original_name} not in {collection_key} collection")
        if self.modified_graph_session is not None:
            self.modified_graph_session.close()
        self.modified_graph_session = tf.Session(config=tf.ConfigProto(allow_soft_placement=True), graph=tf.Graph())
        with self.modified_graph_session.graph.as_default():
            with self.modified_graph_session.as_default():
                input_map = {x.original_name: tf.get_variable(
                    initializer=tf.zeros(merge_if_none(x.shape, self.get_original_intermediate(x.original_name).shape)),
                    dtype=tf.float32, validate_shape=True, name=x.new_name)
                    for x in inject_vars if x.optimize}
                saver = tf.train.import_meta_graph(self.meta_graph_def,
                                                   input_map={name: tf.convert_to_tensor(var)
                                                              for name, var in input_map.items()})
                saver.restore(self.modified_graph_session, self.checkpoint)
        self.input_map_original_names = input_map
        self.input_map_original_names.update(
            {x.original_name: self.modified_graph_session.graph.get_tensor_by_name(x.original_name)
             for x in inject_vars if not x.optimize}
        )
        self.input_map_new_names = {x.new_name: self.input_map_original_names[x.original_name] for x in inject_vars}
        self.opt_vars = [self.input_map_new_names[x.new_name] for x in inject_vars if x.optimize]
        if len(self.opt_vars) > 1:
            raise ValueError("Can't handle more than 1 sequence to optimize yet")

        with self.modified_graph_session.graph.as_default():
            self.softmax = tf.nn.softmax(  # hard coded tensor name because they all have the same predictor
                self.modified_graph_session.graph.get_tensor_by_name(
                    'predictor/compute_logits/fully_connected/BiasAdd:0'),
                name='softmax')
            loss = tf.get_collection(tf.GraphKeys.LOSSES)
            self.loss = tf.add_n(loss, name="loss")
            self.minimize_op = optimizer.minimize(self.loss, var_list=self.opt_vars, name='optimize_inputs')

        initialize_uninitialized(self.modified_graph_session)
        self.modified_feed_dict = switch_dict_keys(self.original_feed_dict, self.modified_graph_session.graph)
        context_len_key = [ph for ph in self.modified_feed_dict if 'context_lens' in ph.name]
        if len(context_len_key) > 1:
            raise ValueError("Too many context_len placeholders")
        self.modified_feed_dict[context_len_key[0]][:] = self.opt_vars[0].shape.as_list()[1]  # todo: make this prettier


def initialize_uninitialized(sess):
    with sess.graph.as_default():
        global_vars = tf.global_variables()
        is_not_initialized = sess.run([tf.is_variable_initialized(var) for var in global_vars])
        not_initialized_vars = [v for (v, f) in zip(global_vars, is_not_initialized) if not f]

        # print(f"Initializing: {[str(i.name) for i in not_initialized_vars]}")  # only for testing
        if len(not_initialized_vars):
            sess.run(tf.variables_initializer(not_initialized_vars))


def switch_dict_keys(orig_dict, new_graph):
    new_dict = {}
    for k, v in orig_dict.items():
        new_dict[new_graph.get_tensor_by_name(k.name)] = v
    return new_dict


def get_full_hotpot_samples(dev=True):
    corpus = HotpotQuestions()
    questions = corpus.get_dev() if dev else corpus.get_train()

    question_preprocessor = HotpotTextLengthPreprocessor(600)
    questions = [question_preprocessor.preprocess(x) for x in questions
                 if (question_preprocessor.preprocess(x) is not None)]

    batcher = ClusteredBatcher(1, multiple_contexts_len, truncate_batches=True)
    data = HotpotFullQuestionParagraphPairsDataset(questions, batcher)
    loader = corpus.get_resource_loader()

    return data, loader


def get_full_squad_samples(dev=True):
    corpus = SquadRelevanceCorpus()
    questions = corpus.get_dev() if dev else corpus.get_train()

    question_preprocessor = SquadTextLengthPreprocessor(600)
    questions = [question_preprocessor.preprocess(x) for x in questions
                 if (question_preprocessor.preprocess(x) is not None)]

    batcher = ClusteredBatcher(1, multiple_contexts_len, truncate_batches=True)
    data = SquadFullQuestionParagraphPairsDataset(questions, batcher)
    loader = corpus.get_resource_loader()

    return data, loader


def init_graph_get_samples(dataset: str, model_dir_path: str, get_dev=True, sequence=False):
    if dataset == 'squad_full':
        data, loader = get_full_squad_samples(get_dev)
    elif dataset == 'hotpot_full':
        data, loader = get_full_hotpot_samples(get_dev)
    else:
        raise ValueError(f"No such dataset: {dataset}")
    graph = GraphModifier(model_dir_path) if not sequence else GraphModifierWithSequence(model_dir_path)
    graph.init_original_model(data, loader)

    return graph, data.samples


def get_dot_products(graph: GraphModifier, original_tensor_name, batch=None, specific_index=None,
                     normalize_originals=False):
    """
    Given a built graph, calculate the dot-product between original and optimized representations
    :param normalize_originals: Whether to normalize the original representations or not
    :param specific_index: If we want a single optimized representation against all the given batch
    :param batch: Evaluate original representations on the given batch. if None, uses the existing batch.
    :param graph: A built graph, already seeded
    :param original_tensor_name: The name of the tensor of the original representation.
    This function performs dot-product elementwise. assumes only 2D tensors.
    :return: an np array with the dot product results
    """
    if specific_index is not None and not batch:
        raise ValueError("No sense in asking for specific index if not changing the batch.")

    if batch is None:
        original_rep = graph.get_original_intermediate(original_tensor_name)
    else:
        original_rep = graph.get_original_intermediate_for_batch(original_tensor_name, batch)
    if normalize_originals:
        original_rep = normalize(original_rep, norm='l2', axis=1)
    optimized_rep = graph.get_optimized_intermediate(original_tensor_name, use_new_names=False)
    res = np.empty(original_rep.shape[0])
    if specific_index is None:
        for i in range(original_rep.shape[0]):
            res[i] = original_rep[i].dot(optimized_rep[i])
    else:
        res = original_rep.dot(optimized_rep[specific_index])
    return res


def graph_modifier_test():
    corpus = HotpotQuestions()
    questions = corpus.get_dev()

    question_preprocessor = HotpotTextLengthPreprocessor(600)
    questions = [question_preprocessor.preprocess(x) for x in questions
                 if (question_preprocessor.preprocess(x) is not None)]

    batcher = ClusteredBatcher(1, multiple_contexts_len, truncate_batches=True)
    data = HotpotFullQuestionParagraphPairsDataset(questions, batcher)

    loader = corpus.get_resource_loader()

    pos = [sample for sample in data.samples if sample.label == 1]
    batch = pos[:1]

    dir_path = './logs/hotpot/c2c_c2q/c2c_new_bidirect_att_c2q_res_rnn_merge=max_encoder=max_batch=45_lr=token-th=600-1203-182149'
    mod_graph = GraphModifier(dir_path)

    mod_graph.init_original_model(data, loader)

    mod_graph.set_input(batch)

    params = [InjectVarParams('map_embed/map_embed_context1:0', 'new_in', True)]

    mod_graph.build_modified_model(params, tf.train.MomentumOptimizer(learning_rate=0.1, momentum=0.9))

    # mod_graph.set_input(batch)

    # new_in_shape = mod_graph.get_original_intermediate('map_embed/map_embed_context1:0').shape
    #
    # mod_graph.assign_intermediate_vals({'new_in': np.zeros(new_in_shape)})

    print(f"Original loss: {mod_graph.get_original_loss()}")
    print(f"Initial loss: {mod_graph.get_loss()}")
    for i in range(10):
        mod_graph.optimize(2)
        print(f"Loss after {(i+1)*2} updates: {mod_graph.get_loss()}")


if __name__ == '__main__':
    graph_modifier_test()
