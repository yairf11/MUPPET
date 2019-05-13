import argparse
import itertools
from time import time
from typing import Dict, List
from itertools import accumulate

import numpy as np
import tensorflow as tf
from os.path import join
from collections import Counter

from sklearn.preprocessing import normalize
from tqdm import tqdm

from hotpot import config
from hotpot.input_optimizing.intermediate_trainer import GraphModifier, InjectVarParams, get_dot_products, \
    init_graph_get_samples

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def get_confidences(graph: GraphModifier, batch: List):
    graph.set_input(batch)
    _, softmax = graph.get_original_loss(return_softmax=True)
    return softmax[:, 1]


def get_confidences_dict(graph: GraphModifier, qid2sample: Dict):
    _, softmaxes = graph.get_original_loss_for_batch(sum([x for x in qid2sample.values()], []), return_softmax=True)
    softmaxes = softmaxes[:, 1]
    lens = [len(x) for x in qid2sample.values()]
    acc_lens = [0] + list(accumulate(lens))
    return {qid: softmaxes[acc_lens[idx]:acc_lens[idx] + len(qid2sample[qid])]
            for idx, qid in enumerate(list(qid2sample.keys()))}


def get_optimized_dots(graph: GraphModifier, qid2sample: Dict, original_tensor_name, opt_steps=10):
    batch = [x[0] for x in qid2sample.values()]
    graph.set_input(batch)
    graph.build_modified_model([InjectVarParams(original_tensor_name, 'new_in', True, None)],
                               tf.train.MomentumOptimizer(learning_rate=0.1, momentum=0.9, use_nesterov=True))
    graph.optimize(opt_steps)
    print(graph.get_loss())
    return {qid: get_dot_products(graph, batch=qid2sample[qid], original_tensor_name=original_tensor_name,
                                  specific_index=idx)
            for idx, qid in enumerate(qid2sample.keys())}


def get_optimized_dots_fast(graph: GraphModifier, qid2sample: Dict, original_tensor_name, opt_steps=10,
                            keys_in_batch=20, print_loss=False, normalize_originals=False):
    batch = [x[0] for x in qid2sample.values()]
    graph.set_input(batch)
    graph.build_modified_model([InjectVarParams(original_tensor_name, 'new_in', True, None)],
                               tf.train.AdamOptimizer(learning_rate=0.001))#, momentum=0.9, use_nesterov=True))
    graph.optimize(opt_steps)
    if print_loss:
        print(graph.get_loss())
    keys = list(qid2sample.keys())
    optimized_rep = np.expand_dims(graph.get_optimized_intermediate(original_tensor_name, use_new_names=False), -1)
    res = {}
    for i in range(0, len(batch), keys_in_batch):
        lens = [len(qid2sample[x]) for x in keys[i:i + keys_in_batch]]
        batch_samples = sum([qid2sample[x] for x in keys[i:i + keys_in_batch]], [])
        original_rep = graph.get_original_intermediate_for_batch(original_tensor_name,
                                                                 batch_samples)
        if normalize_originals:
            original_rep = normalize(original_rep, norm='l2', axis=1)
        if lens.count(lens[0]) == len(lens):
            num_keys = len(keys[i:i + keys_in_batch])
            num_vecs = int(len(batch_samples) / num_keys)
            original_rep = original_rep.reshape((num_keys, num_vecs, -1))
            dots = np.matmul(original_rep, optimized_rep[i:i + keys_in_batch]).squeeze(axis=-1)
            res.update({qid: dots[idx] for idx, qid in enumerate(keys[i:i + keys_in_batch])})
        else:
            acc_lens = [0] + list(accumulate(lens))
            res.update({qid: original_rep[acc_lens[idx]:acc_lens[idx + 1]].dot(optimized_rep[i + idx]).squeeze(axis=-1)
                        for idx, qid in enumerate(keys[i:i + keys_in_batch])})
    return res


def correlation_test():
    parser = argparse.ArgumentParser(description='Evaluate the intermediate input dot-product on SQuAD')
    parser.add_argument('model', help='model directory to evaluate')
    parser.add_argument('tensor', help="The name of the tensor to optimize")
    parser.add_argument('--sample', type=int, default=None)
    parser.add_argument('--opt-steps', type=int, default=100)
    parser.add_argument('--norm', action='store_true')
    args = parser.parse_args()

    tensor_name = args.tensor  # 'seq_enc/encode_context:0'
    model_dir = args.model

    print("Building Graph...")
    graph, samples = init_graph_get_samples('squad_full', model_dir)

    print("Creating sample dict...")
    grouped_samples = {key: sorted(list(group), key=lambda x: x.label, reverse=True) for key, group in
                       itertools.groupby(sorted(samples, key=lambda x: x.question_id),
                                         key=lambda x: x.question_id)}

    all_ids = sorted(list(grouped_samples.keys()), key=lambda x: len(grouped_samples[x]))
    if args.sample is not None:
        all_ids = all_ids[:args.sample]
    print("Calculating softmax confidences...")
    grouped_confidences = {}
    for i in tqdm(range(0, len(all_ids), 20)):
        grouped_confidences.update(get_confidences_dict(graph, {key: grouped_samples[key]
                                                                for key in all_ids[i:i + 20]}))

    precision_at_1 = Counter([np.argmax(grouped_confidences[key]) for key in all_ids])[0] / len(all_ids)
    print(f"P@1: {precision_at_1}")
    mean_rank = np.mean([list((-grouped_confidences[key]).argsort()).index(0) + 1 for key in all_ids])
    print(f"Mean Rank: {mean_rank}")

    # grouped_samples_by_size = {key: list(group) for key, group in
    #                            itertools.groupby(all_ids, key=lambda x: len(grouped_samples[x]))}
    # chunked_samples_by_size =

    grouped_dots = {}
    print("Optimizing and calculating dot products...")
    for i in tqdm(range(0, len(all_ids), 200)):
        grouped_dots.update(get_optimized_dots_fast(graph, {key: grouped_samples[key] for key in all_ids[i:i + 200]},
                                                    original_tensor_name=tensor_name, opt_steps=args.opt_steps,
                                                    keys_in_batch=20, print_loss=(i == 0),
                                                    normalize_originals=args.norm))

    precision_at_1_dots = Counter([np.argmax(grouped_dots[key]) for key in all_ids])[0] / len(all_ids)
    print(f"P@1 for dots: {precision_at_1_dots}")
    mean_rank_dots = np.mean([list((-grouped_dots[key]).argsort()).index(0) + 1 for key in all_ids])
    print(f"Mean Rank for dots: {mean_rank_dots}")

    print("Aggregating results...")
    corr = []
    for some_key in all_ids:
        conf = grouped_confidences[some_key]
        dots = grouped_dots[some_key]
        conf_diff = conf[:, None] >= conf
        dots_diff = dots[:, None] >= dots
        indices = np.tril_indices(len(conf), -1)
        corr.extend((~np.logical_xor(conf_diff[indices], dots_diff[indices])).tolist())
    correlation_acc = np.mean(corr)

    print(f"In {correlation_acc*100} of the cases, the dot-product had the same order as the softmax probabilities")


if __name__ == '__main__':
    correlation_test()
