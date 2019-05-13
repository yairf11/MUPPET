import argparse
import itertools
from time import time
import numpy as np
import tensorflow as tf
from os.path import join
from copy import deepcopy
import numpy as np
from collections import Counter

from sklearn.preprocessing import normalize
from tqdm import tqdm

from hotpot import trainer
from hotpot.data_handling.dataset import ClusteredBatcher, multiple_contexts_len
from hotpot.data_handling.hotpot.hotpot_data import HotpotQuestions
from hotpot.evaluator import BinaryClassificationEvaluator, Evaluator, Evaluation
from hotpot.model_dir import ModelDir
from hotpot.nn.relevance_prediction import BinaryPrediction
from hotpot.scripts.train_eval.relevance_eval import RecordFineGrainedBinaryPrediction
from hotpot.utils import transpose_lists, print_table
from hotpot import configurable
from hotpot.configurable import Configurable
from hotpot.data_handling.dataset import TrainingData, Dataset
from hotpot.evaluator import Evaluator, Evaluation, AysncEvaluatorRunner, EvaluatorRunner
from hotpot.model import Model
from hotpot.model_dir import ModelDir
from hotpot.input_optimizing import intermediate_trainer
from hotpot.input_optimizing import squad_basic_correlation
from importlib import reload


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate the intermediate input dot-product on SQuAD')
    parser.add_argument('model', help='model directory to evaluate')
    # parser.add_argument('tensor', help="The name of the tensor to optimize")
    parser.add_argument('--sample', type=int, default=None)
    parser.add_argument('--opt-steps', type=int, default=100)
    parser.add_argument('--seq-len', type=int, default=300)
    parser.add_argument('--norm', action='store_true')
    args = parser.parse_args()
    model_dir = args.model
    graph, samples = intermediate_trainer.init_graph_get_samples('squad_full', model_dir, sequence=True)

    tensor_name = 'context_encode/fixed:0'
    seq_tensor_name = 'map_embed/sequence:0'

    grouped_samples = {key: sorted(list(group), key=lambda x: x.label, reverse=True) for key, group in
                       itertools.groupby(sorted(samples, key=lambda x: x.question_id),
                                         key=lambda x: x.question_id)}

    all_ids = sorted(list(grouped_samples.keys()), key=lambda x: len(grouped_samples[x]))
    if args.sample is not None:
        all_ids = all_ids[:args.sample]
    print("Calculating softmax confidences...")
    grouped_confidences = {}
    for i in tqdm(range(0, len(all_ids), 20)):
        grouped_confidences.update(squad_basic_correlation.get_confidences_dict(graph, {key: grouped_samples[key]
                                                                                        for key in all_ids[i:i + 20]}))

    precision_at_1 = Counter([np.argmax(grouped_confidences[key]) for key in all_ids])[0] / len(all_ids)
    print(f"P@1: {precision_at_1}")
    mean_rank = np.mean([list((-grouped_confidences[key]).argsort()).index(0) + 1 for key in all_ids])
    print(f"Mean Rank: {mean_rank}")


    def get_optimized_dots_fast(graph, qid2sample, seq_tensor_name, fix_tensor_name, opt_steps=10,
                                keys_in_batch=20, print_loss=False, seq_len=300, normalize_originals=False):
        batch = [x[0] for x in qid2sample.values()]
        graph.set_input(batch)
        graph.build_modified_model(
            [intermediate_trainer.InjectVarParams(seq_tensor_name, 'new_in', True, (None, seq_len, None)),
             intermediate_trainer.InjectVarParams(fix_tensor_name, fix_tensor_name, False, None)],
            tf.train.MomentumOptimizer(learning_rate=0.1, momentum=0.9, use_nesterov=True))
        graph.optimize(opt_steps)
        if print_loss:
            print(graph.get_loss())
        keys = list(qid2sample.keys())
        optimized_rep = np.expand_dims(graph.get_optimized_intermediate(fix_tensor_name, use_new_names=False), -1)
        res = {}
        for i in range(0, len(batch), keys_in_batch):
            lens = [len(qid2sample[x]) for x in keys[i:i + keys_in_batch]]
            batch_samples = sum([qid2sample[x] for x in keys[i:i + keys_in_batch]], [])
            original_rep = graph.get_original_intermediate_for_batch(fix_tensor_name,
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
                acc_lens = [0] + list(itertools.accumulate(lens))
                res.update(
                    {qid: original_rep[acc_lens[idx]:acc_lens[idx + 1]].dot(optimized_rep[i + idx]).squeeze(axis=-1)
                     for idx, qid in enumerate(keys[i:i + keys_in_batch])})
        return res


    grouped_dots = {}
    for i in tqdm(range(0, len(all_ids), 200)):
        grouped_dots.update(get_optimized_dots_fast(graph, {key: grouped_samples[key] for key in all_ids[i:i + 200]},
                                                    seq_tensor_name=seq_tensor_name, fix_tensor_name=tensor_name,
                                                    opt_steps=args.opt_steps,
                                                    keys_in_batch=20, print_loss=(i == 0),
                                                    seq_len=args.seq_len,
                                                    normalize_originals=args.norm))

    precision_at_1_dots = Counter([np.argmax(grouped_dots[key]) for key in all_ids])[0] / len(all_ids)
    print(f"P@1 for dots: {precision_at_1_dots}")
    mean_rank_dots = np.mean([list((-grouped_dots[key]).argsort()).index(0) + 1 for key in all_ids])
    print(f"Mean Rank for dots: {mean_rank_dots}")

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