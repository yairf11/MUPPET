"""Script for creating the vocabulary file and the corresponding embedding file for a pretrained ELMO model
Code adapted from https://github.com/allenai/bilm-tf/blob/master/usage_token.py"""
import argparse
import json
import os

import tensorflow as tf
from hotpot.data_handling.dataset import TrainingData, ClusteredBatcher, multiple_contexts_len
from hotpot.config import LM_DIR
from hotpot.data_handling.hotpot.hotpot_data import HotpotQuestions
from hotpot.data_handling.hotpot.hotpot_relevance_training_data import HotpotBinaryRelevanceTrainingData
from hotpot.data_handling.squad.squad_data import SquadRelevanceCorpus
from hotpot.data_handling.squad.squad_relevance_training_data import SquadBinaryRelevanceTrainingData
from hotpot.elmo.lm_model import dump_token_embeddings


def build_vocab_from_preprocessed(data: TrainingData, vocab_file, embd_file):
    train = data.get_train()
    eval_datasets = data.get_eval()

    datasets = [train] + list(eval_datasets.values())

    voc = {'<S>', '</S>'}
    for dataset in datasets:
        voc.update(dataset.get_vocab())

    with open(vocab_file, 'w') as f:
        f.write('\n'.join(voc))

    if embd_file is None:
        return

    datadir = os.path.join(LM_DIR, '5.5B')
    options_file = os.path.join(datadir, 'elmo_2x4096_512_2048cnn_2xhighway_5.5B_options.json')
    weight_file = os.path.join(datadir, 'elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5')

    # Dump the token embeddings to a file. Run this once for your dataset.
    dump_token_embeddings(
        vocab_file, options_file, weight_file, embd_file
    )


def create_original_embeddings_from_vocab(vocab_file, embed_file, new_options_file=None):
    datadir = os.path.join(LM_DIR, '5.5B')
    options_file = os.path.join(datadir, 'elmo_2x4096_512_2048cnn_2xhighway_5.5B_options.json')
    weight_file = os.path.join(datadir, 'elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5')

    if new_options_file is not None:
        with open(options_file, 'r') as f:
            options = json.load(f)
        with open(vocab_file, 'r') as f:
            vocab_size = len(f.readlines())
        options['n_tokens_vocab'] = vocab_size + 1
        with open(new_options_file, 'w') as f:
            json.dump(options, f)
        options_file = new_options_file

    # Dump the token embeddings to a file. Run this once for your dataset.
    dump_token_embeddings(
        vocab_file, options_file, weight_file, embed_file
    )
    tf.reset_default_graph()


def build_hotpot_elmo(vocab_file, embd_file):
    corpus = HotpotQuestions()
    some_batcher = ClusteredBatcher(64, multiple_contexts_len, truncate_batches=True)
    data = HotpotBinaryRelevanceTrainingData(corpus=corpus, train_batcher=some_batcher, dev_batcher=some_batcher,
                                             sample_filter=None, preprocessor=None,
                                             sample_train=None, sample_dev=None, sample_seed=18)
    build_vocab_from_preprocessed(data, vocab_file, embd_file)


def build_squad_elmo(vocab_file, embd_file):
    corpus = SquadRelevanceCorpus()
    some_batcher = ClusteredBatcher(64, multiple_contexts_len, truncate_batches=True)
    data = SquadBinaryRelevanceTrainingData(corpus=corpus, train_batcher=some_batcher, dev_batcher=some_batcher,
                                            sample_filter=None, preprocessor=None,
                                            sample_train=None, sample_dev=None, sample_seed=18)
    build_vocab_from_preprocessed(data, vocab_file, embd_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create and dump pretrained elmo token embeddings')
    parser.add_argument("dataset", help="Which dataset to use", choices=['hotpot', 'squad'])
    parser.add_argument("vocab_file", help="Where to store the vocabulary file")
    parser.add_argument("embd_file", help="Where to dump the token embeddings")
    parser.add_argument("--new_options_file", type=str, default=None)
    args = parser.parse_args()

    if os.path.isfile(args.vocab_file):
        print("vocab exists, creating embeddings...")
        create_original_embeddings_from_vocab(args.vocab_file, args.embd_file, args.new_options_file)
    else:
        if args.dataset == 'squad':
            build_squad_elmo(args.vocab_file, args.embd_file)
        elif args.dataset == 'hotpot':
            build_hotpot_elmo(args.vocab_file, args.embd_file)
        else:
            raise ValueError("no such dataset")
