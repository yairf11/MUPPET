import argparse
from typing import List
import itertools
import numpy as np

from hotpot import trainer
from hotpot.data_handling.dataset import ClusteredBatcher, multiple_contexts_len
from hotpot.data_handling.hotpot.hotpot_data import HotpotQuestions
from hotpot.data_handling.relevance_training_data import BinaryQuestionAndParagraphs
from hotpot.data_handling.hotpot.hotpot_relevance_training_data import HotpotFullQuestionParagraphPairsDataset, \
    HotpotTextLengthPreprocessor
from hotpot.evaluator import BinaryClassificationEvaluator, Evaluator, Evaluation
from hotpot.model_dir import ModelDir
from hotpot.nn.relevance_prediction import BinaryPrediction
from hotpot.scripts.train_eval.relevance_eval import RecordFineGrainedBinaryPrediction
from hotpot.utils import transpose_lists, print_table


class RecordFullRankings(Evaluator):
    def tensors_needed(self, prediction: BinaryPrediction):
        probs = prediction.get_probs()
        return dict(probs=probs)

    def evaluate(self, data: List[BinaryQuestionAndParagraphs], true_len, **kwargs):
        probs = kwargs['probs']
        data_with_pos_probs = list(zip(data, probs[:, 1]))

        results = {}

        results['question_groups'] = {key: sorted(list(group), key=lambda x: x[1], reverse=True) for key, group in
                                      itertools.groupby(sorted(data_with_pos_probs, key=lambda x: x[0].question_id),
                                                        key=lambda x: x[0].question_id)}
        results['per_question_rank'] = {key: rank + 1 for key, group in results['question_groups'].items()
                                        for rank, sample in enumerate(group) if sample[0].label == 1}
        results['bridge_per_question_rank'] = {key: rank for key, rank in results['per_question_rank'].items()
                                               if results['question_groups'][key][0][0].q_type == 'bridge'}
        results['comparison_per_question_rank'] = {key: rank for key, rank in results['per_question_rank'].items()
                                                   if
                                                   results['question_groups'][key][0][0].q_type == 'comparison'}

        scalars = {}

        def add_metrics(metrics_dict, prefix, per_question_ranks):
            metrics_dict[f'{prefix}/P@1'] = np.mean([rank == 1 for rank in per_question_ranks.values()])
            metrics_dict[f'{prefix}/mean_rank'] = np.mean(list(per_question_ranks.values()))
            metrics_dict[f'{prefix}/MRR'] = np.mean([1.0 / x for x in per_question_ranks.values()])
            return metrics_dict

        scalars = add_metrics(scalars, 'global', results['per_question_rank'])
        scalars = add_metrics(scalars, 'bridge', results['bridge_per_question_rank'])
        scalars = add_metrics(scalars, 'comparison', results['comparison_per_question_rank'])

        return Evaluation(scalars, results)


def main():
    parser = argparse.ArgumentParser(description='Full ranking evaluation on Hotpot')
    parser.add_argument('model', help='model directory to evaluate')
    parser.add_argument('-n', '--sample_questions', type=int, default=None,
                        help="(for testing) run on a subset of questions")
    parser.add_argument('-b', '--batch_size', type=int, default=64,
                        help="Batch size, larger sizes can be faster but uses more memory")
    parser.add_argument('-s', '--step', default=None,
                        help="Weights to load, can be a checkpoint step or 'latest'")
    # parser.add_argument('-c', '--corpus', choices=["dev", "train"], default="dev")
    parser.add_argument('--no_ema', action="store_true", help="Don't use EMA weights even if they exist")
    args = parser.parse_args()

    model_dir = ModelDir(args.model)

    corpus = HotpotQuestions()
    # if args.corpus == "dev":
    #     questions = corpus.get_dev()
    # else:
    #     questions = corpus.get_train()
    questions = corpus.get_dev()

    question_preprocessor = HotpotTextLengthPreprocessor(600)
    questions = [question_preprocessor.preprocess(x) for x in questions
                 if (question_preprocessor.preprocess(x) is not None)]

    if args.sample_questions:
        np.random.RandomState(0).shuffle(sorted(questions, key=lambda x: x.question_id))
        questions = questions[:args.sample_questions]

    batcher = ClusteredBatcher(args.batch_size, multiple_contexts_len, truncate_batches=True)
    data = HotpotFullQuestionParagraphPairsDataset(questions, batcher)

    evaluators = [BinaryClassificationEvaluator(), RecordFineGrainedBinaryPrediction(), RecordFullRankings()]

    if args.step is not None:
        if args.step == "latest":
            checkpoint = model_dir.get_latest_checkpoint()
        else:
            checkpoint = model_dir.get_checkpoint(int(args.step))
    else:
        checkpoint = model_dir.get_best_weights()
        if checkpoint is not None:
            print("Using best weights")
        else:
            print("Using latest checkpoint")
            checkpoint = model_dir.get_latest_checkpoint()

    model = model_dir.get_model()

    evaluation = trainer.test(model, evaluators, {'dev_full': data},
                              corpus.get_resource_loader(), checkpoint, not args.no_ema, 10)['dev_full']

    # Print the scalar results in a two column table
    scalars = evaluation.scalars
    cols = list(sorted(scalars.keys()))
    table = [cols]
    header = ["Metric", ""]
    table.append([("%s" % scalars[x] if x in scalars else "-") for x in cols])
    print_table([header] + transpose_lists(table))


if __name__ == "__main__":
    main()
