import argparse
from typing import List

import numpy as np
from sklearn import metrics

from hotpot import trainer
from hotpot.data_handling.dataset import ClusteredBatcher, multiple_contexts_len
from hotpot.data_handling.hotpot.hotpot_data import HotpotQuestions
from hotpot.data_handling.relevance_training_data import BinaryQuestionAndParagraphs
from hotpot.data_handling.hotpot.hotpot_relevance_training_data import \
    HotpotStratifiedBinaryQuestionParagraphPairsDataset, HotpotTextLengthPreprocessor, HotpotQuestionFilter
from hotpot.evaluator import Evaluator, BinaryClassificationEvaluator, Evaluation
from hotpot.model_dir import ModelDir
from hotpot.nn.relevance_prediction import BinaryPrediction
from hotpot.utils import transpose_lists, print_table


class RecordFineGrainedBinaryPrediction(Evaluator):
    def tensors_needed(self, prediction: BinaryPrediction):
        probs = prediction.get_probs()
        preds = prediction.get_predictions()
        return dict(preds=preds, probs=probs)

    def evaluate(self, data: List[BinaryQuestionAndParagraphs], true_len, **kwargs):
        preds = kwargs['preds']
        probs = kwargs['probs']

        results = {}

        results['trues'] = np.array([point.label for point in data])
        results['preds'] = preds
        results['comparison_trues'] = np.array([point.label for point in data if point.q_type == "comparison"])
        results['comparison_preds'] = np.array([preds[i] for i in range(len(preds)) if data[i].q_type == 'comparison'])
        results['bridge_trues'] = np.array([point.label for point in data if point.q_type == "bridge"])
        results['bridge_preds'] = np.array([preds[i] for i in range(len(preds)) if data[i].q_type == 'bridge'])

        for num_distractors in [0, 1, 2]:
            results[f'{num_distractors}_distractors_trues'] = np.array([point.label for point in data
                                                                        if point.num_distractors == num_distractors])
            results[f'{num_distractors}_distractors_preds'] = np.array([preds[i] for i in range(len(preds))
                                                                        if data[i].num_distractors == num_distractors])

        scalars = {}

        def add_metrics(metrics_dict, prefix, y_true, y_pred, only_accuracy=False):
            if only_accuracy:
                metrics_dict[f'{prefix}/accuracy'] = metrics.accuracy_score(y_pred=y_pred, y_true=y_true)
            else:
                metrics_dict[f'{prefix}/recall'] = metrics.recall_score(y_pred=y_pred, y_true=y_true)
                metrics_dict[f'{prefix}/precision'] = metrics.precision_score(y_pred=y_pred, y_true=y_true)
                metrics_dict[f'{prefix}/f1_score'] = metrics.f1_score(y_pred=y_pred, y_true=y_true)
            return metrics_dict

        scalars = add_metrics(scalars, 'global', results['trues'], results['preds'], False)
        scalars = add_metrics(scalars, 'comparison', results['comparison_trues'], results['comparison_preds'], False)
        scalars = add_metrics(scalars, 'bridge', results['bridge_trues'], results['bridge_preds'], False)
        for num_distractors in [0, 1, 2]:
            scalars = add_metrics(scalars, f'{num_distractors}_distractors',
                                  results[f'{num_distractors}_distractors_trues'],
                                  results[f'{num_distractors}_distractors_preds'], True)

        return Evaluation(scalars, results)

def main():
    parser = argparse.ArgumentParser(description='Evaluate a model on SQuAD')
    parser.add_argument('model', help='model directory to evaluate')
    parser.add_argument('-n', '--sample_questions', type=int, default=None,
                        help="(for testing) run on a subset of questions")
    parser.add_argument('-b', '--batch_size', type=int, default=64,
                        help="Batch size, larger sizes can be faster but uses more memory")
    parser.add_argument('-s', '--step', default=None,
                        help="Weights to load, can be a checkpoint step or 'latest'")
    # parser.add_argument('-c', '--corpus', choices=["dev", "train"], default="dev")
    parser.add_argument('--num_runs', type=int, default=1,
                        help="Number of different seeds to test on, for more accurate results")
    parser.add_argument('--no_ema', action="store_true", help="Don't use EMA weights even if they exist")
    args = parser.parse_args()

    model_dir = ModelDir(args.model)

    corpus = HotpotQuestions()
    # if args.corpus == "dev":
    #     questions = corpus.get_dev()
    # else:
    #     questions = corpus.get_train()
    questions = corpus.get_dev()

    question_filter = HotpotQuestionFilter(2)  # TODO add option to cancel this, and more fine-grained analysis
    question_preprocessor = HotpotTextLengthPreprocessor(600)
    questions = [question_preprocessor.preprocess(x) for x in questions
                 if (question_filter.keep(x) and question_preprocessor.preprocess(x) is not None)]

    if args.sample_questions:
        np.random.RandomState(0).shuffle(sorted(questions, key=lambda x: x.question_id))
        questions = questions[:args.sample_questions]

    batcher = ClusteredBatcher(args.batch_size, multiple_contexts_len, truncate_batches=True)
    datasets = [HotpotStratifiedBinaryQuestionParagraphPairsDataset(questions, batcher, fixed_dataset=True, sample_seed=i)
                for i in range(args.num_runs)]

    evaluators = [BinaryClassificationEvaluator(), RecordFineGrainedBinaryPrediction()]

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

    evaluation = trainer.test(model, evaluators, {f'dev_seed_{i}': dataset for i, dataset in enumerate(datasets)},
                              corpus.get_resource_loader(), checkpoint, not args.no_ema, 10)

    scalars = {key: np.mean([eval_dict.scalars[key] for eval_dict in evaluation.values()])
               for key in evaluation['dev_seed_0'].scalars.keys()}

    # Print the scalar results in a two column table
    # scalars = evaluation.scalars
    cols = list(sorted(scalars.keys()))
    table = [cols]
    header = ["Metric", ""]
    table.append([("%s" % scalars[x] if x in scalars else "-") for x in cols])
    print_table([header] + transpose_lists(table))


if __name__ == "__main__":
    main()
