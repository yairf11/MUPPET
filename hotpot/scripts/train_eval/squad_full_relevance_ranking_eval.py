import argparse
from typing import List, Tuple
import itertools
import numpy as np
from sklearn import metrics
import pickle

from hotpot import trainer
from hotpot.data_handling.dataset import ClusteredBatcher, multiple_contexts_len
from hotpot.data_handling.relevance_training_data import BinaryQuestionAndParagraphs
from hotpot.data_handling.squad.squad_data import SquadRelevanceCorpus
from hotpot.data_handling.squad.squad_relevance_training_data import SquadTextLengthPreprocessor, \
    SquadFullQuestionParagraphPairsDataset, SquadFullDocumentDataset
from hotpot.evaluator import BinaryClassificationEvaluator, Evaluator, Evaluation
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

        for num_distractors in [0, 1]:
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
        for num_distractors in [0, 1]:
            scalars = add_metrics(scalars, f'{num_distractors}_distractors',
                                  results[f'{num_distractors}_distractors_trues'],
                                  results[f'{num_distractors}_distractors_preds'], True)

        return Evaluation(scalars, results)


class RecordFullRankings(Evaluator):
    def tensors_needed(self, prediction: BinaryPrediction):
        probs = prediction.get_probs()
        return dict(probs=probs)

    def evaluate(self, data: List[BinaryQuestionAndParagraphs], true_len, **kwargs):
        probs = kwargs['probs']

        if len(probs.shape) == 2:
            probs = probs[:, 1]

        data_with_pos_probs = list(zip(data, probs))

        results = {}

        results['question_groups'] = {key: sorted(list(group), key=lambda x: x[1], reverse=True) for key, group in
                                      itertools.groupby(sorted(data_with_pos_probs, key=lambda x: x[0].question_id),
                                                        key=lambda x: x[0].question_id)}
        results['per_question_rank'] = {key: rank + 1 for key, group in results['question_groups'].items()
                                        for rank, sample in enumerate(group) if sample[0].label == 1}

        results['per_question_errors'] = {key: (group[0], group[results['per_question_rank'][key]-1])
                                          for key, group in results['question_groups'].items()
                                          if results['per_question_rank'][key] > 1}

        scalars = {}

        def add_metrics(metrics_dict, prefix, per_question_ranks):
            metrics_dict[f'{prefix}/P@1'] = np.mean([rank == 1 for rank in per_question_ranks.values()])
            metrics_dict[f'{prefix}/mean_rank'] = np.mean(list(per_question_ranks.values()))
            metrics_dict[f'{prefix}/MRR'] = np.mean([1.0 / x for x in per_question_ranks.values()])
            return metrics_dict

        scalars = add_metrics(scalars, 'global', results['per_question_rank'])

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
    parser.add_argument('--no-ema', action="store_true", help="Don't use EMA weights even if they exist")
    parser.add_argument('--per-doc', action='store_true', help="Whether to test only against full doc, or against "
                                                               "distractors.")
    parser.add_argument('--save-errors', default=None, type=str)
    args = parser.parse_args()

    model_dir = ModelDir(args.model)

    corpus = SquadRelevanceCorpus()
    # if args.corpus == "dev":
    #     questions = corpus.get_dev()
    # else:
    #     questions = corpus.get_train()
    questions = corpus.get_dev()

    question_preprocessor = SquadTextLengthPreprocessor(600)
    questions = [question_preprocessor.preprocess(x) for x in questions
                 if (question_preprocessor.preprocess(x) is not None)]

    if args.sample_questions:
        questions = sorted(questions, key=lambda x: x.question_id)
        np.random.RandomState(0).shuffle(questions)
        questions = questions[:args.sample_questions]

    batcher = ClusteredBatcher(args.batch_size, multiple_contexts_len, truncate_batches=True)
    if args.per_doc:
        data = SquadFullDocumentDataset(questions, batcher, corpus.dev_title_to_document)
    else:
        data = SquadFullQuestionParagraphPairsDataset(questions, batcher)

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

    if args.save_errors is not None:
        errors_dict = evaluation.per_sample['per_question_errors']

        def format_error(wrong: Tuple[BinaryQuestionAndParagraphs, float],
                         correct: Tuple[BinaryQuestionAndParagraphs, float]):
            question = ' '.join(wrong[0].question)
            qid = wrong[0].question_id
            wrong_text = ' '.join(wrong[0].paragraphs[0])
            wrong_score = wrong[1]
            correct_text = ' '.join(correct[0].paragraphs[0])
            correct_score = correct[1]
            return f"Question: {question}, ID: {qid}\n" \
                   f"Incorrect First Place: (score: {wrong_score})\n{wrong_text}\n" \
                   f"Correct Passage: (score: {correct_score})\n{correct_text}\n"

        with open(args.save_errors, 'wt') as f:
            for false_par, true_par in errors_dict.values():
                f.write(format_error(false_par, true_par))

    # Print the scalar results in a two column table
    scalars = evaluation.scalars
    cols = list(sorted(scalars.keys()))
    table = [cols]
    header = ["Metric", ""]
    table.append([("%s" % scalars[x] if x in scalars else "-") for x in cols])
    print_table([header] + transpose_lists(table))


if __name__ == "__main__":
    main()
