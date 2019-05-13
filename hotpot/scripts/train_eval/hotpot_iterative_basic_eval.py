import argparse
from typing import List, Tuple
import numpy as np
import itertools

from hotpot import trainer
from hotpot.data_handling.dataset import ClusteredBatcher, multiple_contexts_len
from hotpot.data_handling.hotpot.hotpot_data import HotpotQuestions
from hotpot.data_handling.hotpot.hotpot_relevance_training_data import HotpotFullIterativeDataset, \
    HotpotTextLengthPreprocessor
from hotpot.data_handling.relevance_training_data import IterativeQuestionAndParagraphs
from hotpot.evaluator import Evaluator, Evaluation, IterativeRelevanceEvaluator
from hotpot.model_dir import ModelDir
from hotpot.nn.relevance_prediction import MultipleBinaryPredictions
from hotpot.utils import transpose_lists, print_table


class RecordFullIterativeRankings(Evaluator):
    def __init__(self, multiply_iteration_probs=False):
        self.multiply_iteration_probs = multiply_iteration_probs

    def tensors_needed(self, prediction: MultipleBinaryPredictions):
        first_probs, second_probs = prediction.get_probs(0), prediction.get_probs(1)
        return dict(first_probs=first_probs, second_probs=second_probs)

    def evaluate(self, data: List[IterativeQuestionAndParagraphs], true_len, **kwargs):
        first_probs, second_probs = kwargs['first_probs'], kwargs['second_probs']
        if len(first_probs.shape) == 2:
            first_probs = first_probs[:, 1]
        if len(second_probs.shape) == 2:
            second_probs = second_probs[:, 1]
        if self.multiply_iteration_probs:
            second_probs *= first_probs
        data_with_pos_probs = list(zip(data, first_probs, second_probs))

        results = {}

        # first create a list for each group because it is an iterator
        results['question_groups'] = {key: list(group)
                                      for key, group in
                                      itertools.groupby(sorted(data_with_pos_probs, key=lambda x: x[0].question_id),
                                                        key=lambda x: x[0].question_id)}
        # now calculate what you want
        results['question_groups'] = {key: {'first_sorted': sorted(group, key=lambda x: x[1], reverse=True),
                                            'second_sorted': sorted(group, key=lambda x: x[2], reverse=True)}
                                      for key, group in results['question_groups'].items()}
        # remove duplicates for first paragraphs
        for key in results['question_groups'].keys():
            first_group = results['question_groups'][key]['first_sorted']
            seen = []
            ps = [x for x in first_group if not (x[0].paragraphs[0] in seen or seen.append(x[0].paragraphs[0]))]
            results['question_groups'][key]['first_sorted'] = ps

        results['per_question_first_ranks'] = {key: [rank + 1
                                                     for rank, sample in enumerate(group['first_sorted'])
                                                     if sample[0].first_label == 1]
                                               for key, group in results['question_groups'].items()}
        results['bridge_per_question_first_ranks'] = \
            {key: ranks
             for key, ranks in results['per_question_first_ranks'].items()
             if results['question_groups'][key]['first_sorted'][0][0].q_type == 'bridge'}
        results['comparison_per_question_first_ranks'] = \
            {key: ranks
             for key, ranks in results['per_question_first_ranks'].items()
             if results['question_groups'][key]['first_sorted'][0][0].q_type == 'comparison'}

        results['per_question_second_ranks'] = {key: [rank + 1
                                                      for rank, sample in enumerate(group['second_sorted'])
                                                      if sample[0].second_label == 1]
                                                for key, group in results['question_groups'].items()}
        results['bridge_per_question_second_ranks'] = \
            {key: ranks
             for key, ranks in results['per_question_second_ranks'].items()
             if results['question_groups'][key]['second_sorted'][0][0].q_type == 'bridge'}
        results['comparison_per_question_second_ranks'] = \
            {key: ranks
             for key, ranks in results['per_question_second_ranks'].items()
             if results['question_groups'][key]['second_sorted'][0][0].q_type == 'comparison'}

        results['per_question_first_errors'] = {key:
                                                    (group['first_sorted'][0],
                                                     group['first_sorted'][
                                                         results['per_question_first_ranks'][key][0] - 1])
                                                for key, group in results['question_groups'].items()
                                                if results['per_question_first_ranks'][key][0] > 1}

        results['per_question_second_errors'] = {key:
                                                     (group['second_sorted'][0],
                                                      group['second_sorted'][
                                                          results['per_question_second_ranks'][key][0] - 1])
                                                 for key, group in results['question_groups'].items()
                                                 if results['per_question_second_ranks'][key][0] > 1}

        scalars = {}

        def add_metrics(metrics_dict, prefix, per_question_ranks):
            metrics_dict[f'{prefix}/P@1'] = np.mean([any(x == 1 for x in ranks)
                                                     for ranks in per_question_ranks.values()])
            metrics_dict[f'{prefix}/Mean_Rank'] = np.mean([np.mean(ranks)
                                                           for ranks in list(per_question_ranks.values())])
            metrics_dict[f'{prefix}/MRR'] = np.mean([1.0 / np.min(x) for x in per_question_ranks.values()])
            return metrics_dict

        for iteration in ['first', 'second']:
            scalars = add_metrics(scalars, f'global-{iteration}', results[f'per_question_{iteration}_ranks'])
            scalars = add_metrics(scalars, f'bridge-{iteration}', results[f'bridge_per_question_{iteration}_ranks'])
            scalars = add_metrics(scalars, f'comparison-{iteration}',
                                  results[f'comparison_per_question_{iteration}_ranks'])

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
    parser.add_argument('--no-ema', action="store_true", help="Don't use EMA weights even if they exist")
    parser.add_argument('--save-errors', default=None, type=str)
    parser.add_argument('--br-as-cp', action='store_true')
    parser.add_argument('--mult-probs', action='store_true')
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
    data = HotpotFullIterativeDataset(questions, batcher, bridge_as_comparison=args.br_as_cp)

    evaluators = [IterativeRelevanceEvaluator(), RecordFullIterativeRankings(multiply_iteration_probs=args.mult_probs)]

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
        first_errors_dict = evaluation.per_sample['per_question_first_errors']
        second_errors_dict = evaluation.per_sample['per_question_second_errors']

        def format_first_error(wrong: Tuple[IterativeQuestionAndParagraphs, float, float],
                               correct: Tuple[IterativeQuestionAndParagraphs, float, float]):
            question = ' '.join(wrong[0].question)
            qid = wrong[0].question_id
            q_type = wrong[0].q_type
            wrong_text = ' '.join(wrong[0].paragraphs[0])
            wrong_score = wrong[1]
            correct_text = ' '.join(correct[0].paragraphs[0])
            correct_score = correct[1]
            return f"Question: {question}, ID: {qid}, type: {q_type}\n" \
                   f"Incorrect First Place: (score: {wrong_score})\n{wrong_text}\n-\n" \
                   f"Correct Passage: (score: {correct_score})\n{correct_text}\n***\n"

        def format_second_error(wrong: Tuple[IterativeQuestionAndParagraphs, float, float],
                                correct: Tuple[IterativeQuestionAndParagraphs, float, float]):
            question = ' '.join(wrong[0].question)
            qid = wrong[0].question_id
            q_type = wrong[0].q_type
            wrong_texts = [' '.join(par) for par in wrong[0].paragraphs]
            wrong_first_score = wrong[1]
            wrong_final_score = wrong[2]
            correct_texts = [' '.join(par) for par in correct[0].paragraphs]
            correct_first_score = correct[1]
            correct_final_score = correct[2]
            return f"Question: {question}, ID: {qid}, type: {q_type}\n" \
                   f"Incorrect First Place Pair: (score: {wrong_final_score})\n" \
                   f"Paragraph 1 (score: {wrong_first_score})\n" \
                   f"{wrong_texts[0]}\n" \
                   f"Paragraph 2:\n" \
                   f"{wrong_texts[1]}\n-\n" \
                   f"Correct Pair: (score: {correct_final_score})\n" \
                   f"Paragraph 1 (score: {correct_first_score})\n" \
                   f"{correct_texts[0]}\n" \
                   f"Paragraph 2:\n" \
                   f"{correct_texts[1]}\n***\n"

        with open(args.save_errors, 'wt') as f:
            f.write("First paragraph errors:\n*****************************\n")
            for false_par, true_par in first_errors_dict.values():
                f.write(format_first_error(false_par, true_par))
            f.write("Second paragraph errors:\n*****************************\n")
            for false_par, true_par in second_errors_dict.values():
                f.write(format_second_error(false_par, true_par))

    # Print the scalar results in a two column table
    scalars = evaluation.scalars
    cols = list(sorted(scalars.keys()))
    table = [cols]
    header = ["Metric", ""]
    table.append([("%s" % scalars[x] if x in scalars else "-") for x in cols])
    print_table([header] + transpose_lists(table))


if __name__ == "__main__":
    main()
