from typing import Dict, Any, List
from threading import Thread
import tensorflow as tf
import numpy as np
from scipy.stats import kendalltau
from tqdm import tqdm
from sklearn import metrics
import itertools

from hotpot.configurable import Configurable
from hotpot.data_handling.dataset import Dataset
from hotpot.data_handling.qa_training_data import SpanQuestionAndParagraphs
from hotpot.data_handling.relevance_training_data import BinaryQuestionAndParagraphs, IterativeQuestionAndParagraphs
from hotpot.data_handling.span_data import compute_span_f1
from hotpot.model import Prediction, Model
from hotpot.nn.relevance_prediction import BinaryPrediction, MultipleBinaryPredictions
from hotpot.utils import flatten_iterable
from hotpot_evaluate_v1 import exact_match_score as hotpot_em_score
from hotpot_evaluate_v1 import f1_score as hotpot_f1_score


class Evaluation(object):
    """
    Evaluation of model, includes scalar summaries and per-example records
    """

    def __init__(self, scalars: Dict[str, Any], per_sample: Dict[str, List] = None):
        self.scalars = scalars
        self.per_sample = per_sample

    def add(self, other):
        for k in self.scalars:
            if k in other.scalars:
                raise ValueError("Two evaluations had the same scalar key: " + k)
        self.scalars.update(other.scalars)

        if self.per_sample is None:
            self.per_sample = other.per_sample
        elif other.per_sample is not None:
            for k in self.per_sample:
                if k in other.per_sample:
                    raise ValueError("Two evaluations had the same per sample key: " + k)
            self.per_sample.update(other.per_sample)

    def add_prefix(self, prefix):
        self.scalars = {prefix + k: v for k, v in self.scalars.items()}
        self.per_sample = {prefix + k: v for k, v in self.per_sample.items()}

    def to_summaries(self, prefix):
        return [tf.Summary(value=[tf.Summary.Value(tag=prefix + k, simple_value=v)]) for k, v in self.scalars.items()]


class Evaluator(Configurable):
    """ Class to generate statistics on a model's output for some data"""

    def tensors_needed(self, prediction: Prediction):
        """ Return all tensor variables needed by this evaluator in a dict, the results will
        be passed into `build_summary` as numpy arrays """
        raise NotImplementedError()

    def evaluate(self, input: List, true_len, **kwargs) -> Evaluation:
        """
        Build a summary given the input data `input` and the result of the variables requested
        from `tensors_needed`. `true_len` is the total number of examples seen (or an approximation)
        excluding any pre-filtering that was done, its used for the case where some examples could not be
        processed by the model (e.g. too large) and were removed, but we still want to report
        accurate percentages on the entire dataset.
        """
        raise NotImplementedError()


class LossEvaluator(Evaluator):

    def tensors_needed(self, _):
        return dict(loss=tf.add_n(tf.get_collection(tf.GraphKeys.LOSSES)))

    def evaluate(self, data, true_len, loss):
        return Evaluation({"loss": np.mean(loss)})


class BinaryClassificationEvaluator(Evaluator):

    def tensors_needed(self, prediction: BinaryPrediction):
        probs = prediction.get_probs()
        preds = prediction.get_predictions()
        return dict(preds=preds, probs=probs)

    def evaluate(self, data: List[BinaryQuestionAndParagraphs], true_len, **kwargs):
        preds = kwargs['preds']
        probs = kwargs['probs']
        if len(probs.shape) == 2:
            probs = probs[:, 1]

        trues = np.array([point.label for point in data])

        out = {
            'recall': metrics.recall_score(y_pred=preds, y_true=trues),
            'precision': metrics.precision_score(y_pred=preds, y_true=trues),
            'f1_score': metrics.f1_score(y_pred=preds, y_true=trues),
            'roc_auc': metrics.roc_auc_score(y_score=probs, y_true=trues),
            'average_precision': metrics.average_precision_score(y_score=probs, y_true=trues),
            'accuracy': metrics.accuracy_score(y_pred=preds, y_true=trues)
        }

        prefix = 'binary-relevance/'
        return Evaluation({prefix + k: v for k, v in out.items()})


class IterativeRelevanceEvaluator(Evaluator):

    def tensors_needed(self, prediction: MultipleBinaryPredictions):
        first_probs = prediction.get_probs(pred_idx=0)
        first_preds = prediction.get_predictions(pred_idx=0)
        second_probs = prediction.get_probs(pred_idx=1)
        second_preds = prediction.get_predictions(pred_idx=1)
        return dict(first_preds=first_preds, first_probs=first_probs,
                    second_preds=second_preds, second_probs=second_probs)

    def evaluate(self, data: List[IterativeQuestionAndParagraphs], true_len, **kwargs):
        first_preds = kwargs['first_preds']
        first_probs = kwargs['first_probs']
        second_preds = kwargs['second_preds']
        second_probs = kwargs['second_probs']
        if len(first_probs.shape) == 2:
            first_probs = first_probs[:, 1]
        if len(second_probs.shape) == 2:
            second_probs = second_probs[:, 1]

        first_trues = np.array([point.first_label for point in data])
        second_trues = np.array([point.second_label for point in data])

        eval_dict = {}

        for name, preds, probs, trues in [("first", first_preds, first_probs, first_trues),
                                          ("second", second_preds, second_probs, second_trues)]:
            out = {
                'recall': metrics.recall_score(y_pred=preds, y_true=trues),
                'precision': metrics.precision_score(y_pred=preds, y_true=trues),
                'f1_score': metrics.f1_score(y_pred=preds, y_true=trues),
                # 'roc_auc': metrics.roc_auc_score(y_score=probs, y_true=trues),
                'average_precision': metrics.average_precision_score(y_score=probs, y_true=trues),
                'accuracy': metrics.accuracy_score(y_pred=preds, y_true=trues)
            }

            prefix = f'iterative-relevance/{name}/'
            eval_dict.update({prefix + k: v for k, v in out.items()})

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
        results['per_question_second_ranks'] = {key: [rank + 1
                                                      for rank, sample in enumerate(group['second_sorted'])
                                                      if sample[0].second_label == 1]
                                                for key, group in results['question_groups'].items()}

        def add_metrics(metrics_dict, prefix, per_question_ranks):
            vals = [q_vals for q_vals in per_question_ranks.values() if len(q_vals) > 0]
            metrics_dict[f'{prefix}/P@1'] = np.mean([any(x == 1 for x in ranks)
                                                     for ranks in vals])
            metrics_dict[f'{prefix}/Mean_Rank'] = np.mean([np.mean(ranks)
                                                           for ranks in vals])
            metrics_dict[f'{prefix}/MRR'] = np.mean([1.0 / np.min(x) for x in vals])
            return metrics_dict

        for iteration in ['first', 'second']:
            per_q = add_metrics({}, f'{iteration}', results[f'per_question_{iteration}_ranks'])

            prefix = f'iterative-relevance-per-question/{iteration}/'
            eval_dict.update({prefix + k: v for k, v in per_q.items()})

        return Evaluation(eval_dict)


def hotpot_span_scores(data: List[SpanQuestionAndParagraphs],
                       prediction):
    scores = np.zeros((len(data), 6))
    for i, question in enumerate(data):
        answer = question.answer

        pred_span = prediction[i]
        # For Hotpot we have generally called join-on-spaces approach good enough, since the answers here
        # tend to be short and the gold standard has better normalization. Possibly could get a very
        # small gain using the original text
        pred_text = " ".join(question.paragraphs[0][pred_span[0]:pred_span[1] + 1])

        span_correct = False
        span_max_f1 = 0
        text_correct = 0
        text_max_f1 = 0

        for word_start, word_end in question.spans:
            answer_span = (word_start, word_end)
            span_max_f1 = max(span_max_f1, compute_span_f1(answer_span, pred_span))
            if answer_span == tuple(pred_span):
                span_correct = True

        f1, p, r = hotpot_f1_score(pred_text, answer)
        correct = hotpot_em_score(pred_text, answer)
        text_correct = max(text_correct, correct)
        text_max_f1 = max(text_max_f1, f1)

        scores[i] = [span_correct, span_max_f1, text_correct, text_max_f1, p, r]
    return scores


class MultiParagraphSpanEvaluator(Evaluator):
    """
    Measure error with multiple paragraphs per a question.

    Evaluation is a bit tricky in this case, since we are generally sampling paragraphs
    each epoch we can't report exact numbers as your would see when running the
    evaluation scripts. Instead we report some numbers aimed to get an approximate idea of what is going on:

    1: question-text-{em|f1}, accuracy on questions-document pairs (or just questions if `per_doc=False`)
       using all sampled paragraphs when taking the model's highest confidence answer.
       This tends to be an overly-confident estimate since the sampled paragraphs are usually biased
       towards using paragraphs that contain the correct answer
    2: The Kendel Tau relation between the model's confidence and the paragraph's f1/em score,
       (if `k_tau=True`) intended to measure how valid the model's confidence score is
       when it comes to ranking.
    """

    def __init__(self, bound: int, eval, k_tau=True, yes_no_option=False, supporting_facts_option=False):
        if eval not in ["hotpot"]:
            raise ValueError()
        self.bound = bound
        self.eval = eval
        self.k_tau = k_tau
        self.yes_no_option = yes_no_option
        self.supporting_facts_option = supporting_facts_option

    def tensors_needed(self, prediction):
        span, score = prediction.get_best_span(self.bound)
        tensor_dict = dict(span=span, score=score)
        if self.yes_no_option:
            is_yes_no_scores = prediction.get_is_yes_no_scores()
            yes_or_no_scores = prediction.get_yes_or_no_scores()
            tensor_dict.update(dict(is_yes_no_scores=is_yes_no_scores, yes_or_no_scores=yes_or_no_scores))
        if self.supporting_facts_option:
            sentence_scores = prediction.get_sentence_scores()
            tensor_dict.update(dict(sentence_scores=sentence_scores))
        return tensor_dict

    def evaluate(self, data: List[SpanQuestionAndParagraphs], true_len, **kwargs):
        best_spans = kwargs["span"]
        span_logits = kwargs["score"]
        if self.eval == "hotpot":
            scores = hotpot_span_scores(data, best_spans)
        else:
            raise RuntimeError()

        selected_paragraphs = {}
        for i, point in enumerate(data):
            key = point.question_id
            if key not in selected_paragraphs:
                selected_paragraphs[key] = i
            elif span_logits[i] > span_logits[selected_paragraphs[key]]:
                selected_paragraphs[key] = i

        if self.yes_no_option:
            is_yes_no_scores = kwargs["is_yes_no_scores"]
            yes_or_no_scores = kwargs["yes_or_no_scores"]
            yes_no_all_scores = np.concatenate([is_yes_no_scores, yes_or_no_scores], axis=1)
            question_id_yes_no_scores = {}  # each value is a ndarray of [not yes/no, yes/no, no, yes] max scores
            for i, point in enumerate(data):
                key = point.question_id
                if key not in question_id_yes_no_scores:
                    question_id_yes_no_scores[key] = yes_no_all_scores[i]
                else:
                    question_id_yes_no_scores[key] = np.maximum(question_id_yes_no_scores[key],
                                                                yes_no_all_scores[i])
            yes_no_qids = [qid for qid, scores in question_id_yes_no_scores.items() if scores[1] > scores[0]]
            qid_to_answer = {x.question_id: x.answer for x in data}
            for qid in yes_no_qids:
                correct_yes = question_id_yes_no_scores[qid][3] >= question_id_yes_no_scores[qid][2] \
                              and qid_to_answer[qid] == 'yes'
                correct_no = question_id_yes_no_scores[qid][2] > question_id_yes_no_scores[qid][3] \
                             and qid_to_answer[qid] == 'no'
                scores[selected_paragraphs[qid]][2:] = correct_yes or correct_no  # for yes/no, em=f1=p=r

        if self.supporting_facts_option:
            sentence_scores = kwargs['sentence_scores']
            sp_scores = np.zeros((len(data), 4))
            for i, sample in enumerate(data):
                sentence_labels = np.zeros(max(sample.sentence_segments[0])+1, dtype=np.int32)
                sentence_labels[sample.supporting_facts] = 1
                sentence_preds = sentence_scores[i][:len(sentence_labels)] >= 0.5
                precision = metrics.precision_score(y_pred=sentence_preds, y_true=sentence_labels)
                recall = metrics.recall_score(y_pred=sentence_preds, y_true=sentence_labels)
                f1 = metrics.f1_score(y_pred=sentence_preds, y_true=sentence_labels)
                em = f1 == 1
                sp_scores[i] = [em, f1, precision, recall]

        selected_paragraphs = list(selected_paragraphs.values())

        out = {
            "question-text-em": scores[selected_paragraphs, 2].mean(),
            "question-text-f1": scores[selected_paragraphs, 3].mean(),
        }

        if self.k_tau:
            out["text-em-k-tau"] = kendalltau(span_logits, scores[:, 2])[0]
            out["text-f1-k-tau"] = kendalltau(span_logits, scores[:, 3])[0]

        if self.yes_no_option:
            qid2true_yes_no = {x.question_id: x.answer in {'yes', 'no'} and x.q_type == 'comparison' for x in data}
            true_yes_nos = np.array([qid2true_yes_no[qid] for qid in question_id_yes_no_scores])
            pred_yes_nos = np.array([qid in yes_no_qids for qid in question_id_yes_no_scores])
            out['yes-no-recall'] = metrics.recall_score(y_pred=pred_yes_nos, y_true=true_yes_nos)
            out['yes-no-precision'] = metrics.precision_score(y_pred=pred_yes_nos, y_true=true_yes_nos)
            out['yes-no-f1_score'] = metrics.f1_score(y_pred=pred_yes_nos, y_true=true_yes_nos)

        if self.supporting_facts_option:
            out['sp-em'] = sp_scores[selected_paragraphs, 0].mean()
            out['sp-f1'] = sp_scores[selected_paragraphs, 1].mean()
            p_ans = scores[selected_paragraphs, 4]
            p_sp = sp_scores[selected_paragraphs, 2]
            p_joint = p_ans * p_sp
            r_ans = scores[selected_paragraphs, 5]
            r_sp = sp_scores[selected_paragraphs, 3]
            r_joint = r_ans * r_sp
            f1_joint = (2 * p_joint * r_joint) / (p_joint + r_joint + 1e-27)
            em_joint = np.isclose(f1_joint, 1.)
            out['joint-em'] = em_joint.mean()
            out['joint-f1'] = f1_joint.mean()

        prefix = "b%d/" % self.bound
        return Evaluation({prefix + k: v for k, v in out.items()})

    def __setstate__(self, state):
        super().__setstate__(state)


class EvaluatorRunner(object):
    """ Knows how to run a list of evaluators """

    def __init__(self, evaluators: List[Evaluator], model: Model):
        self.evaluators = evaluators
        self.tensors_needed = None
        self.model = model

    def set_input(self, prediction: Prediction):
        tensors_needed = []
        for ev in self.evaluators:
            tensors_needed.append(ev.tensors_needed(prediction))
        self.tensors_needed = tensors_needed

    def run_evaluators(self, sess: tf.Session, dataset: Dataset, name, n_sample=None, feed_dict=None) -> Evaluation:
        all_tensors_needed = list(set(flatten_iterable(x.values() for x in self.tensors_needed)))

        tensors = {x: [] for x in all_tensors_needed}

        if n_sample is None:
            batches, n_batches = dataset.get_epoch(), len(dataset)
        else:
            batches, n_batches = dataset.get_samples(n_sample)

        data_used = []

        for batch in tqdm(batches, total=n_batches, desc=name, ncols=80):
            feed_dict = self.model.encode(batch, is_train=False)
            output = sess.run(all_tensors_needed, feed_dict=feed_dict)
            data_used += batch
            for i in range(len(all_tensors_needed)):
                tensors[all_tensors_needed[i]].append(output[i])

        # flatten the input
        for k in all_tensors_needed:
            v = tensors[k]
            if len(k.shape) == 0:
                v = np.array(v)  # List of scalars
            elif any(x is None for x in k.shape.as_list()):
                # Variable sized tensors, so convert to flat python-list
                v = flatten_iterable(v)
            else:
                v = np.concatenate(v, axis=0)  # concat along the batch dim
            tensors[k] = v

        percent_filtered = dataset.percent_filtered()
        if percent_filtered is None:
            true_len = len(data_used)
        else:
            true_len = len(data_used) * 1 / (1 - percent_filtered)

        combined = None
        for ev, needed in zip(self.evaluators, self.tensors_needed):
            args = {k: tensors[v] for k, v in needed.items()}
            evaluation = ev.evaluate(data_used, true_len, **args)
            if evaluation is None:
                raise ValueError(ev)
            if combined is None:
                combined = evaluation
            else:
                combined.add(evaluation)

        return combined


class AysncEvaluatorRunner(object):
    """ Knows how to run a list of evaluators use a tf.Queue to feed in the data """

    def __init__(self, evaluators: List[Evaluator], model: Model, queue_size: int):
        placeholders = model.get_placeholders()
        self.eval_queue = tf.FIFOQueue(queue_size, [x.dtype for x in placeholders],
                                       name="eval_queue")
        self.enqueue_op = self.eval_queue.enqueue(placeholders)
        self.dequeue_op = self.eval_queue.dequeue()
        self.close_queue = self.eval_queue.close(True)

        # Queue in this form has not shape info, so we have to add it in back here
        for x, p in zip(placeholders, self.dequeue_op):
            p.set_shape(x.shape)
        self.evaluators = evaluators
        self.queue_size = self.eval_queue.size()
        self.model = model
        self.tensors_needed = None

    def set_input(self, prediction: Prediction):
        tensors_needed = []
        for ev in self.evaluators:
            tensors_needed.append(ev.tensors_needed(prediction))
        self.tensors_needed = tensors_needed

    def run_evaluators(self, sess: tf.Session, dataset, name, n_sample, feed_dict, disable_tqdm=False) -> Evaluation:
        all_tensors_needed = list(set(flatten_iterable(x.values() for x in self.tensors_needed)))

        tensors = {x: [] for x in all_tensors_needed}

        data_used = []
        if n_sample is None:
            batches, n_batches = dataset.get_epoch(), len(dataset)
        else:
            batches, n_batches = dataset.get_samples(n_sample)

        def enqueue_eval():
            try:
                for data in batches:
                    encoded = self.model.encode(data, False)
                    data_used.append(data)
                    sess.run(self.enqueue_op, encoded)
            except Exception as e:
                sess.run(self.close_queue)  # Crash the main thread
                raise e
            # we should run out of batches and exit gracefully

        th = Thread(target=enqueue_eval)

        th.daemon = True
        th.start()
        for _ in tqdm(range(n_batches), total=n_batches, desc=name, ncols=80, disable=disable_tqdm):
            output = sess.run(all_tensors_needed, feed_dict=feed_dict)
            for i in range(len(all_tensors_needed)):
                tensors[all_tensors_needed[i]].append(output[i])
        th.join()

        if sess.run(self.queue_size) != 0:
            raise RuntimeError("All batches should be been consumed")

        # flatten the input
        for k in all_tensors_needed:
            v = tensors[k]
            if len(k.shape) == 0:
                v = np.array(v)  # List of scalars -> array
            elif any(x is None for x in k.shape.as_list()[1:]):
                # Variable sized tensors, so convert to flat python-list
                v = flatten_iterable(v)
            else:
                v = np.concatenate(v, axis=0)  # concat along the batch dim
            tensors[k] = v

        # flatten the data if it consists of batches
        if isinstance(data_used[0], List):
            data_used = flatten_iterable(data_used)

        if dataset.percent_filtered() is None:
            true_len = len(data_used)
        else:
            true_len = len(data_used) * 1 / (1 - dataset.percent_filtered())

        combined = None
        for ev, needed in zip(self.evaluators, self.tensors_needed):
            args = {k: tensors[v] for k, v in needed.items()}
            evaluation = ev.evaluate(data_used, true_len, **args)
            if combined is None:
                combined = evaluation
            else:
                combined.add(evaluation)

        return combined
