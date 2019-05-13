import argparse
import string
from typing import List

import numpy as np
from tqdm import tqdm

from hotpot.data_handling.hotpot.hotpot_data import HotpotQuestion, HotpotQuestions
from hotpot.tokenizers import CoreNLPTokenizer
from hotpot.utils import flatten_iterable
from hotpot_evaluate_v1 import normalize_answer, f1_score


class FastNormalizedAnswerDetector(object):
    """ almost twice as fast and very,very close to NormalizedAnswerDetector's output """

    def __init__(self):
        # These come from the TrivaQA official evaluation script
        self.skip = {"a", "an", "the", ""}
        self.strip = string.punctuation + "".join([u"‘", u"’", u"´", u"`", "_"])

        self.answer_tokens = None

    def set_question(self, tokenized_aliases):
        self.answer_tokens = [[w.lower().strip(self.strip) for w in alias] for alias in tokenized_aliases]
        self.answer_tokens = [[w for w in alias if w not in self.skip] for alias in self.answer_tokens]

    def any_found(self, para: List[List[str]]):
        # Normalize the paragraph
        words = [w.lower().strip(self.strip) for w in flatten_iterable(para)]
        occurances = []
        for answer_ix, answer in enumerate(self.answer_tokens):
            # Locations where the first word occurs
            if len(answer) == 0:
                continue
            word_starts = [i for i, w in enumerate(words) if answer[0] == w]
            n_tokens = len(answer)

            # Advance forward until we find all the words, skipping over articles
            for start in word_starts:
                end = start + 1
                ans_token = 1
                while ans_token < n_tokens and end < len(words):
                    next = words[end]
                    if answer[ans_token] == next:
                        ans_token += 1
                        end += 1
                    elif next in self.skip:
                        end += 1
                    else:
                        break
                if n_tokens == ans_token:
                    occurances.append((start, end))
        return list(set(occurances))


def evaluate_question_detector(questions: List[HotpotQuestion], word_tokenize, detector,
                               reference_detector=None, compute_f1s=False):
    """ Just for debugging """
    n_no_docs = 0
    answer_per_q = []
    answer_f1s = []

    for question_ix, q in enumerate(tqdm(questions)):
        if q.answer in {'yes', 'no'} and q.q_type == 'comparison':
            continue
        tokenized_aliases = [word_tokenize(q.answer)]
        detector.set_question(tokenized_aliases)

        output = []
        for i, par in enumerate(q.supporting_facts):

            for s ,e in detector.any_found(par.sentences):
                output.append((i, s, e))

            if len(output) == 0 and reference_detector is not None:
                if reference_detector is not None:
                    reference_detector.set_question(tokenized_aliases)
                    detected = []
                    for j, par in enumerate(q.supporting_facts):
                        for s, e in reference_detector.any_found(par.sentences):
                            detected.append((j, s, e))

                    if len(detected) > 0:
                        print("Found a difference")
                        print(q.answer.normalized_aliases)
                        print(tokenized_aliases)
                        for p, s, e in detected:
                            token = flatten_iterable(q.supporting_facts[p].sentences)[s:e]
                            print(token)

        answer_per_q.append(output)

        if compute_f1s:
            f1s = []
            for p, s, e in output:
                token = flatten_iterable(q.supporting_facts[p].sentences)[s:e]
                answer = normalize_answer(" ".join(token))
                f1, _, _ = f1_score(answer, normalize_answer(q.answer))
                f1s.append(f1)
            answer_f1s.append(f1s)

    n_answers = sum(len(x) for x in answer_per_q)
    print("Found %d answers (av %.4f)" % (n_answers, n_answers / len(answer_per_q)))
    print("%.4f docs have answers" % np.mean([len(x) > 0 for x in answer_per_q]))
    if len(answer_f1s) > 0:
        print("Average f1 is %.4f" % np.mean(flatten_iterable(answer_f1s)))


def compute_answer_spans(questions: List[HotpotQuestion], detector, word_tok):

    for i, q in tqdm(enumerate(questions), total=len(questions)):
        if q.answer is None:
            continue
        tokenized_aliases = [word_tok(q.answer)]
        if len(tokenized_aliases) == 0:
            raise ValueError()
        detector.set_question(tokenized_aliases)
        q.gold_spans = []
        q.distractor_spans = []
        for par in q.supporting_facts + q.distractors:
            spans = []
            for s, e in detector.any_found(par.sentences):
                spans.append((s, e - 1))  # turn into inclusive span

            if len(spans) == 0:
                spans = np.zeros((0, 2), dtype=np.int32)
            else:
                spans = np.array(spans, dtype=np.int32)
            if par in q.supporting_facts:
                q.gold_spans.append(spans)
            else:
                q.distractor_spans.append(spans)

    return questions


def main(evaluate=False):
    corpus = HotpotQuestions()
    dev_qs = corpus.get_dev()
    train_qs = corpus.get_train()
    tokenizer = CoreNLPTokenizer()

    def tokenize(text):
        return tokenizer.tokenize(text).words()

    if evaluate:
        print("Train:")
        evaluate_question_detector(train_qs, tokenize, FastNormalizedAnswerDetector(), compute_f1s=True)
        print("Dev:")
        evaluate_question_detector(dev_qs, tokenize, FastNormalizedAnswerDetector(), compute_f1s=True)
    else:
        train = compute_answer_spans(train_qs, FastNormalizedAnswerDetector(), tokenize)
        dev = compute_answer_spans(dev_qs, FastNormalizedAnswerDetector(), tokenize)
        HotpotQuestions.make_corpus(train, dev)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Add answer spans to Hotpot questions")
    parser.add_argument('--eval', action='store_true')
    args = parser.parse_args()

    main(args.eval)
