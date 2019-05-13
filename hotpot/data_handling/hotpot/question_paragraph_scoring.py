""" Add a tf-idf score for each question & paragraph """
import argparse
from typing import List
from scipy.sparse import vstack
from multiprocessing import Pool as ProcessPool

from tqdm import tqdm

from hotpot.data_handling.hotpot.hotpot_data import HotpotQuestion, HotpotQuestions
from hotpot.tfidf_retriever.tfidf_doc_ranker import TfidfDocRanker
from hotpot.utils import flatten_iterable

PROCESS_RANKER = None


def init():
    global PROCESS_RANKER
    PROCESS_RANKER = TfidfDocRanker()


def assign_scores(question: HotpotQuestion):
    question_spvec = PROCESS_RANKER.text2spvec(question.question_tokens, tokenized=True)
    paragraphs = [flatten_iterable(par.sentences) for par in (question.supporting_facts + question.distractors)]
    pars_spvecs = [PROCESS_RANKER.text2spvec(x, tokenized=True) for x in paragraphs]
    pars_spvecs = vstack(pars_spvecs)
    scores = pars_spvecs.dot(question_spvec.toarray().squeeze(axis=0))
    question.gold_scores = scores[:len(question.supporting_facts)].tolist()
    question.distractor_scores = scores[len(question.supporting_facts):].tolist()
    return question


def score_all_questions(num_workers):
    print("Loading data...")
    corpus = HotpotQuestions()
    train = corpus.get_train()
    dev = corpus.get_dev()

    workers = ProcessPool(
        num_workers,
        initializer=init,
        initargs=[]
    )

    print("Scoring train...")
    new_train = []
    with tqdm(total=len(train)) as pbar:
        for q in tqdm(workers.imap_unordered(assign_scores, train)):
            new_train.append(q)
            pbar.update()

    print("Scoring dev...")
    new_dev = []
    with tqdm(total=len(dev)) as pbar:
        for q in tqdm(workers.imap_unordered(assign_scores, dev)):
            new_dev.append(q)
            pbar.update()

    HotpotQuestions.make_corpus(new_train, new_dev)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Add scores to Hotpot questions")
    parser.add_argument('--num-workers', type=int, default=8, help='Number of CPU processes')
    args = parser.parse_args()

    score_all_questions(args.num_workers)
