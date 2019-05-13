import argparse
from collections import Counter
from typing import Optional, List
import itertools
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, strip_accents_unicode
from sklearn.metrics import pairwise_distances
from tqdm import tqdm
from sklearn.preprocessing import normalize
from scipy.sparse import vstack
from multiprocessing import Pool as ProcessPool


from hotpot.data_handling.squad.squad_data import SquadRelevanceCorpus, SquadQuestionWithDistractors, SquadQuestion, \
    SquadParagraph
from hotpot.data_handling.squad.squad_relevance_training_data import SquadTextLengthPreprocessor
from hotpot.tfidf_retriever.tfidf_doc_ranker import TfidfDocRanker
from hotpot.tfidf_retriever.utils import STOPWORDS


PROCESS_RANKER = None


def init():
    global PROCESS_RANKER
    PROCESS_RANKER = TfidfDocRanker()


def get_rank_in_distractors(question: SquadQuestionWithDistractors) -> int:
    def per_word_prepro(word):
        return strip_accents_unicode(word.lower())

    def tf_idf_prepro(text_or_list):
        if type(text_or_list) == list:
            return [per_word_prepro(x) for x in text_or_list]
        return per_word_prepro(text_or_list)

    def tf_idf_tok(word_or_list):
        if type(word_or_list) == list:
            return word_or_list
        return [word_or_list]

    if PROCESS_RANKER is None:
        vectorizer = TfidfVectorizer(preprocessor=tf_idf_prepro, tokenizer=tf_idf_tok, stop_words=STOPWORDS)
        question_features = vectorizer.fit_transform([question.question])
        question_pars_features = vectorizer.transform([x.par_text for x in [question.paragraph]+question.distractors])
        distances = pairwise_distances(question_features, question_pars_features, "cosine").squeeze(axis=0)
        gold_rank = list(distances.argsort()).index(0) + 1
    else:
        question_spvec = PROCESS_RANKER.text2spvec(question.question, tokenized=True)
        pars_spvecs = [PROCESS_RANKER.text2spvec(x.par_text, tokenized=True) for x in
                       [question.paragraph]+question.distractors]
        pars_spvecs = normalize(vstack(pars_spvecs))
        scores = pars_spvecs.dot(question_spvec.toarray().squeeze(axis=0))
        gold_rank = list((-scores).argsort()).index(0) + 1
    # print(gold_rank)
    # print(' '.join(question.question))
    # print(list(zip([' '.join(x.par_text) for x in [question.paragraph]+question.distractors], scores)))
    return gold_rank


def get_rank_in_document(question: SquadQuestion, paragraphs: List[SquadParagraph]) -> int:
    """ Note that we are assuming contiguous ids in the paragraphs """
    def per_word_prepro(word):
        return strip_accents_unicode(word.lower())

    def tf_idf_prepro(text_or_list):
        if type(text_or_list) == list:
            return [per_word_prepro(x) for x in text_or_list]
        return per_word_prepro(text_or_list)

    def tf_idf_tok(word_or_list):
        if type(word_or_list) == list:
            return word_or_list
        return [word_or_list]

    gold_id = question.paragraph.par_id
    # for safety:
    for idx, par in enumerate(paragraphs):
        if par.par_id == gold_id:
            if idx != par.par_id:
                raise ValueError("return to safety!")

    if PROCESS_RANKER is None:
        vectorizer = TfidfVectorizer(preprocessor=tf_idf_prepro, tokenizer=tf_idf_tok, stop_words=STOPWORDS)
        question_features = vectorizer.fit_transform([question.question])
        question_pars_features = vectorizer.transform([x.par_text for x in paragraphs])
        distances = pairwise_distances(question_features, question_pars_features, "cosine").squeeze(axis=0)
        gold_rank = list(distances.argsort()).index(gold_id) + 1
    else:
        question_spvec = PROCESS_RANKER.text2spvec(question.question, tokenized=True)
        pars_spvecs = [PROCESS_RANKER.text2spvec(x.par_text, tokenized=True) for x in paragraphs]
        pars_spvecs = normalize(vstack(pars_spvecs))
        scores = pars_spvecs.dot(question_spvec.toarray().squeeze(axis=0))
        gold_rank = list((-scores).argsort()).index(gold_id) + 1
    # print(gold_rank)
    # print(' '.join(question.question))
    # print(list(zip([' '.join(x.par_text) for x in [question.paragraph]+question.distractors], scores)))
    return gold_rank


def get_rank_in_document_async(zipped):
    return get_rank_in_document(*zipped)


def main_for_document(use_ranker, num_workers):
    print("Loading data...")
    corpus = SquadRelevanceCorpus()
    # if args.corpus == "dev":
    #     questions = corpus.get_dev()
    # else:
    #     questions = corpus.get_train()
    questions = corpus.get_dev()

    question_preprocessor = SquadTextLengthPreprocessor(600)
    questions = [question_preprocessor.preprocess(x) for x in questions
                 if (question_preprocessor.preprocess(x) is not None)]

    title2max = {key: max(x.paragraph.par_id for x in group) for key, group in
                 itertools.groupby(sorted(questions, key=lambda x: x.paragraph.doc_title),
                                   key=lambda x: x.paragraph.doc_title)}

    if num_workers <= 1:
        if use_ranker:
            init()
        gold_ranks = [get_rank_in_document(q,
                                           corpus.dev_title_to_document[q.paragraph.doc_title].
                                           paragraphs[:title2max[q.paragraph.doc_title]+1]) for q in tqdm(questions)]

    else:
        # Setup worker pool
        workers = ProcessPool(
            num_workers,
            initializer=init if use_ranker else None,
            initargs=[]
        )

        data = [(q, corpus.dev_title_to_document[q.paragraph.doc_title].paragraphs[:title2max[q.paragraph.doc_title]+1])
                for q in questions]

        gold_ranks = []
        with tqdm(total=len(questions)) as pbar:
            for rank in tqdm(workers.imap_unordered(get_rank_in_document_async, data)):
                gold_ranks.append(rank)
                pbar.update()

    mean_rank = np.mean(gold_ranks)
    precision_at_1 = Counter(gold_ranks)[1]/len(gold_ranks)

    print(f"Mean Rank: {mean_rank}")
    print(f"Precision @ 1: {precision_at_1}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate tf-idf scoring on full squad.')
    parser.add_argument('--ranker', action='store_true', help='Whether to use bi-gram hashing or not')
    parser.add_argument('--per-doc', action='store_true')
    parser.add_argument('--num-workers', type=int, default=1)
    args = parser.parse_args()

    ranker = None
    if args.ranker:
        print("Loading ranker...")
        ranker = TfidfDocRanker()

    if args.per_doc:
        return main_for_document(ranker, args.num_workers)

    print("Loading data...")
    corpus = SquadRelevanceCorpus()
    # if args.corpus == "dev":
    #     questions = corpus.get_dev()
    # else:
    #     questions = corpus.get_train()
    questions = corpus.get_dev()

    question_preprocessor = SquadTextLengthPreprocessor(600)
    questions = [question_preprocessor.preprocess(x) for x in questions
                 if (question_preprocessor.preprocess(x) is not None)]

    if args.num_workers <= 1:
        if args.ranker:
            init()
        gold_ranks = [get_rank_in_distractors(q) for q in tqdm(questions)]
    else:
        # Setup worker pool
        workers = ProcessPool(
            args.num_workers,
            initializer=init if args.ranker else None,
            initargs=[]
        )

        gold_ranks = []
        with tqdm(total=len(questions)) as pbar:
            for rank in tqdm(workers.imap_unordered(get_rank_in_distractors, questions)):
                gold_ranks.append(rank)
                pbar.update()

    mean_rank = np.mean(gold_ranks)
    precision_at_1 = Counter(gold_ranks)[1]/len(gold_ranks)

    print(f"Mean Rank: {mean_rank}")
    print(f"Precision @ 1: {precision_at_1}")


if __name__ == '__main__':
    main()