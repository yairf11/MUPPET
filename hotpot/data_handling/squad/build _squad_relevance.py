import argparse
import urllib
from multiprocessing.util import Finalize
from os import mkdir

from os.path import exists, join
from typing import List, Tuple
from multiprocessing import Pool as ProcessPool
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, strip_accents_unicode
from sklearn.metrics import pairwise_distances
from collections import Counter
import random

from tqdm import tqdm

import hotpot
from hotpot import utils
from hotpot import config
from hotpot.data_handling.squad.squad_data import SquadDocument, SquadParagraph, SquadQuestionWithDistractors, \
    SquadRelevanceCorpus
from hotpot.tfidf_retriever.doc_db import DocDB
from hotpot.tfidf_retriever.tfidf_doc_ranker import TfidfDocRanker
from hotpot.tfidf_retriever.utils import STOPWORDS
from hotpot.tokenizers.corenlp_tokenizer import CoreNLPTokenizer
from hotpot.tokenizers.tokenizer import ngrams_from_tokens

PROCESS_TOK = None
PROCESS_DB = None
PROCESS_FULL_DOC_DB = None
PROCESS_RANKER = None
DOC_TITLES = None


def init(db_path, full_doc_db_path, ranker_path):
    global PROCESS_TOK, PROCESS_DB, PROCESS_RANKER, PROCESS_FULL_DOC_DB, DOC_TITLES
    PROCESS_TOK = CoreNLPTokenizer()
    Finalize(PROCESS_TOK, PROCESS_TOK.shutdown, exitpriority=100)
    PROCESS_DB = DocDB(db_path)
    Finalize(PROCESS_DB, PROCESS_DB.close, exitpriority=100)
    PROCESS_FULL_DOC_DB = DocDB(full_doc_db_path, full_docs=True)
    Finalize(PROCESS_FULL_DOC_DB, PROCESS_FULL_DOC_DB.close, exitpriority=100)
    PROCESS_RANKER = TfidfDocRanker(ranker_path)
    DOC_TITLES = PROCESS_FULL_DOC_DB.get_doc_titles()


def get_all_doc_titles():
    global DOC_TITLES
    return DOC_TITLES


def fetch_sentences(doc_title):
    global PROCESS_DB
    return PROCESS_DB.get_doc_sentences(doc_title)


def fetch_paragraphs(doc_title):
    global PROCESS_FULL_DOC_DB
    return PROCESS_FULL_DOC_DB.get_doc_paragraphs(doc_title)


def tokenize(text):
    global PROCESS_TOK
    return PROCESS_TOK.tokenize(text)


def fetch_batch_tfidf(queries, k, tokenized=False):
    global PROCESS_RANKER
    return PROCESS_RANKER.batch_closest_docs(queries, k, num_workers=1, tokenized=tokenized)


def get_random_pars(forbidden_titles: List[str], num_pars,
                    min_num_chars=500) -> List[SquadParagraph]:
    """ Returns a list of paragraphs chosen at random """
    all_titles = get_all_doc_titles()
    chosen_pars = []
    while len(chosen_pars) < num_pars:
        title = random.choice(all_titles)
        if title in forbidden_titles:
            continue
        paragraphs = fetch_paragraphs(title)
        paragraphs = [(idx, x) for idx, x in enumerate(paragraphs) if len(''.join(x)) > min_num_chars]
        if len(paragraphs) == 0:
            continue
        par_id, par_text = random.choice(paragraphs)
        par_tokens = tokenize(' '.join(par_text)).words()
        chosen_pars.append(SquadParagraph(doc_title=title, par_id=par_id, par_text=par_tokens))
    return chosen_pars


def add_random_distractors(questions: List[SquadQuestionWithDistractors], num_pars):
    for q in questions:
        titles = [x.doc_title for x in q.distractors] + [q.paragraph.doc_title]
        q.add_distractors(get_random_pars(forbidden_titles=titles, num_pars=num_pars))


def get_top_in_document(queries: List[List[str]], document: SquadDocument, forbidden_pairs: List[int],
                        num_pars) -> List[List[SquadParagraph]]:
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

    vectorizer = TfidfVectorizer(preprocessor=tf_idf_prepro, tokenizer=tf_idf_tok, stop_words=STOPWORDS)
    queries_features = vectorizer.fit_transform(queries)
    document_pars_features = vectorizer.transform([x.par_text for x in document.paragraphs])
    scores = pairwise_distances(queries_features, document_pars_features, "cosine")
    scores[np.arange(len(queries)), forbidden_pairs] = scores.max() + 1
    closest_pars = np.array([document.paragraphs])[[0], np.argpartition(scores, kth=num_pars, axis=1)[:, :num_pars]]
    return closest_pars


def add_distractors_from_document(questions: List[SquadQuestionWithDistractors], document: SquadDocument, num_pars):
    true_pars_ids = [q.paragraph.par_id for q in questions]
    top_for_questions = get_top_in_document([q.question for q in questions], document, forbidden_pairs=true_pars_ids,
                                            num_pars=num_pars)
    top_for_contexts = get_top_in_document([q.paragraph.par_text for q in questions], document,
                                           forbidden_pairs=true_pars_ids, num_pars=num_pars)
    for q, top_qs, top_contexts in zip(questions, top_for_questions, top_for_contexts):
        q.add_distractors(list(top_qs) + list(top_contexts))


def get_top_in_wikipedia(queries: List[List[str]], forbidden_titles: List[List[str]],
                         num_pars) -> List[List[SquadParagraph]]:
    global PROCESS_RANKER, PROCESS_DB
    num_pars_to_ret = num_pars + max(
        len(x) for x in forbidden_titles)  # so that there are enough paragraphs no matter what
    closest_docs = fetch_batch_tfidf(queries, num_pars_to_ret, tokenized=True)
    final_titles = [[title for title in titles if title not in forbidden_titles[idx]][:num_pars]
                    for idx, (titles, scores) in enumerate(closest_docs)]
    # note that paragraphs from the ranker are given an index of -1 so that there wont be confusion with other documents
    final_pars = [[SquadParagraph(title, par_id=-1, par_text=tokenize(' '.join(fetch_sentences(title))).words())
                   for title in query_titles]
                  for query_titles in final_titles]
    return final_pars


def add_distractors_from_wikipedia(questions: List[SquadQuestionWithDistractors], document: SquadDocument, num_pars):
    top_for_contexts = get_top_in_wikipedia([p.par_text for p in document.paragraphs],
                                            forbidden_titles=[[p.doc_title] for p in document.paragraphs],
                                            num_pars=num_pars)
    par_id_to_closest = {p.par_id: top_for_contexts[idx] for idx, p in enumerate(document.paragraphs)}
    for q in questions:
        q.add_distractors(par_id_to_closest[q.paragraph.par_id])

    existing_titles = [[x.doc_title for x in q.distractors] + [q.paragraph.doc_title] for q in questions]
    top_for_questions = get_top_in_wikipedia([q.question for q in questions], forbidden_titles=existing_titles,
                                             num_pars=num_pars)
    for q, top_qs in zip(questions, top_for_questions):
        q.add_distractors(top_qs)


def clean_title(title):
    """ Squad titles use URL escape formatting, this method undoes it to get the wiki-title"""
    return urllib.parse.unquote(title).replace("_", " ")


def make_squad_document_and_questions(squad_doc_dict, gold_source='squad') -> \
        Tuple[SquadDocument, List[SquadQuestionWithDistractors]]:
    """
    Create the squad relevance gold questions and paragraphs.
    There are two options for constructing this dataset, as the wikipedia pages are different than those used in squad:
    1. gold_source == 'squad':
        The golds paragraphs are completely from squad, no use of the db for the relevant documents,
         only for distractors.
    2. gold_source == 'db':
        Here we use the exact same documents as in our db. We filter questions that do not have answers
         in their most similar paragraph, and otherwise set that paragraph to be the gold one.
        It is possible to filter passages also based on the degree of similarity, but we leave it for now.
    """
    title = clean_title(squad_doc_dict['title'])
    if gold_source == 'squad':
        tokenized_squad_pars = [tokenize(par['context']).words() for par in squad_doc_dict['paragraphs']]
        paragraphs = [SquadParagraph(title, idx, tok_text)
                      for idx, tok_text in enumerate(tokenized_squad_pars)]
        document = SquadDocument(title, paragraphs)
        questions = [SquadQuestionWithDistractors(q['id'], tokenize(q['question']).words(),
                                                  set([ans['text'] for ans in q['answers']]), paragraphs[idx], [])
                     for idx, squad_par in enumerate(squad_doc_dict['paragraphs']) for q in squad_par['qas']]
        # filter out questions that have only stop words
        questions = [q for q in questions if
                     len(ngrams_from_tokens(q.question, n=2, uncased=True,
                                            filter_fn=hotpot.tfidf_retriever.utils.filter_ngram)) > 0]
        return document, questions
    elif gold_source == 'db':
        tokenized_squad_pars = [tokenize(par['context']).words() for par in squad_doc_dict['paragraphs']]
        raw_db_doc_pars = [' '.join(x) for x in fetch_paragraphs(title)]
        tokenized_db_doc_pars = [tokenize(' '.join(sentences)).words() for sentences in
                                 fetch_paragraphs(title)]

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

        vectorizer = TfidfVectorizer(preprocessor=tf_idf_prepro, tokenizer=tf_idf_tok, stop_words=STOPWORDS)
        squad_par_features = vectorizer.fit_transform(tokenized_squad_pars)
        db_par_features = vectorizer.transform(tokenized_db_doc_pars)

        scores = pairwise_distances(squad_par_features, db_par_features, "cosine")
        most_similar_idx = scores.argmin(axis=1)
        # most_similar = np.array(tokenized_db_doc_pars)[most_similar_idx]
        paragraphs = [SquadParagraph(title, idx, tokenized_text)
                      for idx, tokenized_text in enumerate(tokenized_db_doc_pars)]
        document = SquadDocument(title, paragraphs)

        def get_answer_set_from_question(question):
            return set([answer['text'] for answer in question['answers']])

        def get_qas_for_par(par):
            return [q for q in par['qas']]

        def at_least_one_ans(par, answer_set):
            return any(x in par for x in answer_set)

        def get_filtered_par_questions(squad_par, par_text):
            return [q for q in get_qas_for_par(squad_par) if
                    at_least_one_ans(par_text, get_answer_set_from_question(q))]

        questions = [SquadQuestionWithDistractors(q['id'], tokenize(q['question']).words(),
                                                  get_answer_set_from_question(q), paragraphs[sim_idx], [])
                     for idx, sim_idx in enumerate(most_similar_idx)
                     for q in get_filtered_par_questions(squad_doc_dict['paragraphs'][idx], raw_db_doc_pars[sim_idx])]

        return document, questions
    else:
        raise ValueError()


def extend_documents(documents: List[SquadDocument], questions: List[SquadQuestionWithDistractors]) \
        -> List[SquadDocument]:
    title_to_doc = {d.title: d for d in documents}
    for q in questions:
        for par in q.distractors:
            if par.doc_title in title_to_doc:
                if par.par_id in title_to_doc[par.doc_title].id_to_par:
                    continue
                title_to_doc[par.doc_title].add_par(par)
            else:
                title_to_doc[par.doc_title] = SquadDocument(title=par.doc_title,
                                                            paragraphs=[SquadParagraph(par.doc_title,
                                                                                       par.par_id,
                                                                                       par.par_text,
                                                                                       pickle_text=True)])
    return list(title_to_doc.values())


def build_single(squad_doc_dict, gold_source='squad') -> Tuple[SquadDocument, List[SquadQuestionWithDistractors]]:
    document, questions = make_squad_document_and_questions(squad_doc_dict, gold_source)
    add_distractors_from_document(questions, document, 3)
    add_distractors_from_wikipedia(questions, document, 2)
    add_random_distractors(questions, 2)

    return document, questions


def build_squad_data_async(data_path, db_path, full_doc_db_path, ranker_path, num_workers) \
        -> Tuple[List[SquadDocument], List[SquadQuestionWithDistractors]]:
    data = utils.load_json_dataset(data_path)['data']
    documents = []
    questions = []

    # Setup worker pool
    workers = ProcessPool(
        num_workers,
        initializer=init,
        initargs=[db_path, full_doc_db_path, ranker_path]
    )

    with tqdm(total=len(data)) as pbar:
        for doc, doc_qs in tqdm(workers.imap_unordered(build_single, data)):
            questions.extend(doc_qs)
            documents.append(doc)
            pbar.update()

    documents = extend_documents(documents, questions)
    for q in questions:
        q.paragraph = q.paragraph.get_paragraph_without_text_pickling()
        q.distractors = [p.get_paragraph_without_text_pickling() for p in q.distractors]
    return documents, questions


def build_squad_data_sync(data_path, db_path, full_doc_db_path, ranker_path):
    data = utils.load_json_dataset(data_path)['data']
    documents = []
    questions = []

    init(db_path, full_doc_db_path, ranker_path)

    for doc_data in tqdm(data):
        doc, doc_qs = build_single(doc_data)
        documents.append(doc)
        questions.extend(doc_qs)

    documents = extend_documents(documents, questions)
    for q in questions:
        q.paragraph = q.paragraph.get_paragraph_without_text_pickling()
        q.distractors = [p.get_paragraph_without_text_pickling() for p in q.distractors]
    return documents, questions


def main():
    parser = argparse.ArgumentParser("Create a Squad dataset for relevance prediction")
    parser.add_argument("--train-file", default=config.SQUAD_TRAIN_FILE)
    parser.add_argument("--dev-file", default=config.SQUAD_DEV_FILE)
    parser.add_argument("--doc-db", default=config.DOC_DB)
    parser.add_argument("--full-doc-db", default=config.FULL_DOC_DB)
    parser.add_argument("--ranker", default=config.TFIDF_FILE)
    parser.add_argument('--num-workers', type=int, default=1, help='Number of CPU processes')

    if not exists(join(config.CORPUS_DIR, 'squad')):
        mkdir(join(config.CORPUS_DIR, 'squad'))

    args = parser.parse_args()

    # target_dir = config.CORPUS_DIR
    # if exists(target_dir) and len(listdir(target_dir)) > 0:
    #     raise ValueError("Files already exist in " + target_dir)
    if args.num_workers > 1:
        print(f"Multiprocessing with {args.num_workers} threads...")
        print("Parsing train...")
        train_docs, train_qs = build_squad_data_async(args.train_file, args.doc_db, args.full_doc_db, args.ranker,
                                                      args.num_workers)
        print("Parsing dev...")
        dev_docs, dev_qs = build_squad_data_async(args.dev_file, args.doc_db, args.full_doc_db, args.ranker,
                                                  args.num_workers)
    else:
        print("Parsing train...")
        train_docs, train_qs = build_squad_data_sync(args.train_file, args.doc_db, args.full_doc_db, args.ranker)

        print("Parsing dev...")
        dev_docs, dev_qs = build_squad_data_sync(args.dev_file, args.doc_db, args.full_doc_db, args.ranker)

    print("Saving...")
    SquadRelevanceCorpus.make_corpus(train_docs, train_qs, dev_docs, dev_qs)
    print("Done")


if __name__ == "__main__":
    main()
