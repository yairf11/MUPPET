""" Build a Hotpot open dataset by retrieving the top-k documents for each question """
import argparse
import json
from multiprocessing.util import Finalize
from typing import List, Dict, Tuple
from multiprocessing import Pool as ProcessPool

from tqdm import tqdm

from hotpot import config
from hotpot.tfidf_retriever.doc_db import DocDB
from hotpot.tfidf_retriever.tfidf_doc_ranker import TfidfDocRanker

PROCESS_DB = None
PROCESS_RANKER = None
TOP_K = None
GET_TEXT = None


def init(top_k, get_text):
    global PROCESS_DB, PROCESS_RANKER, TOP_K, GET_TEXT
    PROCESS_DB = DocDB()
    Finalize(PROCESS_DB, PROCESS_DB.close, exitpriority=100)
    PROCESS_RANKER = TfidfDocRanker()
    TOP_K = top_k
    GET_TEXT = get_text


def fetch_sentences(doc_title):
    global PROCESS_DB
    return PROCESS_DB.get_doc_sentences(doc_title)


def fetch_batch_tfidf(queries, k, tokenized=False):
    global PROCESS_RANKER
    return PROCESS_RANKER.batch_closest_docs(queries, k, num_workers=1, tokenized=tokenized)


def get_top_k_titles_sync(questions: List[str]) -> List[List[Tuple[str, str]]]:
    closest_docs = fetch_batch_tfidf(questions, TOP_K)
    q_docs = [[(title, fetch_sentences(title) if GET_TEXT else None) for title in titles]
              for titles, scores in closest_docs]
    return q_docs


def get_top_k_async(questions: List[str], k: int, num_workers: int, get_text: bool) -> List[List[Tuple[str, str]]]:
    if num_workers == 1:
        init(k, get_text)
        print("One thread, hang on...")
        return get_top_k_titles_sync(questions)
    chunked_questions = [questions[i:i + 32] for i in range(0, len(questions), 32)]
    # Setup worker pool
    workers = ProcessPool(
        num_workers,
        initializer=init,
        initargs=[k, get_text]
    )

    results = []

    with tqdm(total=len(chunked_questions)) as pbar:
        for docs in tqdm(workers.imap(get_top_k_titles_sync, chunked_questions)):
            results.extend(docs)
            pbar.update()

    return results


def build_hotpot_top_k_dataset(questions_file, docs_file, k: int, num_workers: int, test=False):
    print("loading Hotpot data...")
    if not test:
        with open(config.HOTPOT_DEV_DISTRACTOR_FILE, 'r') as f:
            dev_data = json.load(f)

        questions = [{'qid': q['_id'], 'question': q['question'], 'answers': [q['answer']], 'type': q['type'],
                      'supporting_facts': q['supporting_facts']}
                     for q in dev_data]
    else:
        print("Test Set!")
        with open(config.HOTPOT_TEST_FULL_WIKI_FILE, 'r') as f:
            test_data = json.load(f)

        questions = [{'qid': q['_id'], 'question': q['question']}
                     for q in test_data]
    build_top_k_dataset(questions, questions_file, docs_file, k, num_workers)


def build_top_k_dataset(questions, questions_file, docs_file, k: int, num_workers: int):
    top_k_docs = get_top_k_async([q['question'] for q in questions], k=k, num_workers=num_workers,
                                 get_text=docs_file is not None)
    question_top_titles = [[x for x, _ in q_docs] for q_docs in top_k_docs]
    for q, titles in zip(questions, question_top_titles):
        q['top_titles'] = titles

    with open(questions_file, 'w') as f:
        json.dump(questions, f)

    if docs_file is not None:
        title2text = {title: text for q_docs in top_k_docs for title, text in q_docs}
        title2paragraphs = {title: [' '.join(sentences)]
                            for title, sentences in title2text.items()}

        with open(docs_file, 'w') as f:
            json.dump(title2paragraphs, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Build an open-qa dataset based on top-k documents for each question')
    parser.add_argument('questions_file', help='json file to save the questions with the top titles')
    parser.add_argument('top_k', type=int)
    parser.add_argument('--docs_file', default=None, help="json filename to dump the top-k dataset")
    parser.add_argument('--num-workers', type=int, default=16)
    parser.add_argument('--test', action='store_true')
    args = parser.parse_args()
    build_hotpot_top_k_dataset(args.questions_file, args.docs_file, args.top_k, args.num_workers, args.test)
