""" Build a SQuAD open dataset by retrieving the top-k documents for each question """
import argparse
import json
from multiprocessing.util import Finalize
from typing import List, Dict, Tuple
from multiprocessing import Pool as ProcessPool

from tqdm import tqdm

from hotpot import config
from hotpot.config import DRQA_DOC_DB, DRQA_RANKER
from hotpot.tfidf_retriever.doc_db import OldDocDB
from hotpot.tfidf_retriever.tfidf_doc_ranker import TfidfDocRanker

PROCESS_DB = None
PROCESS_RANKER = None
TOP_K = None


def init(top_k):
    global PROCESS_DB, PROCESS_RANKER, TOP_K
    PROCESS_DB = OldDocbDB(DRQA_DOC_DB)
    Finalize(PROCESS_DB, PROCESS_DB.close, exitpriority=100)
    PROCESS_RANKER = TfidfDocRanker(DRQA_RANKER, strict=False, tokenizer='simple')
    TOP_K = top_k


def fetch_text(doc_title):
    global PROCESS_DB
    return PROCESS_DB.get_doc_text(doc_title)


def fetch_batch_tfidf(queries, k, tokenized=False):
    global PROCESS_RANKER
    return PROCESS_RANKER.batch_closest_docs(queries, k, num_workers=1, tokenized=tokenized)


def get_top_k_titles_sync(questions: List[str]) -> List[List[Tuple[str, str]]]:
    closest_docs = fetch_batch_tfidf(questions, TOP_K)
    q_docs = [[(title, fetch_text(title)) for title in titles] for titles, scores in closest_docs]
    return q_docs


def get_top_k_async(questions: List[str], k: int, num_workers: int) -> List[List[Tuple[str, str]]]:
    chunked_questions = [questions[i:i + 32] for i in range(0, len(questions), 32)]
    # Setup worker pool
    workers = ProcessPool(
        num_workers,
        initializer=init,
        initargs=[k]
    )

    results = []

    with tqdm(total=len(chunked_questions)) as pbar:
        for docs in tqdm(workers.imap(get_top_k_titles_sync, chunked_questions)):
            results.extend(docs)
            pbar.update()

    return results


def build_squad_open_top_k_dataset(questions_file, docs_file, k: int, num_workers: int):
    print("loading SQuAD data...")
    with open(config.SQUAD_DEV_FILE, 'r') as f:
        dev_data = json.load(f)['data']

    questions = [{'qid': q['id'], 'question': q['question'], 'answers': [a['text'] for a in q['answers']]}
                 for doc in dev_data for par in doc['paragraphs'] for q in par['qas']]
    build_top_k_dataset(questions, questions_file, docs_file, k, num_workers)


def build_trec_wikimovies_webq_open_top_k_dataset(dataset: str, questions_file, docs_file, k: int, num_workers: int):
    datasets_dict = {'curated_trec': config.CURATED_TREC_TEST_FILE,
                     'wiki_movies': config.WIKI_MOVIES_TEST_FILE,
                     'web_questions': config.WEB_QUESTIONS_TEST_FILE}
    print(f"loading {dataset} data...")
    questions = []
    with open(datasets_dict[dataset], 'r') as f:
        for idx, line in enumerate(f):
            q_dict = json.loads(line)
            questions.append({'qid': idx,
                              'question': q_dict['question'],
                              'answers': q_dict['answer']})

    build_top_k_dataset(questions, questions_file, docs_file, k, num_workers)


def build_top_k_dataset(questions, questions_file, docs_file, k: int, num_workers: int):
    top_k_docs = get_top_k_async([q['question'] for q in questions], k=k, num_workers=num_workers)
    question_top_titles = [[x for x, _ in q_docs] for q_docs in top_k_docs]
    for q, titles in zip(questions, question_top_titles):
        q['top_titles'] = titles
    title2text = {title: text for q_docs in top_k_docs for title, text in q_docs}
    title2paragraphs = {title: [para.strip() for para in text.split("\n") if len(para.strip()) > 0]
                        for title, text in title2text.items()}

    with open(questions_file, 'w') as f:
        json.dump(questions, f)
    with open(docs_file, 'w') as f:
        json.dump(title2paragraphs, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Build an open-qa dataset based on top-k documents for each question')
    parser.add_argument('dataset', choices=["squad", "curated_trec", "wiki_movies",
                                            "web_questions"])
    parser.add_argument('questions_file', help='json file to save the questions with the top titles')
    parser.add_argument('docs_file', help="json filename to dump the top-k dataset")
    parser.add_argument('top_k', type=int)
    parser.add_argument('--num-workers', type=int, default=16)
    args = parser.parse_args()
    if args.dataset == 'squad':
        build_squad_open_top_k_dataset(args.questions_file, args.docs_file, args.top_k, args.num_workers)
    else:
        build_trec_wikimovies_webq_open_top_k_dataset(args.dataset, args.questions_file, args.docs_file,
                                                      args.top_k, args.num_workers)
