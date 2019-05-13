import argparse
from collections import Counter
from multiprocessing.util import Finalize
from multiprocessing import Pool as ProcessPool

from tqdm import tqdm

from hotpot.config import DOC_DB
from hotpot.tfidf_retriever.doc_db import DocDB
from hotpot.tokenizers import CoreNLPTokenizer


PROCESS_TOK = None
PROCESS_DB = None


def init(db_path):
    global PROCESS_TOK, PROCESS_DB
    PROCESS_TOK = CoreNLPTokenizer()
    Finalize(PROCESS_TOK, PROCESS_TOK.shutdown, exitpriority=100)
    PROCESS_DB = DocDB(db_path)
    Finalize(PROCESS_DB, PROCESS_DB.close, exitpriority=100)


def fetch_sentences(doc_title):
    global PROCESS_DB
    return PROCESS_DB.get_doc_sentences(doc_title)


def tokenize(text):
    global PROCESS_TOK
    return PROCESS_TOK.tokenize(text)


def get_word_count(doc_title):
    sentences = fetch_sentences(doc_title)
    words = [w for s in sentences for w in tokenize(s).words()]
    return Counter(words)


def get_all_word_count(num_workers, db_path, out_path, titles_only=False):
    db = DocDB(db_path)
    all_titles = db.get_doc_titles()

    # Setup worker pool
    workers = ProcessPool(
        num_workers,
        initializer=init,
        initargs=[db_path]
    )

    word_counts = Counter()

    if not titles_only:
        with tqdm(total=len(all_titles), desc='word count') as pbar:
            for w_count in tqdm(workers.imap_unordered(get_word_count, all_titles)):
                word_counts.update(w_count)
                pbar.update()
    else:
        with tqdm(total=len(all_titles), desc='title word count') as pbar:
            for tok_title in tqdm(workers.imap_unordered(tokenize, all_titles)):
                word_counts.update(tok_title.words())
                pbar.update()

    with open(out_path, 'w') as f:
        for k, v in tqdm(word_counts.most_common(), desc='writing counts'):
            f.write(f"{k}\t{v}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create and save word counts from db')
    parser.add_argument("out_path", type=str, help='path to save word counts')
    parser.add_argument("--db_path", type=str, default=DOC_DB)
    parser.add_argument("--num_workers", type=int, default=32)
    parser.add_argument("--titles", action='store_true', help="save only title words")
    args = parser.parse_args()

    get_all_word_count(args.num_workers, args.db_path, args.out_path, args.titles)
