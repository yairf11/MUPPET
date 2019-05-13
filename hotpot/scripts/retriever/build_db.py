"""A script to read in and store documents in a sqlite database.
This should take the wikipedia dump from hotpotqa and for each document save only a single paragraph,
which is the concatenation of the first original paragraphs that form at least 500 characters.
"""

import argparse
import bz2
import sqlite3
import json
import os
import logging
import re

import regex as re
from multiprocessing import Pool as ProcessPool
from tqdm import tqdm
from hotpot.tfidf_retriever import utils
from hotpot.tfidf_retriever.utils import serialize_object

logger = logging.getLogger()
logger.setLevel(logging.INFO)
fmt = logging.Formatter('%(asctime)s: [ %(message)s ]', '%m/%d/%Y %I:%M:%S %p')
console = logging.StreamHandler()
console.setFormatter(fmt)
logger.addHandler(console)

# ------------------------------------------------------------------------------
# Import helper
# ------------------------------------------------------------------------------

BLACKLIST = ['38754454']


# def preprocess(document, min_chars=500):
#     """ This function is deprecated.
#     Creates a single paragraph from the document by concatenating the first paragraphs until
#     it contains at least min_chars characters, and preprocesses a little bit:
#     - Omits the first paragraph, as it is just the title of the document.
#     - Removes hrefs (html tags)
#     The offsets remain structured in sentences to preserve sentence structure, but considered as a single paragraph.
#     The text is a single string.
#
#     :param document: A document in the bz2 format as created in hotpotqa
#     :return: the preprocessed document
#     """
#     if document['id'] in BLACKLIST:
#         return None
#     # Filter some disambiguation pages not caught by the WikiExtractor
#     # if '(disambiguation)' in document['title'].lower():
#     #     return None
#     # if '(disambiguation page)' in document['title'].lower():
#     #     return None
#     #
#     # Take out List/Index/Outline pages (mostly links)
#     # if re.match(r'(List of .+)', #|(Index of .+)|(Outline of .+)',
#     #             document['title']):
#     #     return None
#     final_paragraph_text = ''
#     final_paragraph_offsets = []
#     num_chars = 0
#     for par_id, par_sentences in enumerate(document['text'][1:]):
#         par_text = "".join(par_sentences) if par_id == 0 else ' ' + "".join(par_sentences)
#         num_chars_removed = 0 if par_id == 0 else -1  # TODO pretty ugly but works for now
#         for sent_id, sent_offs in enumerate(document['charoffset'][par_id + 1]):
#             fixed_offsets = []
#             for token_offsets in sent_offs:
#                 correct_offsets = [token_offsets[0] - num_chars_removed,
#                                    token_offsets[1] - num_chars_removed]
#                 token = par_text[correct_offsets[0]:correct_offsets[1]]
#                 if token.startswith('<a href=') or token == '</a>':
#                     par_text = par_text[:correct_offsets[0]] + par_text[correct_offsets[1]:]
#                     num_chars_removed += len(token)
#                     continue
#                 fixed_offsets.append([correct_offsets[0] + num_chars, correct_offsets[1] + num_chars])
#             final_paragraph_offsets.append(fixed_offsets)
#         num_chars += len(par_text)
#         final_paragraph_text += par_text
#         if num_chars >= min_chars:
#             break
#     # if num_chars < 20:  # TODO: necessary?
#     #     return None
#     return {'id': document['id'],
#             'title': document['title'],
#             'url': document['url'],
#             'text': final_paragraph_text,
#             'charoffset': final_paragraph_offsets}


def preprocess_sentences(document, min_chars=500):
    """Creates a single paragraph from the document by concatenating the first paragraphs until
    it contains at least min_chars characters, and preprocesses a little bit:
    - Omits the first paragraph, as it is just the title of the document.
    - Removes hrefs (html tags)
    The text is saved as a list of strings, each string a single sentence.
    No offsets are stored, as I have seen some examples where tokenizing according to the offsets was not so good.

    :param document: A document in the bz2 format as created in hotpotqa
    :return: the preprocessed document
    """
    if document['id'] in BLACKLIST:
        return None
    # Filter some disambiguation pages not caught by the WikiExtractor
    # if '(disambiguation)' in document['title'].lower():
    #     return None
    # if '(disambiguation page)' in document['title'].lower():
    #     return None
    #
    # Take out List/Index/Outline pages (mostly links)
    # if re.match(r'(List of .+)', #|(Index of .+)|(Outline of .+)',
    #             document['title']):
    #     return None
    final_sentences = []
    num_chars = 0

    html_pattern = re.compile('(</a>)|(<a.*?href.*?>)')

    def remove_tags(text):
        return re.sub(html_pattern, '', text)

    for i, par_sentences in enumerate(document['text'][1:]):
        for j, sentence in enumerate(par_sentences):
            clean_text = remove_tags(sentence)
            if len(clean_text) == 0:
                continue
            num_chars += len(clean_text)
            if not clean_text.startswith(' ') and (i > 0 or j > 0):
                clean_text = ' ' + clean_text
            final_sentences.append(clean_text)
        if num_chars >= min_chars:
            break
    if num_chars < 10:  # TODO: necessary?
        return None
    return {'id': document['id'],
            'title': document['title'],
            'url': document['url'],
            'sentences': final_sentences,
            'charoffset': document['charoffset']}


def clean_paragraphs(document):
    """ Just removes html tags from the paragraphs. """
    if document['id'] in BLACKLIST:
        return None
    html_pattern = re.compile('(</a>)|(<a.*?href.*?>)')

    def remove_tags(text):
        return re.sub(html_pattern, '', text)

    cleaned_pars = []
    for par_sentences in document['text'][1:]:
        clean_par = []
        for sentence in par_sentences:
            clean_text = remove_tags(sentence)
            if len(clean_text) == 0:
                continue
            clean_par.append(clean_text)
        if len(clean_par) == 0:
            continue
        cleaned_pars.append(clean_par)
    if len(cleaned_pars) == 0:
        return None

    return {'id': document['id'],
            'title': document['title'],
            'url': document['url'],
            'paragraphs': cleaned_pars,
            'charoffset': document['charoffset']}


# ------------------------------------------------------------------------------
# Store corpus.
# ------------------------------------------------------------------------------


def iter_files(path):
    """Walk through all files located under a root path."""
    if os.path.isfile(path):
        yield path
    elif os.path.isdir(path):
        for dirpath, _, filenames in os.walk(path):
            for f in filenames:
                yield os.path.join(dirpath, f)
    else:
        raise RuntimeError('Path %s is invalid' % path)


def get_contents(filename):
    """Parse the contents of a file. Each line is a JSON encoded document."""
    documents = []
    with bz2.open(filename, mode='rt') as f:
        for line in f:
            # Parse document
            doc = json.loads(line)
            # Maybe preprocess the document with custom function
            doc = preprocess_sentences(doc)
            # Skip if it is empty or None
            if not doc:
                continue
            # Add the document
            documents.append((doc['title'],
                              serialize_object(doc['sentences'])))
    return documents


def get_full_document(filename):
    """Parse the contents of a file. Each line is a JSON encoded document."""
    documents = []
    with bz2.open(filename, mode='rt') as f:
        for line in f:
            # Parse document
            doc = json.loads(line)
            # Maybe preprocess the document with custom function
            doc = clean_paragraphs(doc)
            # Skip if it is empty or None
            if not doc:
                continue
            # Add the document
            documents.append((doc['title'],
                              serialize_object(doc['paragraphs'])))
    return documents


def store_contents(data_path, save_path, num_workers=None, hotpot=True):
    """Preprocess and store a corpus of documents in sqlite.

    Args:
        data_path: Root path to directory (or directory of directories) of files
          containing bz2 encoded documents (must have `id`, `url`, `title`, `text` and `charoffset` fields).
          the specific format is described here:
            https://hotpotqa.github.io/wiki-readme.html
        save_path: Path to output sqlite db.
        preprocess: Path to file defining a custom `preprocess` function. Takes
          in and outputs a structured doc.
        num_workers: Number of parallel processes to use when reading docs.
        hotpot: whether this db is meant for hotpot, or for global use - in that case it's a full document db
    """
    if os.path.isfile(save_path):
        raise RuntimeError('%s already exists! Not overwriting.' % save_path)

    logger.info('Reading into database...')
    conn = sqlite3.connect(save_path)
    c = conn.cursor()
    if hotpot:
        c.execute("CREATE TABLE documents (title PRIMARY KEY, sentences);")
    else:
        c.execute("CREATE TABLE documents (title PRIMARY KEY, paragraphs);")

    workers = ProcessPool(num_workers)
    files = [f for f in iter_files(data_path)]
    count = 0
    content_getter = get_contents if hotpot else get_full_document
    with tqdm(total=len(files)) as pbar:
        for documents in tqdm(workers.imap_unordered(content_getter, files)):
            count += len(documents)
            c.executemany("INSERT INTO documents VALUES (?,?)", documents)
            pbar.update()
    logger.info('Read %d docs.' % count)
    logger.info('Committing...')
    conn.commit()
    conn.close()


# ------------------------------------------------------------------------------
# Main.
# ------------------------------------------------------------------------------


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path', type=str, help='/path/to/data')
    parser.add_argument('save_path', type=str, help='/path/to/saved/db.db')
    parser.add_argument('--num-workers', type=int, default=None,
                        help='Number of CPU processes (for tokenizing, etc)')
    parser.add_argument('--full', action='store_true', help="Whether to save hotpot-style docs or full docs")
    args = parser.parse_args()

    store_contents(
        args.data_path, args.save_path, args.num_workers, not args.full
    )
