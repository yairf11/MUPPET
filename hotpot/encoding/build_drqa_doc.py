import argparse
import json
import sqlite3
from multiprocessing.util import Finalize
from os.path import join
from multiprocessing import Pool as ProcessPool
from typing import Tuple, Dict, List
import itertools

import numpy as np
from tqdm import tqdm

from hotpot.config import DRQA_DOC_DB
from hotpot.data_handling.relevance_training_data import BinaryQuestionAndParagraphs
from hotpot.data_handling.dataset import QuestionAndParagraphsSpec
from hotpot.data_handling.squad.squad_data import SquadRelevanceCorpus
from hotpot.encoding import encode_squad
from hotpot.encoding.knn import simple_numpy_knn
from hotpot.encoding.paragraph_encoder import SentenceEncoderSingleContext
from hotpot.tokenizers import CoreNLPTokenizer
from hotpot.utils import ResourceLoader

PROCESS_TOK = None


def init():
    global PROCESS_TOK
    PROCESS_TOK = CoreNLPTokenizer()
    Finalize(PROCESS_TOK, PROCESS_TOK.shutdown, exitpriority=100)


def tokenize(text):
    global PROCESS_TOK
    return PROCESS_TOK.tokenize(text)


def get_document_paragraphs(doc: Tuple[str, str]) -> Tuple[Dict[str, List[str]], Dict[str, List[List[str]]]]:
    title = doc[0]
    text = doc[1]
    paragraphs = []
    for para in text.split("\n"):
        para = para.strip()
        if len(para) > 0:
            paragraphs.append(para)
    tokenized_paras = [tokenize(x).words() for x in paragraphs]
    return {title: paragraphs}, {title: tokenized_paras}


def squad_build_drqa_doc_encodings(out_dir, encoder_model, num_workers, all_squad=False):
    print("loading data...")
    corpus = SquadRelevanceCorpus()
    questions = corpus.get_dev()
    if all_squad:
        questions.extend(corpus.get_train())
    # docs = corpus.dev_title_to_document.values() if dev else corpus.train_title_to_document.values()
    relevant_titles = list(set([q.paragraph.doc_title for q in questions]))

    conn = sqlite3.connect(DRQA_DOC_DB)
    c = conn.cursor()
    titles = list(set([q.paragraph.doc_title for q in questions]))
    for i, t in enumerate(titles):
        # Had to manually resolve this (due to changes in Wikipedia?)
        if t == "Sky (United Kingdom)":
            titles[i] = "Sky UK"

    title_to_doc_id = {t1: t2 for t1, t2 in zip(titles, relevant_titles)}

    c.execute("CREATE TEMPORARY TABLE squad_docs(id)")
    c.executemany("INSERT INTO squad_docs VALUES (?)", [(x,) for x in titles])

    c.execute("SELECT id, text FROM documents WHERE id IN squad_docs")

    out = c.fetchall()
    conn.close()

    out = [(title_to_doc_id[title], text) for title, text in out]

    spec = QuestionAndParagraphsSpec(batch_size=None, max_num_contexts=1,
                                     max_num_question_words=None, max_num_context_words=None)
    voc = corpus.get_vocab()
    encoder = SentenceEncoderSingleContext(model_dir_path=encoder_model, vocabulary=voc, spec=spec,
                                           loader=ResourceLoader())

    # Setup worker pool
    workers = ProcessPool(
        num_workers,
        initializer=init,
        initargs=[]
    )

    documents = {}
    tokenized_documents = {}

    print("Tokenizing...")
    with tqdm(total=len(out)) as pbar:
        for doc, tok_doc in tqdm(workers.imap_unordered(get_document_paragraphs, out)):
            documents.update(doc)
            tokenized_documents.update(tok_doc)
            pbar.update()

    encodings = {}
    print("Encoding...")
    for title, paragraphs in tqdm(tokenized_documents.items()):
        dummy_question = "Hello Hello".split()
        model_paragraphs = [BinaryQuestionAndParagraphs(question=dummy_question,
                                                        paragraphs=[x], label=1,
                                                        num_distractors=0, question_id='dummy') for x in paragraphs]
        encodings.update({f"{title}_{i}": rep for i, rep in enumerate(encoder.encode_paragraphs(model_paragraphs))})

    with open(join(out_dir, 'docs.json'), 'w') as f:
        json.dump(documents, f)
    np.savez_compressed(join(out_dir, 'encodings.npz'), **encodings)


def build_doc_eval_file(out_file, encodings_dir, encoder_model, k, per_doc=True):
    print("loading data...")
    corpus = SquadRelevanceCorpus()
    questions = corpus.get_dev()
    spec = QuestionAndParagraphsSpec(batch_size=None, max_num_contexts=1,
                                     max_num_question_words=None, max_num_context_words=None)
    voc = corpus.get_vocab()
    encoder = SentenceEncoderSingleContext(model_dir_path=encoder_model, vocabulary=voc, spec=spec,
                                           loader=corpus.get_resource_loader())

    par_encs = np.load(join(encodings_dir, 'encodings.npz'))
    with open(join(encodings_dir, 'docs.json'), 'r') as f:
        documents = json.load(f)

    questions_eval_format = []
    questions = sorted(questions, key=lambda x: x.paragraph.doc_title)
    if per_doc:
        title2par_encs = {}
        for p_name, rep in par_encs.items():
            title = '_'.join(p_name.split('_')[:-1])
            if title in title2par_encs:
                title2par_encs[title].update({p_name: rep})
            else:
                title2par_encs[title] = {p_name: rep}
        for title, doc_qs in tqdm(itertools.groupby(questions, key=lambda x: x.paragraph.doc_title)):
            doc_qs = list(doc_qs)
            q_encodings = encode_squad.encode_questions(encoder, doc_qs)
            par2ids = {}
            reps = []
            total_sentences = 0
            for p_name, rep in title2par_encs[title].items():
                par2ids[p_name] = list(range(total_sentences, total_sentences + len(rep)))
                reps.append(rep)
                total_sentences += len(rep)
            id2par = {i: p for p, ids in par2ids.items() for i in ids}
            reps = np.concatenate(reps, axis=0)
            top_k = simple_numpy_knn(q_encodings, reps, k*2)
            for idx, question in enumerate(doc_qs):
                seen = set()
                p_names = [id2par[x] for x in top_k[idx] if not (id2par[x] in seen or seen.add(id2par[x]))][:k]
                questions_eval_format.append(
                    {'qid': question.question_id, 'question': ' '.join(question.question),
                     'answers': list(question.answers),
                     'paragraphs': [documents['_'.join(p_name.split('_')[:-1])][int(p_name.split('_')[-1])]
                                    for p_name in p_names]})
    else:
        print("encoding questions")
        q_encodings = encode_squad.encode_questions(encoder, questions)
        par2ids = {}
        reps = []
        total_sentences = 0
        for p_name, rep in par_encs.items():
            par2ids[p_name] = list(range(total_sentences, total_sentences + len(rep)))
            reps.append(rep)
            total_sentences += len(rep)
        id2par = {i: p for p, ids in par2ids.items() for i in ids}
        reps = np.concatenate(reps, axis=0)
        print("scoring")
        top_k = simple_numpy_knn(q_encodings, reps, k*2)
        for idx, question in enumerate(questions):
            seen = set()
            p_names = [id2par[x] for x in top_k[idx] if not (id2par[x] in seen or seen.add(id2par[x]))][:k]
            questions_eval_format.append(
                {'qid': question.question_id, 'question': ' '.join(question.question),
                 'answers': list(question.answers),
                 'paragraphs': [documents['_'.join(p_name.split('_')[:-1])][int(p_name.split('_')[-1])]
                                for p_name in p_names]})

    with open(out_file, 'w') as f:
        json.dump(questions_eval_format, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Encode all DrQA-SQuAD data')
    # parser.add_argument('model', help='model directory to use for encoding')
    # parser.add_argument('outdir', help="output directory")
    # parser.add_argument('--num-workers', type=int, default=16)
    # parser.add_argument('--all-squad', action='store_true')
    # args = parser.parse_args()
    #
    # squad_build_drqa_doc_encodings(args.outdir, args.model, args.num_workers, args.all_squad)
    parser.add_argument('model', help='model directory to use for encoding')
    parser.add_argument('encodings_dir', help="directory with drqa data and encodings")
    parser.add_argument('out_file', help="filename to dump the top-k dataset")
    parser.add_argument('top_k', type=int)
    parser.add_argument('--all-dev', action='store_true')
    args = parser.parse_args()
    build_doc_eval_file(args.out_file, args.encodings_dir, args.model, args.top_k, not args.all_dev)
