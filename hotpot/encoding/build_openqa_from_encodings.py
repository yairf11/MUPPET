import argparse
import json
import time
import itertools
from multiprocessing.util import Finalize
from typing import Tuple, List, Dict
from multiprocessing import Pool as ProcessPool
import pickle
import numpy as np
from tqdm import tqdm

from hotpot.data_handling.dataset import QuestionAndParagraphsSpec
from hotpot.encoding.encode_documents import par_name_to_title, DocumentEncodingHandler
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


def tokenize_question(q_with_id: Tuple[str, str]) -> Dict[str, List[str]]:
    return {q_with_id[0]: tokenize(q_with_id[1]).words()}


DOC_ENCS_HANDLER = None


def init_encoding_handler(encoding_dir):
    global DOC_ENCS_HANDLER
    DOC_ENCS_HANDLER = DocumentEncodingHandler(encoding_dir)


def get_title_mappings_from_saver(title):
    return ({title: DOC_ENCS_HANDLER.get_document_encodings(title)},
            {title: DOC_ENCS_HANDLER.get_document_idx2pname(title)})


def build_openqa_top_titles(out_file, questions_file, docs_file, encodings_dir, encoder_model, k,
                            use_ema: bool, checkpoint: str, safety_mult: int, n_titles: int):
    print('Loading data...')
    s = time.time()
    with open(questions_file, 'r') as f:
        questions = json.load(f)
    with open(docs_file, 'r') as f:
        documents = json.load(f)
    print(f'Done, took {time.time()-s} seconds.')

    # Setup worker pool
    workers = ProcessPool(
        16,
        initializer=init,
        initargs=[]
    )

    qid2tokenized = {}
    tupled_questions = [(q['qid'], q['question']) for q in questions]
    print("Tokenizing questions...")
    with tqdm(total=len(tupled_questions)) as pbar:
        for tok_q in tqdm(workers.imap_unordered(tokenize_question, tupled_questions)):
            qid2tokenized.update(tok_q)
            pbar.update()

    workers.close()
    workers.join()

    voc = set()
    for question in qid2tokenized.values():
        voc.update(question)

    all_titles = list(set([title for q in questions for title in q['top_titles']]))

    def parname_to_text(par_name):
        par_title = par_name_to_title(par_name)
        par_num = int(par_name.split('_')[-1])
        return documents[par_title][par_num]

    print(f"Gathering documents...")
    # Setup worker pool
    workers = ProcessPool(
        32,
        initializer=init_encoding_handler,
        initargs=[encodings_dir]
    )
    # title2encs = {}
    # title2idx2par_name = {}
    # with tqdm(total=len(all_titles)) as pbar:
    #     for t2enc, t2id2p in tqdm(workers.imap_unordered(get_title_mappings_from_saver, all_titles)):
    #         title2encs.update(t2enc)
    #         title2idx2par_name.update(t2id2p)
    #         pbar.update()

    print("Loading encoder...")
    spec = QuestionAndParagraphsSpec(batch_size=None, max_num_contexts=1,
                                     max_num_question_words=None, max_num_context_words=None)
    encoder = SentenceEncoderSingleContext(model_dir_path=encoder_model, vocabulary=voc, spec=spec,
                                           loader=ResourceLoader(), use_char_inputs=False,
                                           use_ema=use_ema, checkpoint=checkpoint)

    print("Encoding questions...")
    q_encodings = encoder.encode_text_questions([qid2tokenized[q['qid']] for q in questions],
                                                return_search_vectors=True, show_progress=True)

    print("Calculating similarities...")
    for idx, question in tqdm(enumerate(questions), total=len(questions), ncols=80):
        q_titles = question['top_titles']
        if n_titles is not None:
            q_titles = q_titles[:n_titles]
        title2encs = {}
        title2idx2par_name = {}
        for t2enc, t2id2p in workers.imap_unordered(get_title_mappings_from_saver, q_titles):
            title2encs.update(t2enc)
            title2idx2par_name.update(t2id2p)
        q_enc = q_encodings[idx]
        title2ids = {}
        reps = []
        total_sentences = 0
        titles_offset_dict = {}
        for title in q_titles:
            titles_offset_dict[title] = total_sentences
            rep = title2encs[title]
            title2ids[title] = list(range(total_sentences, total_sentences + len(rep)))
            reps.append(rep)
            total_sentences += len(rep)
        id2title = {i: title for title, ids in title2ids.items() for i in ids}
        reps = np.concatenate(reps, axis=0)
        top_k = simple_numpy_knn(np.expand_dims(q_enc, 0), reps, k * safety_mult)[0]

        def id_to_par_name(rep_id):
            return title2idx2par_name[id2title[rep_id]][rep_id - titles_offset_dict[id2title[rep_id]]]

        seen = set()
        p_names = [id_to_par_name(x)
                   for x in top_k if not (id_to_par_name(x) in seen or seen.add(id_to_par_name(x)))][:k]
        question['paragraphs'] = [parname_to_text(p_name) for p_name in p_names]

    with open(out_file, 'w') as f:
        json.dump(questions, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create an open-qa encoding dataset')
    parser.add_argument('out_file', help="filename to dump the top-k dataset")
    parser.add_argument('questions_file', help="filename load the questions")
    parser.add_argument('docs_file', help="filename to load the documents")
    parser.add_argument('doc_enc_dir', help='directory of document encodings')
    parser.add_argument('encoder_model', help='model directory to use for encoding')
    parser.add_argument('top_k', type=int)
    parser.add_argument('--checkpoint', type=str, default='best', choices=['best', 'latest'])
    parser.add_argument('--ema', action='store_true')
    parser.add_argument('--safety_mult', type=int, default=4, help="retrieve k*safety, "
                                                                     "useful when there are multiple encodings.")
    parser.add_argument('--n_titles', type=int, default=None, help="number of top titles to use")
    args = parser.parse_args()
    build_openqa_top_titles(out_file=args.out_file, questions_file=args.questions_file, docs_file=args.docs_file,
                            encodings_dir=args.doc_enc_dir,
                            encoder_model=args.encoder_model, k=args.top_k,
                            use_ema=args.ema, checkpoint=args.checkpoint, safety_mult=args.safety_mult,
                            n_titles=args.n_titles)


