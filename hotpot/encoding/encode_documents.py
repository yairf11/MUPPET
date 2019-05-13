import argparse
import json
from multiprocessing.util import Finalize
from typing import Dict, List, Tuple
from multiprocessing import Pool as ProcessPool
import itertools
import pickle
import numpy as np
import os

from os.path import join
from tqdm import tqdm

from hotpot.data_handling.relevance_training_data import BinaryQuestionAndParagraphs, \
    IterativeQuestionAndParagraphs
from hotpot.data_handling.dataset import QuestionAndParagraphsSpec
from hotpot.encoding.paragraph_encoder import SentenceEncoderSingleContext, SentenceEncoderIterativeModel
from hotpot.tfidf_retriever.doc_db import DocDB
from hotpot.tokenizers import CoreNLPTokenizer
from hotpot.utils import ResourceLoader

PROCESS_TOK = None
PROCESS_DB = None


def init():
    global PROCESS_TOK, PROCESS_DB
    PROCESS_TOK = CoreNLPTokenizer()
    Finalize(PROCESS_TOK, PROCESS_TOK.shutdown, exitpriority=100)
    PROCESS_DB = DocDB()
    Finalize(PROCESS_DB, PROCESS_DB.close, exitpriority=100)


def fetch_sentences(doc_title):
    global PROCESS_DB
    return PROCESS_DB.get_doc_sentences(doc_title)


def tokenize(text):
    global PROCESS_TOK
    return PROCESS_TOK.tokenize(text)


def tokenize_document(doc: Tuple[str, List[str]]) -> Dict[str, List[List[str]]]:
    return {doc[0]: [tokenize(x).words() for x in doc[1]]}


def tokenize_from_db(title: str) -> Dict[str, List[List[str]]]:
    return {title: [tokenize(' '.join(fetch_sentences(title))).words()]}


# class DocumentsEncodingSaver(object):
#     def __init__(self, encodings_path: str):
#         self.encodings_path = encodings_path
#         self.encodings = None
#         self.title2idx2par_name = None
#
#     def _load_encodings(self):
#         self.encodings = np.load(self.encodings_path)
#
#     def get_document_encodings(self, title: str):
#         if self.encodings is None:
#             self._load_encodings()
#         return self.encodings[title]
#
#     def build_document_encodings_from_paragraphs(self, par_name2enc: Dict[str, np.ndarray]):
#         par_names = list(par_name2enc.keys())
#         title2par_names = {title: list(par_names)
#                            for title, par_names in
#                            itertools.groupby(sorted(par_names, key=par_name_to_title), key=par_name_to_title)}
#         title2encs = {}
#         self.title2idx2par_name = {}
#         for title, p_names in tqdm(title2par_names.items()):
#             par2ids = {}
#             reps = []
#             total_sentences = 0
#             for p_name in p_names:
#                 rep = par_name2enc[p_name]
#                 par2ids[p_name] = list(range(total_sentences, total_sentences + len(rep)))
#                 reps.append(rep)
#                 total_sentences += len(rep)
#             id2par = {i: p for p, ids in par2ids.items() for i in ids}
#             reps = np.concatenate(reps, axis=0)
#             title2encs[title] = reps
#             self.title2idx2par_name[title] = id2par
#         np.savez_compressed(self.encodings_path, **title2encs)


class DocumentEncodingHandler(object):
    def __init__(self, encodings_dir: str):
        self.encodings_dir = os.path.abspath(encodings_dir)
        self.titles2filenames = self._get_titles_to_filenames()

    def _title_to_filename_json(self):
        return join(self.encodings_dir, "title_to_filenames.json")

    def _get_titles_to_filenames(self):
        titles2files = {}
        if not os.path.exists(self._title_to_filename_json()):
            with open(self._title_to_filename_json(), 'w') as f:
                pass
            return {}
        with open(self._title_to_filename_json(), 'r') as f:
            for line in f:
                titles2files.update(json.loads(line))
        return titles2files

    def _title_to_npy(self, title: str):
        return join(self.encodings_dir, f"{self.titles2filenames[title]}.npy")

    def _title_to_idx2parname(self, title: str):
        return join(self.encodings_dir, f"{self.titles2filenames[title]}_idx2pname.pkl")

    def get_document_encodings(self, title: str) -> np.ndarray:
        return np.load(self._title_to_npy(title))

    def get_document_idx2pname(self, title: str) -> Dict[int, str]:
        with open(self._title_to_idx2parname(title), 'rb') as f:
            return pickle.load(f)

    def save_document_encoding(self, par_name2enc: Dict[str, np.ndarray], overwrite=False):
        title = par_name_to_title(next(iter(par_name2enc)))
        if title in self.titles2filenames and not overwrite:
            raise ValueError(f"Overwrite enabled, {title} encodings already exist")
        par2ids = {}
        reps = []
        total_sentences = 0
        for p_name in par_name2enc:
            if par_name_to_title(p_name) != title:
                raise ValueError("All paragraphs must belong to the same title")
            rep = par_name2enc[p_name]
            par2ids[p_name] = list(range(total_sentences, total_sentences + len(rep)))
            reps.append(rep)
            total_sentences += len(rep)
        id2par = {i: p for p, ids in par2ids.items() for i in ids}
        reps = np.concatenate(reps, axis=0)
        if title not in self.titles2filenames:
            self.titles2filenames[title] = str(len(self.titles2filenames))
            with open(self._title_to_filename_json(), 'a') as f:
                json.dump({title: self.titles2filenames[title]}, f)
                f.write(os.linesep)
        with open(self._title_to_idx2parname(title), 'wb') as f:
            pickle.dump(id2par, f)
        np.save(self._title_to_npy(title), reps)

    def save_multiple_documents(self, par_name2enc: Dict[str, np.ndarray], overwrite=False):
        par_names = list(par_name2enc.keys())
        title2par_names = {title: list(par_names)
                           for title, par_names in
                           itertools.groupby(sorted(par_names, key=par_name_to_title), key=par_name_to_title)}
        for title, p_names in tqdm(title2par_names.items()):
            self.save_document_encoding({p_name: par_name2enc[p_name] for p_name in p_names}, overwrite=overwrite)

    # def convert_single_file_to_current_format(self, old_saver: DocumentsEncodingSaver):
    #     for title in tqdm(old_saver.title2idx2par_name.keys()):
    #         encs = old_saver.get_document_encodings(title)
    #         idx2par_names = old_saver.title2idx2par_name[title]
    #         self.titles2filenames[title] = str(len(self.titles2filenames))
    #         with open(self._title_to_filename_json(), 'a') as f:
    #             json.dump({title: self.titles2filenames[title]}, f)
    #             f.write(os.linesep)
    #         with open(self._title_to_idx2parname(title), 'wb') as f:
    #             pickle.dump(idx2par_names, f)
    #         np.save(self._title_to_npy(title), encs)


def par_name_to_title(par_name):
    return '_'.join(par_name.split('_')[:-1])


def encode_from_file(docs_file, questions_file, encodings_dir, encoder_model, num_workers, hotpot: bool,
                     long_batch: int, short_batch: int, use_chars: bool, use_ema: bool, checkpoint: str,
                     document_chunk_size=1000, samples=None, encode_all_db=False):
    """

    :param out_file: .npz file to dump the encodings
    :param docs_file: path to json file whose structure is [{title: list of paragraphs}, ...]
    :return:
    """
    doc_encs_handler = DocumentEncodingHandler(encodings_dir)
    # Setup worker pool
    workers = ProcessPool(
        num_workers,
        initializer=init,
        initargs=[]
    )

    if docs_file is not None:
        with open(docs_file, 'r') as f:
            documents = json.load(f)
        documents = {k: v for k, v in documents.items() if k not in doc_encs_handler.titles2filenames}

        tokenized_documents = {}
        tupled_doc_list = [(title, pars) for title, pars in documents.items()]

        if samples is not None:
            print(f"sampling {samples} samples")
            tupled_doc_list = tupled_doc_list[:samples]

        print("Tokenizing from file...")
        with tqdm(total=len(tupled_doc_list), ncols=80) as pbar:
            for tok_doc in tqdm(workers.imap_unordered(tokenize_document, tupled_doc_list)):
                tokenized_documents.update(tok_doc)
                pbar.update()
    else:
        if questions_file is not None:
            with open(questions_file, 'r') as f:
                questions = json.load(f)
            all_titles = list(set([title for q in questions for title in q['top_titles']]))
        else:
            print("encoding all DB!")
            all_titles = DocDB().get_doc_titles()

        if samples is not None:
            print(f"sampling {samples} samples")
            all_titles = all_titles[:samples]

        all_titles = [t for t in all_titles if t not in doc_encs_handler.titles2filenames]
        tokenized_documents = {}

        print("Tokenizing from DB...")
        with tqdm(total=len(all_titles), ncols=80) as pbar:
            for tok_doc in tqdm(workers.imap_unordered(tokenize_from_db, all_titles)):
                tokenized_documents.update(tok_doc)
                pbar.update()

    workers.close()
    workers.join()

    voc = set()
    for paragraphs in tokenized_documents.values():
        for par in paragraphs:
            voc.update(par)

    if not hotpot:
        spec = QuestionAndParagraphsSpec(batch_size=None, max_num_contexts=1,
                                         max_num_question_words=None, max_num_context_words=None)
        encoder = SentenceEncoderSingleContext(model_dir_path=encoder_model, vocabulary=voc, spec=spec,
                                               loader=ResourceLoader(), use_char_inputs=use_chars,
                                               use_ema=use_ema, checkpoint=checkpoint)
    else:
        spec = QuestionAndParagraphsSpec(batch_size=None, max_num_contexts=2,
                                         max_num_question_words=None, max_num_context_words=None)
        encoder = SentenceEncoderIterativeModel(model_dir_path=encoder_model, vocabulary=voc, spec=spec,
                                                loader=ResourceLoader(), use_char_inputs=use_chars,
                                                use_ema=use_ema, checkpoint=checkpoint)

    tokenized_documents_items = list(tokenized_documents.items())
    for tokenized_doc_chunk in tqdm([tokenized_documents_items[i:i + document_chunk_size]
                                     for i in range(0, len(tokenized_documents_items), document_chunk_size)],
                                    ncols=80):
        flattened_pars_with_names = [(f"{title}_{i}", par)
                                     for title, pars in tokenized_doc_chunk for i, par in enumerate(pars)]

        # filtering out empty paragraphs (probably had some short string the tokenization removed)
        # important to notice that the filtered paragraphs will have no representation,
        # but they still exist in the numbering of paragraphs for consistency with the docs.
        flattened_pars_with_names = [(name, par) for name, par in flattened_pars_with_names if len(par) > 0]

        # sort such that longer paragraphs are first to identify OOMs early on
        flattened_pars_with_names = sorted(flattened_pars_with_names, key=lambda x: len(x[1]), reverse=True)
        long_paragraphs_ids = [i for i, name_par in enumerate(flattened_pars_with_names) if len(name_par[1]) >= 900]
        short_paragraphs_ids = [i for i, name_par in enumerate(flattened_pars_with_names) if len(name_par[1]) < 900]

        # print(f"Encoding {len(flattened_pars_with_names)} paragraphs...")
        name2enc = {}
        dummy_question = "Hello Hello".split()
        if not hotpot:
            model_paragraphs = [BinaryQuestionAndParagraphs(question=dummy_question,
                                                            paragraphs=[x], label=1,
                                                            num_distractors=0, question_id='dummy')
                                for _, x in flattened_pars_with_names]
        else:
            # todo allow precomputed sentence segments
            model_paragraphs = [IterativeQuestionAndParagraphs(question=dummy_question,
                                                               paragraphs=[x, dummy_question],
                                                               first_label=1, second_label=1,
                                                               question_id='dummy', sentence_segments=None)
                                for _, x in flattened_pars_with_names]

        # print("Encoding long paragraphs...")
        long_pars = [model_paragraphs[i] for i in long_paragraphs_ids]
        name2enc.update({flattened_pars_with_names[long_paragraphs_ids[i]][0]: enc
                         for i, enc in
                         enumerate(encoder.encode_paragraphs(long_pars, batch_size=long_batch, show_progress=True)
                                   if not hotpot
                                   else encoder.encode_first_paragraphs(long_pars, batch_size=long_batch,
                                                                        show_progress=True))})

        # print("Encoding short paragraphs...")
        short_pars = [model_paragraphs[i] for i in short_paragraphs_ids]
        name2enc.update({flattened_pars_with_names[short_paragraphs_ids[i]][0]: enc
                         for i, enc in enumerate(encoder.encode_paragraphs(short_pars, batch_size=short_batch,
                                                                           show_progress=True)
                                                 if not hotpot
                                                 else encoder.encode_first_paragraphs(short_pars,
                                                                                      batch_size=short_batch,
                                                                                      show_progress=True)
                                                 )})

        doc_encs_handler.save_multiple_documents(name2enc)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Encode a dataset')
    parser.add_argument('encodings_dir', help="directory to dump the encodings")
    parser.add_argument('encoder_model', help="model to encode with")
    parser.add_argument('--docs_file', default=None, help="a document json filename from which to load the top-k dataset")
    parser.add_argument('--questions_file', default=None,
                        help="a questions json filename from which to load the top-k dataset."
                             " For hotpot, loads docs from DB")
    parser.add_argument('--encode-all-db', action='store_true')
    parser.add_argument('--checkpoint', type=str, default='best', choices=['best', 'latest'])
    parser.add_argument('--ema', action='store_true')
    parser.add_argument('--num-workers', type=int, default=16)
    parser.add_argument('--hotpot', action='store_true')
    parser.add_argument('--long-batch', type=int, default=8)
    parser.add_argument('--short-batch', type=int, default=128)
    parser.add_argument('--use-chars', action='store_true')
    parser.add_argument('--doc-chunk', type=int, default=1000)
    parser.add_argument('--samples', type=int, default=None)
    args = parser.parse_args()
    if (args.docs_file and args.questions_file) or (not args.docs_file and not args.questions_file):
        if not args.encode_all_db or (args.encode_all_db and (args.docs_file or args.questions_file)):
            raise ValueError("please, questions file or docs file")
    if not args.hotpot and not args.docs_file:
        raise ValueError("only hotpot supports retrieving from db")
    encode_from_file(args.docs_file, args.questions_file, args.encodings_dir,
                     args.encoder_model, args.num_workers, hotpot=args.hotpot,
                     long_batch=args.long_batch, short_batch=args.short_batch, use_chars=args.use_chars,
                     document_chunk_size=args.doc_chunk, use_ema=args.ema, checkpoint=args.checkpoint,
                     samples=args.samples, encode_all_db=args.encode_all_db)
