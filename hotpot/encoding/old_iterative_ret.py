import argparse
import json
import time
import itertools
from multiprocessing.util import Finalize
from typing import Tuple, List, Dict
from multiprocessing import Pool as ProcessPool
import pickle
import numpy as np
import prettytable
from tqdm import tqdm

from hotpot.data_handling.dataset import QuestionAndParagraphsSpec
from hotpot.encoding.encode_documents import par_name_to_title, DocumentEncodingHandler
from hotpot.encoding.knn import simple_numpy_knn, numpy_global_knn
from hotpot.encoding.paragraph_encoder import SentenceEncoderSingleContext, SentenceEncoderIterativeModel
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


def build_openqa_iterative_top_titles(out_file, questions_file, docs_file, encodings_dir, encoder_model, k1, k2,
                                      n1, n2, evaluate: bool, reformulate_from_text: bool,
                                      use_ema: bool, checkpoint: str):
    print('Loading data...')
    s = time.time()
    with open(questions_file, 'r') as f:
        questions = json.load(f)
    with open(docs_file, 'r') as f:
        documents = json.load(f)
    print(f'Done, took {time.time()-s} seconds.')

    if n1 is not None and n2 is not None:
        for q in questions:
            q['top_titles'] = q['top_titles'][:max(n1, n2)]

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
    title2encs = {}
    title2idx2par_name = {}
    with tqdm(total=len(all_titles)) as pbar:
        for t2enc, t2id2p in tqdm(workers.imap_unordered(get_title_mappings_from_saver, all_titles)):
            title2encs.update(t2enc)
            title2idx2par_name.update(t2id2p)
            pbar.update()
    title2par_name2idxs = {}
    for title, id2par in title2idx2par_name.items():
        par2idxs = {}
        for idx, parname in id2par.items():
            if parname in par2idxs:
                par2idxs[parname].append(idx)
            else:
                par2idxs[parname] = [idx]
        title2par_name2idxs[title] = {par: sorted(idxs) for par, idxs in par2idxs.items()}

    print("Loading encoder...")
    spec = QuestionAndParagraphsSpec(batch_size=None, max_num_contexts=2,
                                     max_num_question_words=None, max_num_context_words=None)
    encoder = SentenceEncoderIterativeModel(model_dir_path=encoder_model, vocabulary=voc, spec=spec,
                                            loader=ResourceLoader(), use_char_inputs=False, use_ema=use_ema,
                                            checkpoint=checkpoint)

    print("Encoding questions...")
    q_original_encodings = encoder.encode_text_questions([qid2tokenized[q['qid']] for q in questions],
                                                         return_search_vectors=False, show_progress=True)
    q_search_encodings = encoder.question_rep_to_search_vector(question_encodings=q_original_encodings)

    init()  # for initializing the tokenizer

    print("Calculating similarities...")
    for idx, question in tqdm(enumerate(questions), total=len(questions)):
        title2ids = {}
        all_par_reps = []
        total_sentences = 0
        titles_offset_dict = {}
        for title in question['top_titles'][:n1]:
            titles_offset_dict[title] = total_sentences
            rep = title2encs[title]
            title2ids[title] = list(range(total_sentences, total_sentences + len(rep)))
            all_par_reps.append(rep)
            total_sentences += len(rep)
        id2title = {i: title for title, ids in title2ids.items() for i in ids}
        all_par_reps = np.concatenate(all_par_reps, axis=0)

        q_enc = q_search_encodings[idx]
        top_k = simple_numpy_knn(np.expand_dims(q_enc, 0), all_par_reps, k1 * 2)[0]

        def id_to_par_name(rep_id):
            return title2idx2par_name[id2title[rep_id]][rep_id - titles_offset_dict[id2title[rep_id]]]

        seen = set()
        p_names = [id_to_par_name(x)
                   for x in top_k if not (id_to_par_name(x) in seen or seen.add(id_to_par_name(x)))][:k1]
        iteration1_paragraphs = \
            [title2encs[par_name_to_title(pname)][title2par_name2idxs[par_name_to_title(pname)][pname], :]
             for pname in p_names]
        if not reformulate_from_text:
            reformulations = encoder.reformulate_questions(questions_rep=np.tile(q_original_encodings[idx],
                                                                                 reps=(len(p_names), 1)),
                                                           paragraphs_rep=iteration1_paragraphs,
                                                           return_search_vectors=True)
        else:
            tok_q = tokenize(question['question']).words()
            par_texts = [tokenize(parname_to_text(pname)).words() for pname in p_names]
            reformulations = encoder.reformulate_questions_from_texts(
                tokenized_questions=[tok_q for _ in range(len(par_texts))],
                tokenized_pars=par_texts,
                return_search_vectors=True
            )

        title2ids = {}
        all_par_reps = []
        total_sentences = 0
        titles_offset_dict = {}
        for title in question['top_titles'][:n2]:
            titles_offset_dict[title] = total_sentences
            rep = title2encs[title]
            title2ids[title] = list(range(total_sentences, total_sentences + len(rep)))
            all_par_reps.append(rep)
            total_sentences += len(rep)
        id2title = {i: title for title, ids in title2ids.items() for i in ids}
        all_par_reps = np.concatenate(all_par_reps, axis=0)

        top_k_second = numpy_global_knn(reformulations, all_par_reps, k2 * k1)
        seen = set()
        final_p_name_pairs = [(p_names[x1], id_to_par_name(x2))
                              for x1, x2 in top_k_second
                              if not ((p_names[x1], id_to_par_name(x2)) in seen
                                      or seen.add((p_names[x1], id_to_par_name(x2))))][:k2]

        # important to note that in the iterative dataset the paragraphs of each question are in pairs
        question['paragraph_pairs'] = final_p_name_pairs

    with open(out_file, 'w') as f:
        json.dump(questions, f)

    if evaluate:
        eval_questions(questions)


def get_stats_for_single_k(true_titles, title_pairs):
    true_titles = tuple(sorted(true_titles))
    title_pairs = [tuple(sorted(x)) for x in title_pairs]
    single_titles = set([title for titles in title_pairs for title in titles])
    return dict(hits=np.mean([1 if title in single_titles else 0 for title in true_titles]),
                perfect=true_titles in title_pairs,
                at_least_one=np.mean([1 if title in single_titles else 0 for title in true_titles]) > 0)


def get_stats_for_sample(true_titles, sorted_title_pairs, top_k=None):
    top_k = top_k or len(sorted_title_pairs)
    return {k: get_stats_for_single_k(true_titles, sorted_title_pairs[:k]) for k in range(1, top_k+1)}


def eval_questions(questions, top_k=None):
    k_scores = {}
    bridge_k_scores = {}
    comparision_k_scores = {}
    for question in questions:
        q_stats = get_stats_for_sample(set([x[0] for x in question['supporting_facts']]),
                                       sorted_title_pairs=[[par_name_to_title(x) for x in pair]
                                                           for pair in question['paragraph_pairs']],
                                       top_k=top_k)
        type_scores = bridge_k_scores if question['type'] == 'bridge' else comparision_k_scores
        for k in q_stats:
            for score_dict in [k_scores, type_scores]:
                if k not in score_dict:
                    score_dict[k] = {key: [val] for key, val in q_stats[k].items()}
                else:
                    for key, val in q_stats[k].items():
                        score_dict[k][key].append(val)

    for score_dict in [k_scores, bridge_k_scores, comparision_k_scores]:
        for k in score_dict.keys():
            score_dict[k] = {key: np.mean(val) for key, val in score_dict[k].items()}

    for score_dict, name in [(k_scores, 'Overall'), (bridge_k_scores, 'Bridge'), (comparision_k_scores, 'Comparison')]:
        results = prettytable.PrettyTable(['Top K Pairs', 'Hits', 'Perfect Questions', 'At Least One'])
        for k, k_dict in score_dict.items():
            results.add_row([k, k_dict['hits'], k_dict['perfect'], k_dict['at_least_one']])
        results.sortby = 'Top K Pairs'

        print(f"{name} scores:")
        print(results)
        print('\n**********************************************\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create an iterative open-qa encoding dataset')
    parser.add_argument('out_file_prefix', help="filename prefix to dump the top-k dataset")
    parser.add_argument('questions_file', help="filename load the questions")
    parser.add_argument('docs_file', help="filename to load the documents")
    parser.add_argument('doc_enc_dir', help='directory of document encodings')
    parser.add_argument('encoder_model', help='model directory to use for encoding')
    parser.add_argument('k1', type=int)
    parser.add_argument('k2', type=int)
    parser.add_argument('--n1', type=int, default=None)
    parser.add_argument('--n2', type=int, default=None)
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--rft', action='store_true', help="reformulate from text?")
    parser.add_argument('--checkpoint', type=str, default='best', choices=['best', 'latest'])
    parser.add_argument('--ema', action='store_true')
    args = parser.parse_args()
    n1_str = f"_n1-{args.n1}" if args.n1 else ''
    n2_str = f"_n2-{args.n2}" if args.n2 else ''
    out_file = args.out_file_prefix + f"_k1-{args.k1}_k2-{args.k2}{n1_str}{n2_str}.json"
    build_openqa_iterative_top_titles(out_file=out_file, questions_file=args.questions_file,
                                      docs_file=args.docs_file,
                                      encodings_dir=args.doc_enc_dir,
                                      encoder_model=args.encoder_model, k1=args.k1, k2=args.k2, n1=args.n1, n2=args.n2,
                                      evaluate=args.eval,
                                      reformulate_from_text=args.rft, use_ema=args.ema, checkpoint=args.checkpoint)
