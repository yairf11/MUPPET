"""Script for evaluating tf-idf retriever"""
import argparse
import json
import numpy as np
import time
from copy import deepcopy
import prettytable
from tqdm import tqdm

from hotpot import config
from hotpot.tfidf_retriever.tfidf_doc_ranker import TfidfDocRanker
from hotpot.tfidf_retriever.doc_db import DocDB


def load_dataset(dataset_path):
    with open(dataset_path, 'r') as f:
        data = json.load(f)
    return data


def build_gold_dict(dataset):
    """Build a mapping from a question id to its gold paragraphs' titles

    :param dataset: the loaded dataset
    :return: dictionary of id: list of strings
    """
    return {q['_id']: list(set([fact[0] for fact in q['supporting_facts']])) for q in dataset}


def build_questions_dict(dataset):
    return {q['_id']: q['question'] for q in dataset}


def build_ranked_golds(dataset, docdb, ranker):
    golds = build_gold_dict(dataset)
    questions = build_questions_dict(dataset)
    ranked_dict = {}
    for q_id, pars in tqdm(golds.items()):
        par1_score = ranker.get_similarity_with_doc(questions[q_id], pars[0])
        par2_score = ranker.get_similarity_with_doc(questions[q_id], pars[1])
        best_par = pars[0] if par1_score > par2_score else pars[1]
        second_par = pars[1] if par1_score > par2_score else pars[0]
        new_query = questions[q_id] + ' '.join(docdb.get_doc_sentences(best_par))
        ranked_dict[q_id] = {'best_par': best_par, 'question': questions[q_id], 'q_with_best_par': new_query,
                             'second_par': second_par}
    return ranked_dict


def get_top_k(dataset, ranker, k, num_workers=1):
    questions = [q['question'] for q in dataset]
    ids = [q['_id'] for q in dataset]
    top_k = ranker.batch_closest_docs(questions, k=k, num_workers=num_workers)
    return {ids[i]: {'pars': top_k[i][0], 'level': dataset[i]['level'], 'type': dataset[i]['type']}
            for i in range(len(ids))}


def get_ranked_top_k(dataset, golds_rank_dict, ranker, k, num_workers=1):
    questions = [q['question'] for q in dataset]
    ids = [q['_id'] for q in dataset]
    new_questions = [golds_rank_dict[qid]['q_with_best_par'] for qid in ids]
    top_k = ranker.batch_closest_docs(questions, k=k, num_workers=num_workers)
    regular_top_k_dict = {ids[i]: {'pars': top_k[i][0], 'level': dataset[i]['level'], 'type': dataset[i]['type']}
                          for i in range(len(ids))}
    new_top_k = ranker.batch_closest_docs(new_questions, k=k, num_workers=num_workers)
    new_top_k_dict = {ids[i]: {'pars': new_top_k[i][0], 'level': dataset[i]['level'], 'type': dataset[i]['type']}
                      for i in range(len(ids))}
    return regular_top_k_dict, new_top_k_dict


CATEGORIES = ['comparison-hard', 'comparison-medium', 'comparison-easy', 'bridge-hard',
              'bridge-medium', 'bridge-easy', 'comparison', 'bridge', 'easy', 'medium', 'hard']


def top_k_coverage_score(gold_dict, top_k_dict, k):
    regular_scores = {'Hits': [], 'Perfect Questions': [], 'At Least One': []}
    categories_scores = {cat: deepcopy(regular_scores) for cat in CATEGORIES}

    def update_single(score_dict, hits):
        perfect = 1 if sum(hits) == len(hits) else 0
        at_least_one = 1 if sum(hits) > 0 else 0
        score_dict['Hits'].extend(hits)
        score_dict['Perfect Questions'].append(perfect)
        score_dict['At Least One'].append(at_least_one)

    def update_cats(hits, q_type, level):
        update_single(categories_scores[f"{q_type}-{level}"], hits)
        update_single(categories_scores[f"{q_type}"], hits)
        update_single(categories_scores[f"{level}"], hits)

    for qid, gold_pars in gold_dict.items():
        question_hits = []
        for gold_par in gold_pars:
            question_hits.append(1 if gold_par in top_k_dict[qid]['pars'][:k] else 0)
        update_single(regular_scores, question_hits)
        update_cats(question_hits, q_type=top_k_dict[qid]['type'], level=top_k_dict[qid]['level'])

    return {k: np.mean(v) for k, v in regular_scores.items()}, \
           {cat: {k: np.mean(v) for k, v in cat_scores.items()} for cat, cat_scores in categories_scores.items()}


def modified_top_k_coverage_score(ranked_gold_dict, top_k_dict, k):
    regular_scores = {'Second Paragraph Hits': []}
    categories_scores = {cat: deepcopy(regular_scores) for cat in CATEGORIES}

    def update_single(score_dict, hit):
        score_dict['Second Paragraph Hits'].append(hit)

    def update_cats(hits, q_type, level):
        update_single(categories_scores[f"{q_type}-{level}"], hits)
        update_single(categories_scores[f"{q_type}"], hits)
        update_single(categories_scores[f"{level}"], hits)

    for qid, vals in ranked_gold_dict.items():
        second_par_hit = 1 if vals['second_par'] in top_k_dict[qid]['pars'][:k] else 0
        update_single(regular_scores, second_par_hit)
        update_cats(second_par_hit, q_type=top_k_dict[qid]['type'], level=top_k_dict[qid]['level'])

    return {k: np.mean(v) for k, v in regular_scores.items()}, \
           {cat: {k: np.mean(v) for k, v in cat_scores.items()} for cat, cat_scores in categories_scores.items()}


def modified_main(dataset_path, k_list_to_check, ranker_path=None, normalize_ranker=False, num_workers=1,
                  tokenizer='corenlp', docdb_path=None, out=None):
    dataset = load_dataset(dataset_path)
    ranker = TfidfDocRanker(tfidf_path=ranker_path, normalize_vectors=normalize_ranker, tokenizer=tokenizer)
    docdb = DocDB(docdb_path)
    print("Building modified queries...")
    ranked_gold_dict = build_ranked_golds(dataset, docdb=docdb, ranker=ranker)
    regular_table = prettytable.PrettyTable(['Top K', 'Second Paragraph Hits', 'Second Paragraph Hits Modified Query'])
    cat_table_dict = {cat: prettytable.PrettyTable(['Top K', 'Second Paragraph Hits',
                                                    'Second Paragraph Hits Modified Query'])
                      for cat in CATEGORIES}
    max_k = max(k_list_to_check)
    print(f"Retrieving top {max_k} ...")
    start = time.time()
    reg_result_dict, ranked_result_dict = get_ranked_top_k(dataset, ranked_gold_dict, ranker, max_k, num_workers)
    print(f"Done, took {time.time()-start} ms.")
    for k in k_list_to_check:
        print(f"Calculating scores for top {k}...")
        start = time.time()
        reg_scores, reg_category_scores = modified_top_k_coverage_score(ranked_gold_dict, reg_result_dict, k)
        mod_scores, mod_category_scores = modified_top_k_coverage_score(ranked_gold_dict, ranked_result_dict, k)
        print(f"Done, took {time.time()-start} ms.")
        regular_table.add_row([k, reg_scores['Second Paragraph Hits'], mod_scores['Second Paragraph Hits']])
        for cat in cat_table_dict:
            cat_table_dict[cat].add_row([k, reg_category_scores[cat]['Second Paragraph Hits'],
                                         mod_category_scores[cat]['Second Paragraph Hits']])
    output_str = 'Overall Results:\n'
    output_str += regular_table.__str__() + '\n'
    for cat, table in cat_table_dict.items():
        output_str += '\n**********************************************\n'
        output_str += f"Category: {cat} Results:\n"
        output_str += table.__str__() + '\n'

    if out is None:
        print(output_str)
    else:
        with open(out, 'w') as f:
            f.write(output_str)


def main(dataset_path, k_list_to_check, ranker_path=None, normalize_ranker=False, num_workers=1, filter=False,
         tokenizer='corenlp'):
    dataset = load_dataset(dataset_path)
    if filter:
        print(f"Filtering questions without question mark.")
        old_len = len(dataset)
        dataset = [x for x in dataset if '?' in x['question']]
        print(f"Removed {len(dataset)-old_len} questions.")
    ranker = TfidfDocRanker(tfidf_path=ranker_path, normalize_vectors=normalize_ranker, tokenizer=tokenizer)
    gold_dict = build_gold_dict(dataset)
    regular_table = prettytable.PrettyTable(['Top K', 'Hits', 'Perfect Questions', 'At Least One'])
    cat_table_dict = {cat: prettytable.PrettyTable(['Top K', 'Hits', 'Perfect Questions', 'At Least One'])
                      for cat in CATEGORIES}
    max_k = max(k_list_to_check)
    print(f"Retrieving top {max_k} ...")
    start = time.time()
    result_dict = get_top_k(dataset, ranker, max_k, num_workers)
    print(f"Done, took {time.time()-start} ms.")
    for k in k_list_to_check:
        print(f"Calculating scores for top {k}...")
        start = time.time()
        scores, category_scores = top_k_coverage_score(gold_dict, result_dict, k)
        print(f"Done, took {time.time()-start} ms.")
        regular_table.add_row([k, scores['Hits'], scores['Perfect Questions'], scores['At Least One']])
        for cat in cat_table_dict:
            cat_table_dict[cat].add_row([k, category_scores[cat]['Hits'], category_scores[cat]['Perfect Questions'],
                                         category_scores[cat]['At Least One']])
    print("Overall Results:")
    print(regular_table)
    for cat, table in cat_table_dict.items():
        print('\n**********************************************\n')
        print(f"Category: {cat} Results:")
        print(table)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=str)
    parser.add_argument('--normalize_ranker', action="store_true")
    parser.add_argument('--ranker', type=str, default=None,
                        help='Path to tf-idf ranker')
    parser.add_argument('--docdb', type=str, default=None, help='Path to document DB')
    parser.add_argument('-k', '--top_k', nargs='+', type=int, default=[5])
    parser.add_argument('--num-workers', type=int, default=1)
    parser.add_argument('--tokenizer', type=str, default='corenlp')
    parser.add_argument('--filter', action='store_true')
    parser.add_argument('--modified', action='store_true')
    parser.add_argument('--out', type=str, default=None, help="name of output file")
    args = parser.parse_args()

    if args.modified:
        modified_main(args.dataset, args.top_k, args.ranker, args.normalize_ranker, num_workers=args.num_workers,
                      tokenizer=args.tokenizer, docdb_path=args.docdb, out=args.out)
    else:
        main(args.dataset, args.top_k, args.ranker, args.normalize_ranker, num_workers=args.num_workers,
             filter=args.filter,
             tokenizer=args.tokenizer)
