""" Build questions and documents files for Hotpot dev with distractors. Just for consistency with when we'll
 use the whole wikipedia dump """


import argparse
import json
from multiprocessing.util import Finalize
from typing import List, Dict, Tuple
from multiprocessing import Pool as ProcessPool

from tqdm import tqdm

from hotpot import config


def build_hotpot_distractors_dataset(questions_file, docs_file):
    print("loading Hotpot data...")
    with open(config.HOTPOT_DEV_DISTRACTOR_FILE, 'r') as f:
        dev_data = json.load(f)

    questions = [{'qid': q['_id'], 'question': q['question'], 'answers': [q['answer']], 'type': q['type'],
                  'supporting_facts': q['supporting_facts'], 'top_titles': [x[0] for x in q['context']]}
                 for q in dev_data]
    # todo adjust hotpot to work with sentences instead of plain texts?
    title2paragraphs = {x[0]: [' '.join(x[1])] for q in dev_data for x in q['context']}
    with open(questions_file, 'w') as f:
        json.dump(questions, f)
    with open(docs_file, 'w') as f:
        json.dump(title2paragraphs, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Build the HotpotQA dataset for encoding')
    parser.add_argument('questions_file', help='json file to save the questions with the top titles')
    parser.add_argument('docs_file', help="json filename to dump the top-k dataset")
    args = parser.parse_args()
    build_hotpot_distractors_dataset(args.questions_file, args.docs_file)
