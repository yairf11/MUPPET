import argparse
import json

import nltk
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import pairwise_distances

from hotpot import config
from hotpot.tfidf_retriever.utils import STOPWORDS


def build_all_dev_rankings_file(out_file, docs_json, k):
    print("loading data...")
    with open(config.SQUAD_DEV_FILE, 'r') as f:
        dev_data = json.load(f)['data']

    questions = [{'qid': q['id'], 'question': q['question'], 'answers': [a['text'] for a in q['answers']]}
                 for doc in dev_data for par in doc['paragraphs'] for q in par['qas']]

    with open(docs_json, 'r') as f:
        documents = json.load(f)

    paragraphs = [par for doc in documents.values() for par in doc]

    # tokenizer = nltk.TreebankWordTokenizer()
    tfidf = TfidfVectorizer(strip_accents="unicode", stop_words=STOPWORDS)

    para_features = tfidf.fit_transform(paragraphs)
    q_features = tfidf.transform([q['question'] for q in questions])
    distances = pairwise_distances(q_features, para_features, "cosine")
    top_k = np.argpartition(distances, kth=list(range(k)), axis=1)[:, :k]
    for idx, q in enumerate(questions):
        q['paragraphs'] = [paragraphs[i] for i in top_k[idx]]

    with open(out_file, 'w') as f:
        json.dump(questions, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Encode all DrQA-SQuAD data')
    parser.add_argument('docs_json', help='model directory to use for ranking')
    parser.add_argument('out_file', help="filename to dump the top-k dataset")
    parser.add_argument('top_k', type=int)
    # parser.add_argument('--all-dev', action='store_true')
    args = parser.parse_args()
    build_all_dev_rankings_file(args.out_file, args.docs_json, args.top_k)
