import argparse
import json

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import pairwise_distances
from tqdm import tqdm

from hotpot.tfidf_retriever.utils import STOPWORDS


def build_paragraph_tfidf_dataset(out_file, questions_file, docs_file, num_paragraphs, n_titles):
    with open(questions_file, 'r') as f:
        questions = json.load(f)

    with open(docs_file, 'r') as f:
        documents = json.load(f)

    all_questions = [question['question'] for question in questions]
    tfidf = TfidfVectorizer(strip_accents="unicode", stop_words=STOPWORDS)
    tfidf.fit(all_questions)

    for question in tqdm(questions):
        paragraphs = [par for doc in [documents[title] for title in question['top_titles'][:n_titles]] for par in doc]

        para_features = tfidf.transform(paragraphs)
        q_features = tfidf.transform([question['question']])
        distances = pairwise_distances(q_features, para_features, "cosine")
        top_k = np.argpartition(distances,
                                kth=list(range(min(len(paragraphs), num_paragraphs))), axis=1)[:, :num_paragraphs]
        question['paragraphs'] = [paragraphs[i] for i in top_k[0]]

    with open(out_file, 'w') as f:
        json.dump(questions, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create an open-qa tfidf dataset')
    parser.add_argument('out_file', help="filename to dump the top-k dataset")
    parser.add_argument('questions_file', help="filename load the questions")
    parser.add_argument('docs_file', help="filename to load the documents")
    parser.add_argument('top_k', type=int)
    parser.add_argument('--n_titles', type=int, default=None, help="number of top titles to use")
    args = parser.parse_args()
    build_paragraph_tfidf_dataset(args.out_file, args.questions_file, args.docs_file, args.top_k, args.n_titles)
