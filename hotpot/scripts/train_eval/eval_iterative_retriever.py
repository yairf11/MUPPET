import argparse
import json

from hotpot.encoding.iterative_encoding_retrieval import eval_questions

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate retriever for hotpot')
    parser.add_argument('iterative_dataset', help="filename of the iterative dataset")
    parser.add_argument('k', type=int, help="number of top pairs to evaluate on")
    args = parser.parse_args()

    with open(args.iterative_dataset, 'r') as f:
        questions = json.load(f)

    eval_questions(questions, top_k=args.k)
