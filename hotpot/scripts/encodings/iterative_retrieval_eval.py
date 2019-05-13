import argparse
import json

from hotpot.encoding.iterative_encoding_retrieval import eval_questions

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate retrieved paragraphs')
    parser.add_argument('questions_file', help="filename to load the questions with the paragraphs")
    parser.add_argument('--top-k', type=int, default=None)
    parser.add_argument('--specific', nargs='+', type=int, default=None)
    args = parser.parse_args()

    with open(args.questions_file, 'r') as f:
        questions = json.load(f)
    eval_questions(questions, args.top_k, args.specific)
