import argparse
from collections import Counter


def load_counts(counts_file):
    word_counter = Counter()
    with open(counts_file, 'r') as f:
        for line in f:
            k, v = line.split('\t')
            if k.strip() == '':
                continue
            word_counter[k] = int(v.strip())
    return word_counter


def build_elmo_vocab(word_counts_file, title_word_counts_file, word_th, titles_th, out_file):
    word_counts = load_counts(word_counts_file)
    title_word_counts = load_counts(title_word_counts_file)

    words = set([x for x in word_counts if word_counts[x] >= word_th])
    title_words = set([x for x in title_word_counts if title_word_counts[x] >= titles_th])

    voc = {'<S>', '</S>'}
    voc.update(words)
    voc.update(title_words)

    with open(out_file, 'w') as f:
        f.write('\n'.join(voc))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create and save a vocabulary from word counts')
    parser.add_argument("out_path", type=str, help='path to save word counts')
    parser.add_argument("counts_file", type=str)
    parser.add_argument("titles_counts_file", type=str)
    parser.add_argument("--word_th", type=int, default=5, help="threshold on word counts")
    parser.add_argument("--title_th", type=int, default=1, help="threshold on title word counts")
    args = parser.parse_args()

    build_elmo_vocab(args.counts_file, args.titles_counts_file, args.word_th, args.title_th, args.out_path)
