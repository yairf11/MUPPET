import argparse

from hotpot.elmo.lm_model import dump_token_embeddings

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='dump pretrained elmo token embeddings')
    parser.add_argument("vocab_file", help="Where to store the vocabulary file")
    parser.add_argument("options_file")
    parser.add_argument("weight_file")
    parser.add_argument("embed_file", help="Where to dump the token embeddings")
    args = parser.parse_args()
    dump_token_embeddings(
        args.vocab_file, args.options_file, args.weight_file, args.embed_file
    )
