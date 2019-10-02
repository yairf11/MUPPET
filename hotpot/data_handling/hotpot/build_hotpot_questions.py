import argparse
from multiprocessing.util import Finalize
from os import mkdir

from os.path import exists, join
from typing import List
from multiprocessing import Pool as ProcessPool


from tqdm import tqdm

from hotpot import utils
from hotpot import config
from hotpot.data_handling.hotpot.hotpot_data import HotpotQuestions, HotpotQuestion, HotpotGoldParagraph, HotpotParagraph
from hotpot.tfidf_retriever.doc_db import DocDB
from hotpot.tokenizers.corenlp_tokenizer import CoreNLPTokenizer


DOC2IDX = None
PROCESS_TOK = None
PROCESS_DB = None


def init(db_path):
    global PROCESS_TOK, PROCESS_DB
    PROCESS_TOK = CoreNLPTokenizer()
    Finalize(PROCESS_TOK, PROCESS_TOK.shutdown, exitpriority=100)
    PROCESS_DB = DocDB(db_path)
    Finalize(PROCESS_DB, PROCESS_DB.close, exitpriority=100)


def fetch_sentences(doc_title):
    global PROCESS_DB
    return PROCESS_DB.get_doc_sentences(doc_title)


def tokenize(text):
    global PROCESS_TOK
    return PROCESS_TOK.tokenize(text)


def parse_single(question):
    supporting_dict = {}
    gold_pars = []
    for sup in question['supporting_facts']:
        if sup[0] in supporting_dict:
            supporting_dict[sup[0]].append(sup[1])
        else:
            supporting_dict[sup[0]] = [sup[1]]

    distractor_pars = []
    distractor_titles = [c[0] for c in question['context'] if c[0] not in supporting_dict]

    def get_sentences(title):
        sentences = fetch_sentences(title)
        if sentences is None:
            raise ValueError(f"{title} was not found! (for question id: {question['_id']})")
        return [s for s in sentences if len(s) != 0]

    for sup_title, sup_sentence_ids in supporting_dict.items():
        sentences = get_sentences(sup_title)
        tok_sentences = [tokenize(s).words() for s in sentences]
        gold_pars.append(HotpotGoldParagraph(sup_title, sentences=tok_sentences,
                                             question_id=question['_id'], supporting_sentence_ids=sup_sentence_ids))
    for distractor_title in distractor_titles:
        sentences = get_sentences(distractor_title)
        tok_sentences = [tokenize(s).words() for s in sentences]
        distractor_pars.append(HotpotParagraph(distractor_title, sentences=tok_sentences))
    q_tokens = tokenize(question['question']).words()
    return HotpotQuestion(question_id=question['_id'], question_tokens=q_tokens,
                          answer=question['answer'], supporting_facts=gold_pars, distractors=distractor_pars,
                          q_type=question['type'], level=question['level'],
                          gold_scores=None, distractor_scores=None,
                          gold_spans=None, distractor_spans=None)  # todo this is a hack, but works for now


def parse_hotpot_data_async(data_path, db_path, num_workers) -> List[HotpotQuestion]:
    data = utils.load_json_dataset(data_path)
    questions = []

    # Setup worker pool
    workers = ProcessPool(
        num_workers,
        initializer=init,
        initargs=[db_path]
    )

    with tqdm(total=len(data)) as pbar:
        for parsed_question in tqdm(workers.imap_unordered(parse_single, data)):
            questions.append(parsed_question)
            pbar.update()

    return questions


def parse_hotpot_data(data_path, tokenizer, docdb: DocDB) -> List[HotpotQuestion]:  # TODO: why aren't we loading this straight from the json?
    data = utils.load_json_dataset(data_path)
    questions = []
    for question in tqdm(data):
        supporting_dict = {}
        gold_pars = []
        for sup in question['supporting_facts']:
            if sup[0] in supporting_dict:
                supporting_dict[sup[0]].append(sup[1])
            else:
                supporting_dict[sup[0]] = [sup[1]]

        distractor_pars = []
        distractor_titles = [c[0] for c in question['context'] if c[0] not in supporting_dict]

        def get_sentences(title):
            sentences = docdb.get_doc_sentences(title)
            if sentences is None:
                raise ValueError(f"{title} was not found! (for question id: {question['_id']})")
            return [s for s in sentences if len(s) != 0]

        for sup_title, sup_sentence_ids in supporting_dict.items():
            sentences = get_sentences(sup_title)
            tok_sentences = [tokenizer.tokenize(s).words() for s in sentences]
            gold_pars.append(HotpotGoldParagraph(sup_title, sentences=tok_sentences,
                                                 question_id=question['_id'], supporting_sentence_ids=sup_sentence_ids))
        for distractor_title in distractor_titles:
            sentences = get_sentences(distractor_title)
            tok_sentences = [tokenizer.tokenize(s).words() for s in sentences]
            distractor_pars.append(HotpotParagraph(distractor_title, sentences=tok_sentences))
        q_tokens = tokenizer.tokenize(question['question']).words()
        questions.append(HotpotQuestion(question_id=question['_id'], question_tokens=q_tokens,
                                        answer=question['answer'], supporting_facts=gold_pars, distractors=distractor_pars,
                                        q_type=question['type'], level=question['level'],
                                        gold_scores=None, distractor_scores=None,
                                        gold_spans=None, distractor_spans=None)
                         )  # todo this is a hack, but works for now
    return questions


def main():  # todo: add scores to questions & paragraphs
    parser = argparse.ArgumentParser("Preprocess Hotpot Questions")
    parser.add_argument("--train_file", default=config.HOTPOT_TRAIN_FILE)
    parser.add_argument("--dev_file", default=config.HOTPOT_DEV_DISTRACTOR_FILE)
    parser.add_argument("--doc_db", default=None)
    parser.add_argument('--num-workers', type=int, default=1, help='Number of CPU processes')

    if not exists(join(config.CORPUS_DIR, 'hotpot')):
        mkdir(join(config.CORPUS_DIR, 'hotpot'))

    args = parser.parse_args()

    # target_dir = config.CORPUS_DIR
    # if exists(target_dir) and len(listdir(target_dir)) > 0:
    #     raise ValueError("Files already exist in " + target_dir)

    if args.num_workers > 1:
        print(f"Multiprocessing with {args.num_workers} threads...")
        print("Parsing train...")
        train = parse_hotpot_data_async(args.train_file, args.doc_db, args.num_workers)

        print("Parsing dev...")
        dev = parse_hotpot_data_async(args.dev_file, args.doc_db, args.num_workers)

    else:
        tokenzier = CoreNLPTokenizer()
        docdb = DocDB(args.doc_db)

        print("Parsing train...")
        train = parse_hotpot_data(args.train_file, tokenzier, docdb)

        print("Parsing dev...")
        dev = parse_hotpot_data(args.dev_file, tokenzier, docdb)

    print("Saving...")
    HotpotQuestions.make_corpus(train, dev)
    print("Done")


if __name__ == "__main__":
    main()
