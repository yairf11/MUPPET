import argparse
import code
from copy import deepcopy
from multiprocessing.util import Finalize

from os.path import join
from typing import List, Tuple
from multiprocessing import Pool as ProcessPool
from termcolor import colored

import numpy as np
import pandas as pd
import tensorflow as tf

from hotpot.config import LOCAL_DATA_DIR
from hotpot.data_handling.dataset import QuestionAndParagraphsSpec, multiple_contexts_len, ClusteredBatcher
from hotpot.data_handling.hotpot.hotpot_data import HotpotQuestions
from hotpot.data_handling.hotpot.hotpot_qa_training_data import get_segments_from_sentences_fix_sup
from hotpot.elmo.build_elmo_vocab_from_db_counts import load_counts
from hotpot.elmo.lm_model import load_elmo_pretrained_token_embeddings
from hotpot.encoding.iterative_encoding_retrieval_batch import get_workers, initial_retrieval, reformulation_retrieval
from hotpot.encoding.paragraph_encoder import SentenceEncoderIterativeModel
from hotpot.evaluator import AysncEvaluatorRunner
from hotpot.model_dir import ModelDir
from hotpot.models.single_context_qa_models import AttentionQAFullHotpot
from hotpot.scripts.train_eval.hotpot_qa_distractors_eval import RecordHotpotQAPrediction, RankedQAPair, DummyDataset
from hotpot.scripts.train_eval.ranked_scores_to_hotpot_pred import df_to_pred
from hotpot.tfidf_retriever.doc_db import DocDB
from hotpot.tfidf_retriever.tfidf_doc_ranker import TfidfDocRanker
from hotpot.tokenizers import CoreNLPTokenizer
from hotpot.utils import ResourceLoader, flatten_iterable

parser = argparse.ArgumentParser(description='Interactive Multi-Hop Open-Domain QA!')
parser.add_argument('encoder_model', help='encoder model')
parser.add_argument('encodings_dir', help='directory of document encodings to use')
parser.add_argument('qa_model', help='QA model')
parser.add_argument('--num-workers', type=int, default=4, help="Number of workers for tokenizing etc")
parser.add_argument('--checkpoint', type=str, default='latest', choices=['best', 'latest'])
parser.add_argument('--no-ema', action="store_true", help="Don't use EMA weights even if they exist")
args = parser.parse_args()

PROCESS_TOK = None


def init():
    global PROCESS_TOK
    PROCESS_TOK = CoreNLPTokenizer()
    Finalize(PROCESS_TOK, PROCESS_TOK.shutdown, exitpriority=100)


def tokenize_words(text):
    global PROCESS_TOK
    return PROCESS_TOK.tokenize(text).words()


def tokenize_sentences(sentences):
    global PROCESS_TOK
    return [PROCESS_TOK.tokenize(s).words() if s != '' else [] for s in sentences]


print("Loading TF-IDF...")
tfidf_ranker = TfidfDocRanker()
db = DocDB()

loader = ResourceLoader()
# loader = HotpotQuestions().get_resource_loader()
word_counts = load_counts(join(LOCAL_DATA_DIR, 'hotpot', 'wiki_word_counts.txt'))
title_counts = load_counts(join(LOCAL_DATA_DIR, 'hotpot', 'wiki_title_word_counts.txt'))
word_counts.update(title_counts)
voc = set(word_counts.keys())

print("Loading encoder...")

spec = QuestionAndParagraphsSpec(batch_size=None, max_num_contexts=2,
                                 max_num_question_words=None, max_num_context_words=None)
encoder = SentenceEncoderIterativeModel(model_dir_path=args.encoder_model, vocabulary=voc,
                                        spec=spec, loader=loader, use_char_inputs=False,
                                        use_ema=not args.no_ema,
                                        checkpoint=args.checkpoint)

print("Loading QA model...")
evaluators = [RecordHotpotQAPrediction(15, True, sp_prediction=True, disable_tqdm=True)]
batcher = ClusteredBatcher(64, multiple_contexts_len, truncate_batches=True)
qa_model_dir = ModelDir(args.qa_model)
checkpoint = None
if checkpoint == 'best':
    checkpoint = qa_model_dir.get_best_weights()
if checkpoint is not None:
    print("Using best weights")
else:
    print("Using latest checkpoint")
    checkpoint = qa_model_dir.get_latest_checkpoint()
qa_model = qa_model_dir.get_model()
assert isinstance(qa_model, AttentionQAFullHotpot)
qa_sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True), graph=tf.Graph())
qa_spec = QuestionAndParagraphsSpec(batch_size=None, max_num_contexts=1,
                                    max_num_question_words=None, max_num_context_words=None)
with qa_sess.graph.as_default():
    qa_model.set_inputs(None, loader, voc=voc, input_spec=qa_spec)
    evaluator_runner = AysncEvaluatorRunner(evaluators, qa_model, 1)
    inputs = evaluator_runner.dequeue_op
    input_dict = {p: x for p, x in zip(qa_model.get_placeholders(), inputs)}
    with qa_sess.as_default():
        pred = qa_model.get_predictions_for(input_dict)  # for building the model
    evaluator_runner.set_input(pred)
    if not qa_model.use_elmo:
        saver = tf.train.Saver()
        saver.restore(qa_sess, checkpoint)
    if not args.no_ema:
        ema = tf.train.ExponentialMovingAverage(0)
        reader = tf.train.NewCheckpointReader(checkpoint)
        expected_ema_names = {ema.average_name(x): x for x in tf.trainable_variables()
                              if reader.has_tensor(ema.average_name(x))}
        if len(expected_ema_names) > 0:
            print("Restoring EMA variables")
            saver = tf.train.Saver(expected_ema_names)
            saver.restore(qa_sess, checkpoint)
if qa_model.use_elmo:
    elmo_token_embed_placeholder, elmo_token_embed_init = qa_model.get_elmo_token_embed_ph_and_op()
    print("Loading ELMo weights...")
    elmo_token_embed_weights = load_elmo_pretrained_token_embeddings(qa_model.lm_model.embed_weights_file)
    qa_sess.run(elmo_token_embed_init, feed_dict={elmo_token_embed_placeholder: elmo_token_embed_weights})

qa_sess.graph.finalize()

retrieval_workers = get_workers(args.num_workers, args.encodings_dir)

tok_workers = ProcessPool(
    4,
    initializer=init,
    initargs=[]
)

init()


def get_tfidf_top_k_pars(question: str, k: int):
    top_titles, scores = tfidf_ranker.closest_docs(question, k)
    top_pars = [(title, db.get_doc_sentences(title)) for title in top_titles]
    return top_pars


def get_answers(question: str, top_title_tuples: List[Tuple[str]]):
    question_tok = tokenize_words(question)
    all_titles = list(set([title for titles in top_title_tuples for title in titles]))
    texts_tok = [x for x in tok_workers.imap(tokenize_sentences, [db.get_doc_sentences(title) for title in all_titles])]
    title2tok_sents = {title: tok_sents for title, tok_sents in zip(all_titles, texts_tok)}
    questions = []
    for rank, title_tuple in enumerate(top_title_tuples):
        tokenized_sents = [sent for t in title_tuple for sent in title2tok_sents[t]]
        sentence_segments, _ = get_segments_from_sentences_fix_sup(tokenized_sents, np.zeros(0))
        missing_sent_idx = [[i for i, sent in enumerate(title2tok_sents[title]) if len(sent) == 0]
                            for title in title_tuple]
        questions.append(RankedQAPair(question=question_tok, paragraphs=[flatten_iterable(tokenized_sents)],
                                      spans=np.zeros((0, 2), dtype=np.int32), question_id='bla',
                                      answer='noanswer',
                                      rank=rank,
                                      q_type='n/a', sentence_segments=[sentence_segments],
                                      par_titles_num_sents=
                                      [(title,
                                        sum(1 for sent in title2tok_sents[title] if len(sent) > 0))
                                       for title in title_tuple],
                                      missing_sent_idxs=missing_sent_idx,
                                      true_sp=[]))
    data = DummyDataset(questions, batcher)
    evaluation = evaluator_runner.run_evaluators(qa_sess, data, 'bla', None, {}, disable_tqdm=True)
    df = pd.DataFrame(evaluation.per_sample)

    df.sort_values(["question_id", "rank"], inplace=True, ascending=True)
    answer_dict, sp_dict = df_to_pred(df, None, return_results=True)
    # sp_raw = [db.get_doc_sentences(title)[idx] for title, idx in sp_dict['bla']]
    title2idxs = {}
    for title, idx in sp_dict['bla']:
        if title not in title2idxs:
            title2idxs[title] = []
        title2idxs[title].append(idx)
    sp_titles_idxs = [(title, idxs) for title, idxs in title2idxs.items()]
    return answer_dict['bla'], sp_titles_idxs


def get_top_pars(question: str, k1: int, k2: int, n1: int, n2: int, possible_iters: List[int]):
    if max(possible_iters) < 1:
        print("Must have at least one iteration.")
        return None
    tfidf_titles, _ = tfidf_ranker.closest_docs(question, max(n1, n2))
    question_dict = dict(question=question, top_titles=tfidf_titles)
    iteration_pars = [deepcopy(initial_retrieval(encoder, retrieval_workers, [question_dict],
                                                 k1=max(k1, k2), n1=n1, safety_mult=1)[0]['top_pars_titles'])]
    question_dict['top_pars_titles'] = question_dict['top_pars_titles'][:k1]
    for i in range(max(possible_iters)-1):
        iteration_pars.append(deepcopy(reformulation_retrieval(encoder, retrieval_workers,
                                                               [question_dict], doc_db=db, k2=k2, n2=n2,
                                                               safety_mult=1)[0]['top_pars_titles']))
    return [titles for i, iteration in enumerate(iteration_pars) for titles in iteration if i+1 in possible_iters]


def answer(question: str, hops=(2,), k1=8, k2=45, n1=32, n2=512):
    if type(hops) == int:
        hops = (hops,)
    print("Retrieving...")
    top_pars = get_top_pars(question, k1, k2, n1, n2, hops)
    print("Reasoning...")
    answer, sp_facts = get_answers(question, top_pars)
    print(f"The answer is: {colored(answer, 'green', attrs=['bold'])}")
    print(f"The supporting facts are: ")
    for i, fact in enumerate(sp_facts):
        title = colored(fact[0], 'magenta', attrs=['bold'])
        sents = ' '.join([colored(sent, 'green', attrs=['underline']) if idx in fact[1] else sent
                          for idx, sent in enumerate(db.get_doc_sentences(fact[0]))])
        print(f"{i+1}. {title}: {sents}")


banner = """
Interactive Multi Hop QA
>> answer(question, hops=(2,), k1=8, k2=45, n1=32, n2=512)
>> usage()
"""


def usage():
    print(banner)


code.interact(banner=banner, local=locals())
