import argparse
from multiprocessing.util import Finalize
from typing import List, Tuple
from multiprocessing import Pool as ProcessPool

import json
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import pairwise_distances
from tqdm import tqdm
from collections import Counter

from hotpot import trainer, config
from hotpot.data_handling.dataset import multiple_contexts_len, ClusteredBatcher, QuestionAndParagraphsDataset, \
    ListBatcher, QuestionAndParagraphsSpec
from hotpot.data_handling.hotpot.hotpot_data import HotpotQuestions
from hotpot.data_handling.hotpot.hotpot_qa_training_data import HotpotTextLengthPreprocessorWithSpans, \
    HotpotFullQADistractorsDataset, get_segments_from_sentences_fix_sup
from hotpot.data_handling.qa_training_data import SpanQuestionAndParagraphs
from hotpot.encoding.encode_documents import par_name_to_title
from hotpot.evaluator import Evaluator, Evaluation
from hotpot.model_dir import ModelDir
from hotpot.scripts.train_eval.ranked_scores import compute_ranked_scores, compute_ranked_scores_with_yes_no
from hotpot.scripts.train_eval.ranked_scores_to_hotpot_pred import df_to_pred
from hotpot.tfidf_retriever.doc_db import DocDB
from hotpot.tfidf_retriever.utils import STOPWORDS
from hotpot.tokenizers import CoreNLPTokenizer
from hotpot.utils import transpose_lists, print_table, ResourceLoader, flatten_iterable
from hotpot_evaluate_v1 import exact_match_score as hotpot_em_score, update_sp
from hotpot_evaluate_v1 import f1_score as hotpot_f1_score

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


def tokenize_texts(texts, num_workers=None, sentences=False):
    if num_workers is None or num_workers == 1:
        init()
        return [tokenize_words(text) if not sentences else tokenize_sentences(text) for text in texts]
    else:
        workers = ProcessPool(
            num_workers,
            initializer=init,
            initargs=[]
        )
        res = []
        with tqdm(total=len(texts), desc='Tokenize') as pbar:
            for tok in tqdm(workers.imap(tokenize_words if not sentences else tokenize_sentences, texts)):
                res.append(tok)
                pbar.update()
        return res


class DummyDataset(QuestionAndParagraphsDataset):
    def __init__(self, questions, batcher: ListBatcher):
        self.questions = questions
        self.batcher = batcher

    def get_epoch(self):
        return self.batcher.get_epoch(self.questions)

    def get_spec(self):
        batch_size = self.batcher.get_fixed_batch_size()
        num_contexts = 1
        max_q_words = max(len(q.question) for q in self.questions)
        max_c_words = max(multiple_contexts_len(q) for q in self.questions)
        return QuestionAndParagraphsSpec(batch_size=batch_size, max_num_contexts=num_contexts,
                                         max_num_question_words=max_q_words, max_num_context_words=max_c_words)

    def get_vocab(self):
        voc = set()
        for q in self.questions:
            voc.update(q.question)
            for para in q.paragraphs:
                voc.update(para)
        return voc

    def get_word_counts(self):
        count = Counter()
        for q in self.questions:
            count.update(q.question)
            for para in q.paragraphs:
                count.update(para)
        return count

    def __len__(self):
        return self.batcher.epoch_size(len(self.questions))


class RankedQAPair(SpanQuestionAndParagraphs):
    def __init__(self, question, paragraphs, spans,
                 question_id: str, answer: str, rank: int,
                 q_type: str = 'not_relevant', sentence_segments=None,
                 par_titles_num_sents: List[Tuple[str, int]] = None, missing_sent_idxs: List[List[int]] = None,
                 true_sp=None):
        super().__init__(question, paragraphs, spans, question_id, answer, q_type, sentence_segments)
        self.rank = rank
        self.par_titles_num_sents = par_titles_num_sents
        self.missing_sent_idxs = missing_sent_idxs
        self.true_sp = true_sp

    def get_text_span(self, start, end_exclusive):
        return " ".join(self.paragraphs[0][start:end_exclusive])

    def get_supporting_facts(self, sent_idxs):
        if len(sent_idxs) == 0:
            return []
        # (title1, num_sents1), (title2, num_sents2) = self.par_titles_num_sents
        #
        # def fix_idxs(num_sents, missing_idxs):
        #     sent_idxs = list(range(num_sents))
        #     for i in missing_idxs:
        #         sent_idxs.insert(i, -1)
        #     return sent_idxs
        #
        # title1_sent_idxs = fix_idxs(num_sents1, self.missing_sent_idxs[0])
        # title2_sent_idxs = fix_idxs(num_sents2, self.missing_sent_idxs[1])
        #
        # sp_facts = []
        # for idx in sent_idxs:
        #     if idx >= num_sents1 + num_sents2:
        #         continue
        #     if idx >= num_sents1:
        #         sp_facts.append([title2, title2_sent_idxs.index(idx - num_sents1)])
        #     else:
        #         sp_facts.append([title1, title1_sent_idxs.index(idx)])
        # return sp_facts

        def fix_idxs(num_sents, missing_idxs):
            sent_idxs = list(range(num_sents))
            for i in missing_idxs:
                sent_idxs.insert(i, -1)
            return sent_idxs

        titles_sent_idxs = [fix_idxs(num_sents, missing) for (title, num_sents), missing in
                            zip(self.par_titles_num_sents, self.missing_sent_idxs)]

        sp_facts = []
        curr_par_idx = 0
        for idx in sorted(sent_idxs):
            idx -= sum(x[1] for x in self.par_titles_num_sents[:curr_par_idx])
            while curr_par_idx < len(self.par_titles_num_sents) and idx >= self.par_titles_num_sents[curr_par_idx][1]:
                idx -= self.par_titles_num_sents[curr_par_idx][1]
                curr_par_idx += 1
            if curr_par_idx >= len(self.par_titles_num_sents):
                break
            sp_facts.append([self.par_titles_num_sents[curr_par_idx][0], titles_sent_idxs[curr_par_idx].index(idx)])
        return sp_facts


class RecordHotpotQAPrediction(Evaluator):

    def __init__(self, bound: int, record_text_ans: bool, sp_prediction: bool, disable_tqdm=False):
        self.bound = bound
        self.record_text_ans = record_text_ans
        self.sp_prediction = sp_prediction
        self.disable_tqdm = disable_tqdm

    def tensors_needed(self, prediction):
        span, score = prediction.get_best_span(self.bound)
        is_yes_no_scores = prediction.get_is_yes_no_scores()
        yes_or_no_scores = prediction.get_yes_or_no_scores()
        tensor_dict = dict(spans=span, span_scores=score, is_yes_no_scores=is_yes_no_scores,
                           yes_or_no_scores=yes_or_no_scores)
        if self.sp_prediction:
            sentence_scores = prediction.get_sentence_scores()
            tensor_dict.update(dict(sentence_scores=sentence_scores))
        return tensor_dict

    def evaluate(self, data: List[RankedQAPair], true_len, **kargs):
        spans, span_scores = np.array(kargs["spans"]), np.array(kargs["span_scores"])
        is_yes_no_scores = np.array(kargs["is_yes_no_scores"])
        yes_or_no_scores = np.array(kargs["yes_or_no_scores"])
        if self.sp_prediction:
            sp_scores = np.array(kargs["sentence_scores"])

        pred_f1s = np.zeros(len(data))
        pred_em = np.zeros(len(data))
        text_answers = []
        supporting_facts = []
        pred_sp_f1s = np.zeros(len(data))
        pred_sp_em = np.zeros(len(data))
        pred_joint_f1s = np.zeros(len(data))
        pred_joint_em = np.zeros(len(data))

        for i in tqdm(range(len(data)), total=len(data), ncols=80, desc="scoring", disable=self.disable_tqdm):
            point = data[i]
            if point.answer is None and not self.record_text_ans:
                continue
            pred_span = spans[i]
            if is_yes_no_scores[i][0] >= is_yes_no_scores[i][1]:
                pred_text = point.get_text_span(pred_span[0], pred_span[1] + 1)
            else:
                pred_text = 'yes' if yes_or_no_scores[i][1] >= yes_or_no_scores[i][0] else 'no'

            sp_metrics = {'sp_em': 0, 'sp_f1': 0, 'sp_prec': 0, 'sp_recall': 0}
            if self.sp_prediction:
                pred_sp = point.get_supporting_facts(np.argwhere(sp_scores[i] >= 0.5).squeeze(axis=1))
                update_sp(sp_metrics, pred_sp, point.true_sp)
                pred_sp_em[i] = sp_metrics['sp_em']
                pred_sp_f1s[i] = sp_metrics['sp_f1']
                supporting_facts.append(pred_sp)

            if self.record_text_ans:
                text_answers.append(pred_text)
                if point.answer is None:
                    continue

            f1, precision, recall = hotpot_f1_score(pred_text, data[i].answer)
            em = hotpot_em_score(pred_text, data[i].answer)

            pred_f1s[i] = f1
            pred_em[i] = em

            if self.sp_prediction:
                joint_prec = precision * sp_metrics['sp_prec']
                joint_recall = recall * sp_metrics['sp_recall']
                if joint_prec + joint_recall > 0:
                    pred_joint_f1s[i] = 2 * joint_prec * joint_recall / (joint_prec + joint_recall)
                pred_joint_em[i] = em * sp_metrics['sp_em']

        results = {}
        results["n_answers"] = [0 if x.answer is None else 1 for x in data]
        if self.record_text_ans:
            results["text_answer"] = text_answers
        if self.sp_prediction:
            results["predicted_sp"] = supporting_facts
            results["sp_em"] = pred_sp_em
            results["sp_f1"] = pred_sp_f1s
            results["joint_em"] = pred_joint_em
            results["joint_f1"] = pred_joint_f1s
        results["predicted_score"] = span_scores
        results["predicted_start"] = spans[:, 0]
        results["predicted_end"] = spans[:, 1]
        results["span_question_scores"] = is_yes_no_scores[:, 0]
        results["yes_no_question_scores"] = is_yes_no_scores[:, 1]
        results["yes_no_confidence_scores"] = yes_or_no_scores.max(axis=1)
        results["text_f1"] = pred_f1s
        results["rank"] = [x.rank for x in data]
        results["text_em"] = pred_em
        results["question_id"] = [x.question_id for x in data]
        results["type"] = [x.q_type for x in data]
        results["titles"] = [[title for title, _ in x.par_titles_num_sents] for x in data]
        return Evaluation({}, results)


def get_paragraph_ranks(question: str, paragraphs: List[str]):
    tfidf = TfidfVectorizer(strip_accents="unicode", stop_words=STOPWORDS)

    # para_features = tfidf.fit_transform(paragraphs)
    # q_features = tfidf.transform([question])
    q_features = tfidf.fit_transform([question])
    para_features = tfidf.transform(paragraphs)
    distances = pairwise_distances(q_features, para_features, "cosine")[0]
    return np.argsort(np.argsort(distances))


def truncate_paragraph(tokenized_sentences: List[List[str]], num_tokens):
    while len(flatten_iterable(tokenized_sentences)) > num_tokens:
        tokenized_sentences = tokenized_sentences[:-1]
    return tokenized_sentences


def main():
    parser = argparse.ArgumentParser(description='Full ranking evaluation on Hotpot')
    parser.add_argument('model', help='model directory to evaluate')
    parser.add_argument('output', type=str,
                        help="Store the per-paragraph results in csv format in this file, "
                             "or the json prediction if in test mode")
    parser.add_argument('-n', '--sample_questions', type=int, default=None,
                        help="(for testing) run on a subset of questions")
    parser.add_argument('-b', '--batch_size', type=int, default=64,
                        help="Batch size, larger sizes can be faster but uses more memory")
    parser.add_argument('-s', '--step', default=None,
                        help="Weights to load, can be a checkpoint step or 'latest'")
    parser.add_argument('-a', '--answer_bound', type=int, default=8,
                        help="Max answer span length")
    parser.add_argument('-c', '--corpus',
                        choices=["distractors", "gold", "hotpot_file", "retrieval_file", "top_titles"],
                        default="distractors")
    parser.add_argument('-t', '--tokens', type=int, default=None,
                        help="Max tokens per a paragraph")
    parser.add_argument('--input_file', type=str, default=None)
    parser.add_argument('--docs_file', type=str, default=None)
    parser.add_argument('--num_workers', type=int, default=16, help='Number of workers for tokenizing')
    parser.add_argument('--no_ema', action="store_true", help="Don't use EMA weights even if they exist")
    parser.add_argument('--no_sp', action="store_true", help="Don't predict supporting facts")
    parser.add_argument('--test_mode', action='store_true', help="produce a prediction file, no answers given")
    args = parser.parse_args()

    model_dir = ModelDir(args.model)
    batcher = ClusteredBatcher(args.batch_size, multiple_contexts_len, truncate_batches=True)
    loader = ResourceLoader()

    if args.corpus not in {"distractors", "gold"} and args.input_file is None:
        raise ValueError("Must pass an input file if not using precomputed dataset")

    if args.corpus in {"distractors", "gold"} and args.test_mode:
        raise ValueError("Test mode not available in 'distractors' or 'gold' mode")

    if args.corpus in {"distractors", "gold"}:

        corpus = HotpotQuestions()
        loader = corpus.get_resource_loader()
        questions = corpus.get_dev()

        question_preprocessor = HotpotTextLengthPreprocessorWithSpans(args.tokens)
        questions = [question_preprocessor.preprocess(x) for x in questions
                     if (question_preprocessor.preprocess(x) is not None)]

        if args.sample_questions:
            np.random.RandomState(0).shuffle(sorted(questions, key=lambda x: x.question_id))
            questions = questions[:args.sample_questions]

        data = HotpotFullQADistractorsDataset(questions, batcher)
        gold_idxs = set(data.gold_idxs)
        if args.corpus == 'gold':
            data.samples = [data.samples[i] for i in data.gold_idxs]
        qid2samples = {}
        qid2idx = {}
        for i, sample in enumerate(data.samples):
            key = sample.question_id
            if key in qid2samples:
                qid2samples[key].append(sample)
                qid2idx[key].append(i)
            else:
                qid2samples[key] = [sample]
                qid2idx[key] = [i]
        questions = []
        print("Ranking pairs...")
        gold_ranks = []
        for qid, samples in tqdm(qid2samples.items()):
            question = " ".join(samples[0].question)
            pars = [" ".join(x.paragraphs[0]) for x in samples]
            ranks = get_paragraph_ranks(question, pars)
            for sample, rank, idx in zip(samples, ranks, qid2idx[qid]):
                questions.append(RankedQAPair(question=sample.question, paragraphs=sample.paragraphs,
                                              spans=np.zeros((0, 2), dtype=np.int32), question_id=sample.question_id,
                                              answer=sample.answer, rank=rank,
                                              q_type=sample.q_type, sentence_segments=sample.sentence_segments))
                if idx in gold_idxs:
                    gold_ranks.append(rank + 1)
        print(f"Mean rank: {np.mean(gold_ranks)}")
        ranks_counter = Counter(gold_ranks)
        for i in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
            print(f"Hits at {i}: {ranks_counter[i]}")

    elif args.corpus == 'hotpot_file':  # a hotpot json format input file. We rank the pairs with tf-idf
        with open(args.input_file, 'r') as f:
            hotpot_data = json.load(f)
        if args.sample_questions:
            np.random.RandomState(0).shuffle(sorted(hotpot_data, key=lambda x: x['_id']))
            hotpot_data = hotpot_data[:args.sample_questions]
        title2sentences = {context[0]: context[1] for q in hotpot_data for context in q['context']}
        question_tok_texts = tokenize_texts([q['question'] for q in hotpot_data], num_workers=args.num_workers)
        sentences_tok = tokenize_texts(list(title2sentences.values()), num_workers=args.num_workers, sentences=True)
        if args.tokens is not None:
            sentences_tok = [truncate_paragraph(p, args.tokens) for p in sentences_tok]
        title2tok_sents = {title: sentences for title, sentences in zip(title2sentences.keys(), sentences_tok)}
        questions = []
        for idx, question in enumerate(tqdm(hotpot_data, desc='tf-idf ranking')):
            q_titles = [title for title, _ in question['context']]
            par_pairs = [(title1, title2) for i, title1 in enumerate(q_titles) for title2 in q_titles[i + 1:]]
            if len(par_pairs) == 0:
                continue
            ranks = get_paragraph_ranks(question['question'],
                                        [' '.join(title2sentences[t1] + title2sentences[t2]) for t1, t2 in par_pairs])
            for rank, par_pair in zip(ranks, par_pairs):
                sent_tok_pair = title2tok_sents[par_pair[0]] + title2tok_sents[par_pair[1]]
                sentence_segments, _ = get_segments_from_sentences_fix_sup(sent_tok_pair, np.zeros(0))
                missing_sent_idx = [[i for i, sent in enumerate(title2tok_sents[title]) if len(sent) == 0]
                                    for title in par_pair]
                questions.append(RankedQAPair(question=question_tok_texts[idx],
                                              paragraphs=[flatten_iterable(sent_tok_pair)],
                                              spans=np.zeros((0, 2), dtype=np.int32), question_id=question['_id'],
                                              answer='noanswer' if args.test_mode else question['answer'], rank=rank,
                                              q_type='null' if args.test_mode else question['type'],
                                              sentence_segments=[sentence_segments],
                                              par_titles_num_sents=
                                              [(title, sum(1 for sent in title2tok_sents[title] if len(sent) > 0))
                                               for title in par_pair],
                                              missing_sent_idxs=missing_sent_idx,
                                              true_sp=[] if args.test_mode else question['supporting_facts']))
    elif args.corpus == 'retrieval_file' or args.corpus == 'top_titles':
        if args.docs_file is None:
            print("Using DB documents")
            doc_db = DocDB(config.DOC_DB, full_docs=False)
        else:
            with open(args.docs_file, 'r') as f:
                docs = json.load(f)
        with open(args.input_file, 'r') as f:
            retrieval_data = json.load(f)
        if args.sample_questions:
            np.random.RandomState(0).shuffle(sorted(retrieval_data, key=lambda x: x['qid']))
            retrieval_data = retrieval_data[:args.sample_questions]

        def parname_to_text(par_name):
            par_title = par_name_to_title(par_name)
            par_num = int(par_name.split('_')[-1])
            if args.docs_file is None:
                return doc_db.get_doc_sentences(par_title)
            return docs[par_title][par_num]

        if args.corpus == 'top_titles':
            print("Top TF-IDF!")
            for q in retrieval_data:
                top_titles = q['top_titles'][:10]
                q['paragraph_pairs'] = [(title1 + '_0', title2 + '_0') for i, title1 in enumerate(top_titles)
                                        for title2 in top_titles[i + 1:]]

        question_tok_texts = tokenize_texts([q['question'] for q in retrieval_data], num_workers=args.num_workers)
        all_parnames = list(set([parname for q in retrieval_data for pair in q['paragraph_pairs'] for parname in pair]))
        texts_tok = tokenize_texts([parname_to_text(x) for x in all_parnames], num_workers=args.num_workers,
                                   sentences=True)
        if args.tokens is not None:
            texts_tok = [truncate_paragraph(p, args.tokens) for p in texts_tok]
        parname2tok_text = {parname: text for parname, text in zip(all_parnames, texts_tok)}
        questions = []
        for idx, question in enumerate(retrieval_data):
            for rank, par_pair in enumerate(question['paragraph_pairs']):
                tok_pair = parname2tok_text[par_pair[0]] + parname2tok_text[par_pair[1]]
                sentence_segments, _ = get_segments_from_sentences_fix_sup(tok_pair, np.zeros(0))
                missing_sent_idx = [[i for i, sent in enumerate(parname2tok_text[parname]) if len(sent) == 0]
                                    for parname in par_pair]
                questions.append(RankedQAPair(question=question_tok_texts[idx], paragraphs=[flatten_iterable(tok_pair)],
                                              spans=np.zeros((0, 2), dtype=np.int32), question_id=question['qid'],
                                              answer='noanswer' if args.test_mode else question['answers'][0],
                                              rank=rank,
                                              q_type='null' if args.test_mode else question['type'],
                                              sentence_segments=[sentence_segments],
                                              par_titles_num_sents=
                                              [(par_name_to_title(parname),
                                                sum(1 for sent in parname2tok_text[parname] if len(sent) > 0))
                                               for parname in par_pair],
                                              missing_sent_idxs=missing_sent_idx,
                                              true_sp=[] if args.test_mode else question['supporting_facts']))
    else:
        raise NotImplementedError()

    data = DummyDataset(questions, batcher)
    evaluators = [RecordHotpotQAPrediction(args.answer_bound, True, sp_prediction=not args.no_sp)]

    if args.step is not None:
        if args.step == "latest":
            checkpoint = model_dir.get_latest_checkpoint()
        else:
            checkpoint = model_dir.get_checkpoint(int(args.step))
    else:
        checkpoint = model_dir.get_best_weights()
        if checkpoint is not None:
            print("Using best weights")
        else:
            print("Using latest checkpoint")
            checkpoint = model_dir.get_latest_checkpoint()

    model = model_dir.get_model()

    evaluation = trainer.test(model, evaluators, {args.corpus: data},
                              loader, checkpoint, not args.no_ema, 10)[args.corpus]

    print("Saving result")
    output_file = args.output

    df = pd.DataFrame(evaluation.per_sample)

    df.sort_values(["question_id", "rank"], inplace=True, ascending=True)
    group_by = ["question_id"]

    def get_ranked_scores(score_name):
        filtered_df = df[df.type == 'comparison'] if "Cp" in score_name else \
            df[df.type == 'bridge'] if "Br" in score_name else df
        target_prefix = 'joint' if 'joint' in score_name else 'sp' if 'sp' in score_name else 'text'
        target_score = f"{target_prefix}_{'em' if 'EM' in score_name else 'f1'}"
        return compute_ranked_scores_with_yes_no(filtered_df, span_q_col="span_question_scores",
                                                 yes_no_q_col="yes_no_question_scores",
                                                 yes_no_scores_col="yes_no_confidence_scores",
                                                 span_scores_col="predicted_score", span_target_score=target_score,
                                                 group_cols=group_by)

    if not args.test_mode:
        score_names = ["EM", "F1", "Br EM", "Br F1", "Cp EM", "Cp F1"]
        if not args.no_sp:
            score_names.extend([f"{prefix} {name}" for prefix in ['sp', 'joint'] for name in score_names])

        table = [["N Paragraphs"] + score_names]
        scores = [get_ranked_scores(score_name) for score_name in score_names]
        table += list([str(i + 1), *["%.4f" % x for x in score_vals]]
                      for i, score_vals in enumerate(zip(*scores)))
        print_table(table)

        df.to_csv(output_file, index=False)

    else:
        df_to_pred(df, output_file)


if __name__ == "__main__":
    main()
