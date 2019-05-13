import argparse
from typing import Dict, List
import numpy as np
import itertools

import os
from os.path import join

from tqdm import tqdm

from hotpot import config
from hotpot.data_handling.relevance_training_data import BinaryQuestionAndParagraphs
from hotpot.data_handling.dataset import QuestionAndParagraphsSpec
from hotpot.data_handling.squad.squad_data import SquadDocument, SquadQuestion, SquadRelevanceCorpus
from hotpot.encoding.paragraph_encoder import SentenceEncoderSingleContext


def encode_document(encoder: SentenceEncoderSingleContext, doc: SquadDocument) -> Dict[int, np.ndarray]:
    dummy_question = "Hello Hello".split()

    paragraphs = [BinaryQuestionAndParagraphs(question=dummy_question, paragraphs=[x.par_text], label=1,
                                              num_distractors=0, question_id='dummy') for x in doc.paragraphs]
    id_to_index = {x.par_id: idx for idx, x in enumerate(doc.paragraphs)}

    reps = encoder.encode_paragraphs(paragraphs)

    return {x.par_id: reps[id_to_index[x.par_id]] for x in doc.paragraphs}


def encode_questions(encoder: SentenceEncoderSingleContext, questions: List[SquadQuestion]):
    dummy_par = "Hello Hello".split()

    samples = [BinaryQuestionAndParagraphs(question=q.question, paragraphs=[dummy_par], label=1, num_distractors=0,
                                           question_id='dummy') for q in questions]

    return encoder.encode_questions(samples, return_search_vectors=True)


def get_filename(is_train: bool, doc_title: str):
    return join(config.SQUAD_ENCODINGS, 'train' if is_train else 'dev', doc_title + '.npz')


def encode_all_squad(encoder_model: str):
    print("loading data...")
    corpus = SquadRelevanceCorpus()
    train = corpus.get_train()
    dev = corpus.get_dev()

    spec = QuestionAndParagraphsSpec(batch_size=None, max_num_contexts=1,
                                     max_num_question_words=None, max_num_context_words=None)
    voc = corpus.get_vocab()

    encoder = SentenceEncoderSingleContext(model_dir_path=encoder_model, vocabulary=voc, spec=spec,
                                           loader=corpus.get_resource_loader())

    for questions, title2doc in [(train, corpus.train_title_to_document), (dev, corpus.dev_title_to_document)]:
        print(f"Starting encoding of {'train' if questions == train else 'dev'}")
        # eliminating distractors not from original squad
        title2max = {key: max(x.paragraph.par_id for x in group) for key, group in
                     itertools.groupby(sorted(questions, key=lambda x: x.paragraph.doc_title),
                                       key=lambda x: x.paragraph.doc_title)}
        for title in title2max:
            title2doc[title].paragraphs = title2doc[title].paragraphs[:title2max[title] + 1]

        for title in tqdm(title2max):
            np.savez_compressed(get_filename(questions == train, title),
                                **{str(k): v for k, v in encode_document(encoder, title2doc[title]).items()})


def load_all_squad():
    """ Loads all squad representations to one numpy array, and returns also index maps """
    par2ids = {}
    reps = []
    total_sentences = 0
    for dirname, _, filenames in os.walk(config.SQUAD_ENCODINGS):
        if dirname.split('/')[-1] not in ['train', 'dev']:
            continue
        for f in tqdm(filenames):
            if not f.endswith('.npz'):
                continue
            doc_reps = np.load(join(dirname, f))
            for pid, rep in doc_reps.items():
                par2ids[f"{f[:-4]}_{pid}"] = list(range(total_sentences, total_sentences + len(rep)))
                reps.append(rep)
                total_sentences += len(rep)
    return np.concatenate(reps, axis=0), par2ids


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Encode all SQuAD data')
    parser.add_argument('model', help='model directory to use for encoding')
    args = parser.parse_args()

    encode_all_squad(args.model)
