import pickle
from os import listdir, makedirs
from typing import List, Tuple, Union, Optional
import json

from os.path import join, exists, isfile, isdir

from hotpot import config
from hotpot.configurable import Configurable
from hotpot.data_handling.data import RelevanceQuestion
from hotpot.data_handling.word_vectors import load_word_vectors
from hotpot.utils import flatten_iterable, ResourceLoader


class HotpotParagraph(object):
    """ Paragraph with its title and sentence information """

    def __init__(self, title: str, sentences: List[List[str]]):
        self.title = title
        self.sentences = sentences

    @property
    def num_tokens(self):
        return len(flatten_iterable(self.sentences))

    def __repr__(self) -> str:
        return f"{self.title}: {' '.join(flatten_iterable(self.sentences))}"


class HotpotGoldParagraph(HotpotParagraph):
    """ Paragraph that is a gold paragraph to a question """

    def __init__(self, title: str, sentences: List[List[str]],
                 question_id: str, supporting_sentence_ids: List[int]):
        super().__init__(title, sentences)
        self.question_id = question_id
        self.supporting_sentence_ids = supporting_sentence_ids

    def repr_supporting_facts(self) -> str:
        return f"{self.title}: {' '.join(flatten_iterable([self.sentences[i] for i in self.supporting_sentence_ids]))}"


class HotpotQuestion(object):
    """ Question with its answer and supporting facts"""

    def __init__(self, question_id: str, question_tokens: List[str],
                 answer: str, supporting_facts: List[HotpotGoldParagraph],
                 distractors: List[HotpotParagraph], q_type: str, level: str,
                 gold_scores: List[float], distractor_scores: List[float],
                 gold_spans: List[Tuple[int, int]], distractor_spans: List[Tuple[int, int]]):
        self.question_id = question_id
        self.question_tokens = question_tokens
        self.answer = answer
        self.supporting_facts = supporting_facts
        self.distractors = distractors
        self.q_type = q_type
        self.level = level
        self.gold_scores = gold_scores
        self.distractor_scores = distractor_scores
        self.gold_spans = gold_spans
        self.distractor_spans = distractor_spans

    def __repr__(self) -> str:
        return f"{self.question_id}: {' '.join(self.question_tokens)}\nAnswer: {self.answer}\n" \
               f"Supporting Facts:\n" + '\n'.join([p.repr_supporting_facts() for p in self.supporting_facts])

    def __setstate__(self, state):
        if 'gold_scores' not in state:
            state['gold_scores'] = None
        if 'distractor_scores' not in state:
            state['distractor_scores'] = None
        if 'gold_spans' not in state:
            state['gold_spans'] = None
            state['distractor_spans'] = None
        self.__dict__ = state


def hotpot_question_to_relevance_question(hotpot_question: HotpotQuestion) -> RelevanceQuestion:
    return RelevanceQuestion(dataset_name='hotpot',
                             question_id=hotpot_question.question_id,
                             question_tokens=hotpot_question.question_tokens,
                             supporting_facts=[flatten_iterable(x.sentences) for x in hotpot_question.supporting_facts],
                             distractors=[flatten_iterable(x.sentences) for x in hotpot_question.distractors])


class HotpotQuestions(Configurable):
    TRAIN_FILE = "train_questions.pkl"
    DEV_FILE = "dev_questions.pkl"

    NAME = "hotpot"

    VOCAB_FILE = "hotpot_vocab.txt"
    WORD_VEC_SUFFIX = "_pruned"

    @staticmethod
    def make_corpus(train: List[HotpotQuestion],
                    dev: List[HotpotQuestion]):
        dir = join(config.CORPUS_DIR, HotpotQuestions.NAME)
        # if isfile(dir) or (exists(dir) and len(listdir(dir))) > 0:
        #     raise ValueError("Directory %s already exists and is non-empty" % dir)
        if not exists(dir):
            makedirs(dir)

        for name, data in [(HotpotQuestions.TRAIN_FILE, train), (HotpotQuestions.DEV_FILE, dev)]:
            if data is not None:
                with open(join(dir, name), 'wb') as f:
                    pickle.dump(data, f)

    def __init__(self):
        dir = join(config.CORPUS_DIR, self.NAME)
        if not exists(dir) or not isdir(dir):
            raise ValueError("No directory %s, corpus not built yet?" % dir)
        self.dir = dir

    @property
    def evidence(self):
        return None

    def get_vocab_file(self):
        if not exists(self.dir):
            self.dir = join(config.CORPUS_DIR, self.NAME)
        self.get_vocab()
        return join(self.dir, self.VOCAB_FILE)

    def get_vocab(self):
        """ get all-lower cased unique words for this corpus, includes train/dev/test files """
        if not exists(self.dir):
            self.dir = join(config.CORPUS_DIR, self.NAME)
        voc_file = join(self.dir, self.VOCAB_FILE)
        if exists(voc_file):
            with open(voc_file, "r") as f:
                return [x.rstrip() for x in f]
        else:
            voc = set()
            for fn in [self.get_train, self.get_dev, self.get_test]:
                for question in fn():
                    voc.update(x.lower() for x in question.question_tokens)
                    for para in (question.distractors + question.supporting_facts):
                        voc.update(x.lower() for x in flatten_iterable(para.sentences))
            voc_list = sorted(list(voc))
            with open(voc_file, "w") as f:
                for word in voc_list:
                    f.write(word)
                    f.write("\n")
            return voc_list

    def get_pruned_word_vecs(self, word_vec_name, voc=None):
        """
        Loads word vectors that have been pruned to the case-insensitive vocab of this corpus.
        WARNING: this includes dev words

        This exists since loading word-vecs each time we startup can be a big pain, so
        we cache the pruned vecs on-disk as a .npy file we can re-load quickly.
        """
        if not exists(self.dir):
            self.dir = join(config.CORPUS_DIR, self.NAME)
        vec_file = join(self.dir, word_vec_name + self.WORD_VEC_SUFFIX + ".npy")
        if isfile(vec_file):
            print("Loading word vec %s for %s from cache" % (word_vec_name, self.name))
            with open(vec_file, "rb") as f:
                return pickle.load(f)
        else:
            print("Building pruned word vec %s for %s" % (self.name, word_vec_name))
            voc = self.get_vocab()
            vecs = load_word_vectors(word_vec_name, voc)
            with open(vec_file, "wb") as f:
                pickle.dump(vecs, f)
            return vecs

    def get_resource_loader(self):
        return ResourceLoader(self.get_pruned_word_vecs)

    def get_train(self) -> List[HotpotQuestion]:
        if not exists(self.dir):
            self.dir = join(config.CORPUS_DIR, self.NAME)
        return self._load(join(self.dir, self.TRAIN_FILE))

    def get_dev(self) -> List[HotpotQuestion]:
        if not exists(self.dir):
            self.dir = join(config.CORPUS_DIR, self.NAME)
        return self._load(join(self.dir, self.DEV_FILE))

    def get_test(self) -> List[HotpotQuestion]:
        return []

    def _load(self, file) -> List[HotpotQuestion]:
        if not exists(file):
            return []
        with open(file, "rb") as f:
            return pickle.load(f)
