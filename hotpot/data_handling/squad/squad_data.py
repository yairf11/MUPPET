import pickle
from typing import List, Set

from os.path import join, exists, isfile, isdir

from os import makedirs, listdir

from hotpot.config import CORPUS_DIR
from hotpot.configurable import Configurable
from hotpot.data_handling.data import RelevanceQuestion
from hotpot.data_handling.word_vectors import load_word_vectors
from hotpot.utils import ResourceLoader

""" Squad data. For now, leaving out answer spans. When we want to predict answers, we will deal with it."""


class SquadParagraph(object):
    def __init__(self, doc_title: str, par_id: int, par_text: List[str], pickle_text=True):
        self.doc_title = doc_title
        self.par_id = par_id
        self.par_text = par_text
        self.pickle_text = pickle_text

    @property
    def num_tokens(self):
        return len(self.par_text)

    def get_paragraph_without_text_pickling(self):
        return SquadParagraph(self.doc_title, self.par_id, self.par_text, pickle_text=False)

    def __repr__(self) -> str:
        return f"Title: {self.doc_title}, Id: {self.par_id}\n" \
               f"Paragraph:\n" + ' '.join(self.par_text)

    def __getstate__(self):
        if not self.pickle_text:
            state = self.__dict__.copy()
            state['par_text'] = None
            return state
        return self.__dict__


class SquadDocument(object):
    def __init__(self, title: str, paragraphs: List[SquadParagraph]):
        self.title = title
        self.paragraphs = paragraphs
        self.id_to_par = self._build_id_paragraph_dict()

    def _build_id_paragraph_dict(self):
        return {x.par_id: x for x in self.paragraphs}

    def get_par(self, par_id) -> SquadParagraph:
        return self.id_to_par[par_id]

    def add_par(self, par: SquadParagraph):
        if par.par_id in self.id_to_par:
            raise ValueError("This paragraph id already exists in this document!")
        if par.doc_title != self.title:
            raise ValueError("Paragraph title not matching document title!")
        self.paragraphs.append(SquadParagraph(par.doc_title, par.par_id, par.par_text, pickle_text=True))
        self.id_to_par[par.par_id] = self.paragraphs[-1]

    def __repr__(self) -> str:
        return f"Title: {self.title}. Number of paragraphs: {len(self.paragraphs)}"


class SquadQuestion(object):
    """ Squad Question and paragraphs."""

    def __init__(self, question_id: str, question: List[str],
                 answers: Set[str], paragraph: SquadParagraph):
        self.question_id = question_id
        self.question = question
        self.answers = answers
        self.paragraph = paragraph  # .get_paragraph_without_text_pickling()

    def __repr__(self) -> str:
        return f"{self.question_id}: {' '.join(self.question)}\nAnswer(s): {self.answers}\n" \
               f"Paragraph:\n" + ' '.join(self.paragraph.par_text)


class SquadQuestionWithDistractors(SquadQuestion):
    def __init__(self, question_id: str, question: List[str],
                 answers: Set[str], paragraph: SquadParagraph,
                 distractors: List[SquadParagraph]):
        super().__init__(question_id, question, answers, paragraph)
        # self.distractors = [x.get_paragraph_without_text_pickling() for x in distractors]
        self.distractors = distractors

    def add_distractors(self, paragraphs: List[SquadParagraph]):
        """ Doesn't add duplicates """
        for paragraph in paragraphs:
            if any((x.par_id == paragraph.par_id and x.doc_title == paragraph.doc_title) for x in self.distractors):
                continue
            # self.distractors.append(paragraph.get_paragraph_without_text_pickling())
            self.distractors.append(paragraph)


def squad_question_to_relevance_question(squad_question: SquadQuestionWithDistractors) -> RelevanceQuestion:
    return RelevanceQuestion(dataset_name='squad',
                             question_id=squad_question.question_id,
                             question_tokens=squad_question.question,
                             supporting_facts=[squad_question.paragraph.par_text],
                             distractors=[x.par_text for x in squad_question.distractors])


class SquadRelevanceCorpus(Configurable):
    TRAIN_DOC_FILE = "train_documents.pkl"
    TRAIN_FILE = "train_questions.pkl"
    DEV_DOC_FILE = "dev_documents.pkl"
    DEV_FILE = "dev_questions.pkl"
    NAME = "squad"

    VOCAB_FILE = "squad_vocab.txt"
    WORD_VEC_SUFFIX = "_pruned"

    @staticmethod
    def make_corpus(train_documents: List[SquadDocument],
                    train: List[SquadQuestionWithDistractors],
                    dev_documents: List[SquadDocument],
                    dev: List[SquadQuestionWithDistractors]):
        dir = join(CORPUS_DIR, SquadRelevanceCorpus.NAME)
        # if isfile(dir) or (exists(dir) and len(listdir(dir))) > 0:
        #     raise ValueError("Directory %s already exists and is non-empty" % dir)
        if not exists(dir):
            makedirs(dir)

        train_document_dict = {doc.title: doc for doc in train_documents}
        if len(train_document_dict) != len(train_documents):
            raise ValueError("different train documents have the same title!")

        dev_document_dict = {doc.title: doc for doc in dev_documents}
        if len(dev_document_dict) != len(dev_documents):
            raise ValueError("different dev documents have the same title!")

        for name, data in [(SquadRelevanceCorpus.TRAIN_FILE, train), (SquadRelevanceCorpus.DEV_FILE, dev),
                           (SquadRelevanceCorpus.TRAIN_DOC_FILE, train_document_dict),
                           (SquadRelevanceCorpus.DEV_DOC_FILE, dev_document_dict)]:
            if data is not None:
                with open(join(dir, name), 'wb') as f:
                    pickle.dump(data, f)

    def __init__(self):
        dir = join(CORPUS_DIR, self.NAME)
        if not exists(dir) or not isdir(dir):
            raise ValueError("No directory %s, corpus not built yet?" % dir)
        self.dir = dir
        self.train_title_to_document = None
        self.dev_title_to_document = None

    @property
    def evidence(self):
        return None

    def get_vocab_file(self):
        self.get_vocab()
        return join(self.dir, self.VOCAB_FILE)

    def get_vocab(self):
        """ get all-lower cased unique words for this corpus, includes train/dev/test files """
        voc_file = join(self.dir, self.VOCAB_FILE)
        if exists(voc_file):
            with open(voc_file, "r") as f:
                return [x.rstrip() for x in f]
        else:
            voc = set()
            for fn in [self.get_train, self.get_dev, self.get_test]:
                for question in fn():
                    voc.update(x.lower() for x in question.question)
                    for para in (question.distractors + [question.paragraph]):
                        voc.update(x.lower() for x in para.par_text)
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

    def _load_document_dict(self, train: bool):
        if train:
            if self.train_title_to_document is None:
                self.train_title_to_document = self._load(join(self.dir, self.TRAIN_DOC_FILE))
        else:
            if self.dev_title_to_document is None:
                self.dev_title_to_document = self._load(join(self.dir, self.DEV_DOC_FILE))

    def _insert_text_to_paragraph(self, paragraph: SquadParagraph, train: bool):
        title_to_doc = self.train_title_to_document if train else self.dev_title_to_document
        paragraph.par_text = title_to_doc[paragraph.doc_title].get_par(paragraph.par_id).par_text
        paragraph.pickle_text = True  # So that there will be no problems later

    def _insert_text_to_question(self, question: SquadQuestionWithDistractors, train: bool):
        for par in [question.paragraph] + question.distractors:
            self._insert_text_to_paragraph(par, train)

    def _populate_questions(self, questions: List[SquadQuestionWithDistractors], train: bool):
        self._load_document_dict(train)
        for q in questions:
            self._insert_text_to_question(q, train)

    def get_train(self) -> List[SquadQuestionWithDistractors]:
        questions = self._load(join(self.dir, self.TRAIN_FILE))
        self._populate_questions(questions, train=True)
        return questions

    def get_dev(self) -> List[SquadQuestionWithDistractors]:
        questions = self._load(join(self.dir, self.DEV_FILE))
        self._populate_questions(questions, train=False)
        return questions

    def get_test(self) -> List[SquadQuestionWithDistractors]:
        return []

    def _load(self, file):
        if not exists(file):
            return []
        with open(file, "rb") as f:
            return pickle.load(f)

    def __getstate__(self):
        state = self.__dict__.copy()
        state['train_title_to_document'] = None
        state['dev_title_to_document'] = None
        return state

    def __setstate__(self, state):
        self.__dict__ = state
