from os.path import join, expanduser, dirname

"""
Global config options
"""

GLOBAL_DATA_DIR = join(expanduser("~"), "data")
LOCAL_DATA_DIR = join(dirname(dirname(__file__)), "data")

VEC_DIR = join(GLOBAL_DATA_DIR, 'glove')

DOC_DB = join(LOCAL_DATA_DIR, "db/wiki_hotpot_sentences_v1.1.db")
FULL_DOC_DB = join(LOCAL_DATA_DIR, "db/wiki_full.db")
TFIDF_FILE = join(LOCAL_DATA_DIR, "db/wiki_hotpot_sentences_v1.1-tfidf-ngram=2-hash=16777216-tokenizer=charoffset_.npz")
WIKIPEDIA_FILES = join(LOCAL_DATA_DIR, "wikipedia/enwiki-20171001-pages-meta-current-withlinks-processed")
DRQA_DOC_DB = join(LOCAL_DATA_DIR, "db/drqa-docs.db")
DRQA_RANKER = join(LOCAL_DATA_DIR, "db/drqa-docs-tfidf-ngram=2-hash=16777216-tokenizer=simple.npz")

CORPUS_DIR = LOCAL_DATA_DIR

HOTPOT_TRAIN_FILE = join(LOCAL_DATA_DIR, 'hotpot', "hotpot_train_v1.1.json")
HOTPOT_DEV_DISTRACTOR_FILE = join(LOCAL_DATA_DIR, 'hotpot', "hotpot_dev_distractor_v1.json")
HOTPOT_DEV_FULL_WIKI_FILE = join(LOCAL_DATA_DIR, 'hotpot', "hotpot_dev_fullwiki_v1.json")
HOTPOT_TEST_FULL_WIKI_FILE = join(LOCAL_DATA_DIR, 'hotpot', "hotpot_test_fullwiki_v1.json")
HOTPOT_DATASET_DICT = {'train': HOTPOT_TRAIN_FILE, 'dev': HOTPOT_DEV_DISTRACTOR_FILE,
                       'dev_full': HOTPOT_DEV_FULL_WIKI_FILE}

SQUAD_TRAIN_FILE = join(LOCAL_DATA_DIR, "squad/train-v1.1.json")
SQUAD_DEV_FILE = join(LOCAL_DATA_DIR, "squad/dev-v1.1.json")
SQUAD_DATASET_DICT = {'train': SQUAD_TRAIN_FILE, 'dev': SQUAD_DEV_FILE}
SQUAD_ENCODINGS = join(LOCAL_DATA_DIR, 'paragraph_encodings', 'squad')

LM_DIR = join(expanduser("~"), "data", "lm")
SQUAD_ELMO_VOCAB = join(LOCAL_DATA_DIR, 'elmo', 'squad', 'vocab.txt')
SQUAD_ELMO_EMBEDDINGS = join(LOCAL_DATA_DIR, 'elmo', 'squad', 'original_embeddings.hdf5')
SQUAD_ELMO_FINETUNED_EMBEDDINGS = join(LOCAL_DATA_DIR, 'elmo', 'squad', 'pretrained_embeddings.hdf5')
SQUAD_ELMO_FINETUNED_OPTIONS = join(LOCAL_DATA_DIR, 'elmo', 'squad', 'elmo_2x4096_512_2048cnn_2xhighway_5.5B_options.json')
SQUAD_ELMO_FINETUNED_WEIGHTS = join(LM_DIR, 'squad-context-concat-skip',
                          'squad_context_concat_lm_2x4096_512_2048cnn_2xhighway_skip.hdf5')
SQUAD_ELMO_FINETUNED_WIKI_EMBEDDINGS = join(LOCAL_DATA_DIR, 'elmo', 'squad',
                                            'squad_finetuned_wiki_vocab_embeddings.hdf5')
HOTPOT_ELMO_VOCAB = join(LOCAL_DATA_DIR, 'elmo', 'hotpot', 'vocab.txt')
HOTPOT_ELMO_EMBEDDINGS = join(LOCAL_DATA_DIR, 'elmo', 'hotpot', 'embeddings.hdf5')
HOTPOT_ELMO_OPTIONS = "elmo_2x4096_512_2048cnn_2xhighway_5.5B_options_hotpot.json"
HOTPOT_WIKI_ELMO_VOCAB = join(LOCAL_DATA_DIR, 'elmo', 'hotpot', 'wiki_vocab_word-th-5_title-th-1.txt')
HOTPOT_WIKI_ELMO_EMBEDDINGS = join(LOCAL_DATA_DIR, 'elmo', 'hotpot', 'wiki-vocab-5-1_embeddings.hdf5')
HOTPOT_WIKI_ELMO_OPTIONS = join(LOCAL_DATA_DIR, 'elmo', 'hotpot',  "wiki-vocab-5-1_5.5B_options.json")

WIKI_MOVIES_TEST_FILE = join(LOCAL_DATA_DIR, 'wiki_movies', 'WikiMovies-test.txt')
WIKI_MOVIES_ENTITIES = join(LOCAL_DATA_DIR, 'wiki_movies', 'WikiMovies-entities.txt')

WEB_QUESTIONS_TEST_FILE = join(LOCAL_DATA_DIR, 'web_questions', 'WebQuestions-test.txt')
WEB_QUESTIONS_ENTITIES = join(LOCAL_DATA_DIR, 'web_questions', 'freebase-entities.txt')

CURATED_TREC_TEST_FILE = join(LOCAL_DATA_DIR, 'curated_trec', 'CuratedTrec-test.txt')
