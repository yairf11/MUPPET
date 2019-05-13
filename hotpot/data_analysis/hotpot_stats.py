import json
from typing import Optional

from hotpot import config


def load_dataset(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)


class HotpotStats(object):
    """Class for analysing various stats of the dataset"""

    def __init__(self, dataset: Optional[str] = None):
        self.data = config.HOTPOT_DATASET_DICT[dataset]

        self.train = load_dataset(self.data)

        self.average_sentences = None
        self.paragraph_lens = None
        self.has_answers = None
        self.has_gold_paragraphs = None

