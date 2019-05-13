from typing import List


class RelevanceQuestion(object):
    """
    A class for handling a question with relevant and non-relevant paragraphs.
    Meant to support any dataset, so minimal assumptions have been made on the structure of this type.

    Important: the gold paragraphs of this data-type are all-or-none - all must be present for them to be relevant.
    """
    def __init__(self, dataset_name: str, question_id: str, question_tokens: List[str],
                 supporting_facts: List[List[str]], distractors: List[List[str]]):
        if dataset_name not in ['squad', 'hotpot']:
            raise NotImplementedError()
        self.dataset_name = dataset_name
        self.question_id = question_id
        self.question_tokens = question_tokens
        self.supporting_facts = supporting_facts
        self.distractors = distractors