""" Classes that deal with the training data of classifying the correct gold-paragraph pair """
from typing import List, Optional

from hotpot.data_handling.dataset import QuestionAndParagraphs


class BinaryQuestionAndParagraphs(QuestionAndParagraphs):
    """ A basic data point for binary labeled data """

    def __init__(self, question: List[str], paragraphs: List[List[str]], label, num_distractors: int, question_id: str,
                 q_type: str='not_relevant', sentence_segments: Optional[List[List[int]]]=None):
        super().__init__(question, paragraphs, sentence_segments)
        self.label = label
        self.num_distractors = num_distractors
        self.q_type = q_type
        self.question_id = question_id


class IterativeQuestionAndParagraphs(QuestionAndParagraphs):
    """ A basic data point for iterative paragraph relevance """

    def __init__(self, question: List[str], paragraphs: List[List[str]], first_label, second_label,
                 question_id: str, q_type: str='not_relevant', sentence_segments: Optional[List[List[int]]]=None):
        super().__init__(question, paragraphs, sentence_segments)
        self.first_label = first_label
        self.second_label = second_label
        self.q_type = q_type
        self.question_id = question_id


def multiple_contexts_len(sample):
    return max(len(c) for c in sample.paragraphs)
