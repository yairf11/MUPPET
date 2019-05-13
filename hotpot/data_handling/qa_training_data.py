from typing import List, Optional

import numpy as np

from hotpot.data_handling.dataset import QuestionAndParagraphs


class SpanQuestionAndParagraphs(QuestionAndParagraphs):
    """ A basic data point for span labeled data """

    def __init__(self, question: List[str], paragraphs: List[List[str]], spans: np.ndarray,
                 question_id: str, answer: str,
                 q_type: str='not_relevant', sentence_segments: Optional[List[List[int]]]=None,
                 supporting_facts: Optional[np.ndarray]=None):
        super().__init__(question, paragraphs, sentence_segments)
        self.spans = spans
        self.q_type = q_type
        self.question_id = question_id
        self.answer = answer
        self.supporting_facts = supporting_facts

