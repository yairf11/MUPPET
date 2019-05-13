"""Basic tokenizer that splits text into alpha-numeric tokens and
non-whitespace tokens by the offsets given to it
"""

import logging
from .tokenizer import Tokens, Tokenizer

logger = logging.getLogger(__name__)


class CharOffsetTokenizer(Tokenizer):
    def __init__(self, **kwargs):
        """
        Args:
            annotators: None or empty set (only tokenizes).
        """
        if len(kwargs.get('annotators', {})) > 0:
            logger.warning('%s only tokenizes! Skipping annotators: %s' %
                           (type(self).__name__, kwargs.get('annotators')))
        self.annotators = set()

    def tokenize(self, text, offsets):
        data = []
        offsets = [token_span for sentence in offsets for token_span in sentence]
        for i in range(len(offsets)):
            # Get text
            token = text[offsets[i][0]:offsets[i][1]]

            # Get whitespace
            span = offsets[i]
            start_ws = span[0]
            if i + 1 < len(offsets):
                end_ws = offsets[i + 1][0]
            else:
                end_ws = span[1]

            # Format data
            data.append((
                token,
                text[start_ws: end_ws],
                span,
            ))
        return Tokens(data, self.annotators)
