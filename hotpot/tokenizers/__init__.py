# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import os

DEFAULTS = {
    'corenlp_classpath': os.getenv('CLASSPATH')
}


def set_default(key, value):
    global DEFAULTS
    DEFAULTS[key] = value


from .corenlp_tokenizer import CoreNLPTokenizer
from .simple_tokenizer import SimpleTokenizer

# Spacy is optional
try:
    from .spacy_tokenizer import SpacyTokenizer
except ImportError:
    pass


def get_class(name):
    if name == 'spacy':
        return SpacyTokenizer
    if name == 'corenlp':
        return CoreNLPTokenizer
    if name == 'simple':
        return SimpleTokenizer

    raise RuntimeError('Invalid tokenizer: %s' % name)