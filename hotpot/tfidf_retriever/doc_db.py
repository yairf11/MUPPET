# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Documents, in a sqlite database."""

import sqlite3
from . import utils
from hotpot import config
from hotpot.tfidf_retriever.utils import deserialize_object


class OldDocDB(object):
    """Sqlite backed document storage.

    This is the old version - should be destroyed shortly. For now, kept for compatibility reasons.
    """

    def __init__(self, db_path=None):
        self.path = db_path or config.DOC_DB
        self.connection = sqlite3.connect(self.path, check_same_thread=False)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def path(self):
        """Return the path to the file that backs this database."""
        return self.path

    def close(self):
        """Close the connection to the database."""
        self.connection.close()

    def get_doc_ids(self):
        """Fetch all ids of docs stored in the db."""
        cursor = self.connection.cursor()
        cursor.execute("SELECT id FROM documents")
        results = [r[0] for r in cursor.fetchall()]
        cursor.close()
        return results

    def get_doc_titles(self):
        """Fetch all titles of docs stored in the db."""
        cursor = self.connection.cursor()
        cursor.execute("SELECT title FROM documents")
        results = [r[0] for r in cursor.fetchall()]
        cursor.close()
        return results

    def get_doc_text_offsets_from_id(self, doc_id):
        """Fetch the raw text and offsets of the doc for 'doc_id'."""
        cursor = self.connection.cursor()
        cursor.execute(
            "SELECT text, charoffset FROM documents WHERE id = ?",
            (doc_id,)
        )
        result = cursor.fetchone()
        cursor.close()
        return result if result is None else deserialize_object(result[0]), deserialize_object(result[1])

    def get_doc_text_offsets_from_title(self, doc_title):
        """Fetch the raw text and offsets of the doc for 'doc_title'."""
        cursor = self.connection.cursor()
        cursor.execute(
            "SELECT text, charoffset FROM documents WHERE title = ?",
            (doc_title,)
        )
        result = cursor.fetchone()
        cursor.close()
        return result if result is None else deserialize_object(result[0]), deserialize_object(result[1])

    def get_doc_text(self, doc_id):
        """Fetch the raw text of the doc for 'doc_id'."""
        cursor = self.connection.cursor()
        cursor.execute(
            "SELECT text FROM documents WHERE id = ?",
            (utils.normalize(doc_id),)
        )
        result = cursor.fetchone()
        cursor.close()
        return result if result is None else result[0]


class DocDB(object):
    """Sqlite backed document storage.

    stores a title of a document along with its sentences, represented as a list of strings. Each string is a sentence.
    """

    def __init__(self, db_path=None, full_docs=False):
        self.path = db_path or config.DOC_DB
        self.full_docs = full_docs
        self.connection = sqlite3.connect(self.path, check_same_thread=False)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def path(self):
        """Return the path to the file that backs this database."""
        return self.path

    def close(self):
        """Close the connection to the database."""
        self.connection.close()

    def get_doc_titles(self):
        """Fetch all titles of docs stored in the db."""
        cursor = self.connection.cursor()
        cursor.execute("SELECT title FROM documents")
        results = [r[0] for r in cursor.fetchall()]
        cursor.close()
        return results

    def get_doc_sentences(self, doc_title):
        """Fetch the sentences of the doc for 'doc_title'."""
        if self.full_docs:
            raise ValueError('This DB is in full docs mode. Try `get_doc_paragraphs`')
        cursor = self.connection.cursor()
        cursor.execute(
            "SELECT sentences FROM documents WHERE title = ?",
            (doc_title,)
        )
        result = cursor.fetchone()
        cursor.close()
        if result is None:
            return None
        return deserialize_object(result[0])

    def get_doc_paragraphs(self, doc_title):
        """Fetch the paragraphs of the doc for 'doc_title'."""
        if not self.full_docs:
            raise ValueError('This DB is in Hotpot mode. Try `get_doc_sentences`')
        cursor = self.connection.cursor()
        cursor.execute(
            "SELECT paragraphs FROM documents WHERE title = ?",
            (doc_title,)
        )
        result = cursor.fetchone()
        cursor.close()
        if result is None:
            return None
        return deserialize_object(result[0])
