"""
202104060 Muyobo, AM Computer Science
202102742 Mndolo, BK Computer Science
"""
import heapq
import os
import re
from collections import defaultdict

import numpy as np

from probabilistic_learning import CountingProbDist
from utils import hashabledict


class UnigramWordModel(CountingProbDist):
    """This is a discrete probability distribution over words."""
    def __init__(self, observations, default=0):
        super(UnigramWordModel, self).__init__(observations, default)

    def samples(self, n):
        """Return a string of n words, random according to the model."""
        return ' '.join(self.sample() for i in range(n))


class Document:
    """Metadata for a document: title and url."""
    def __init__(self, title, url, nwords):
        self.title = title
        self.url = url
        self.nwords = nwords


def words(text, reg=re.compile('[a-z0-9]+')):
    """Return a list of the words in text, ignoring punctuation."""
    return reg.findall(text.lower())


class IRSystem:
    """A very simple Information Retrieval System."""
    def __init__(self, stopwords='the a of'):
        self.index = defaultdict(lambda: defaultdict(int))
        self.stopwords = set(words(stopwords))
        self.documents = []

    def index_collection(self, filenames):
        """Index a whole collection of files."""
        prefix = os.path.dirname(__file__)
        for filename in filenames:
            self.index_document(open(filename).read(), os.path.relpath(filename, prefix))

    def index_document(self, text, url):
        """Index the text of a document."""
        title = text[:text.index('\n')].strip()
        docwords = words(text)
        docid = len(self.documents)
        self.documents.append(Document(title, url, len(docwords)))
        for word in docwords:
            if word not in self.stopwords:
                self.index[word][docid] += 1

    def query(self, query_text, n=10):
        """Return a list of n (score, docid) pairs for the best matches."""
        qwords = [w for w in words(query_text) if w not in self.stopwords]
        docids = set(docid for word in qwords if word in self.index for docid in self.index[word])
        return heapq.nlargest(n, ((self.total_score(qwords, docid), docid) for docid in docids))

    def score(self, word, docid):
        """Compute a score for this word on the document with this docid based on frequency."""
        return self.index[word][docid]

    def total_score(self, words, docid):
        """Compute the sum of the scores of these words on the document with this docid."""
        return sum(self.score(word, docid) for word in words)

    def present(self, results):
        """Present the results as a list."""
        for (score, docid) in results:
            doc = self.documents[docid]
            print("{:5.2f}|{:25} | {}".format(100 * score, doc.url, doc.title[:45].expandtabs()))

    def present_results(self, query_text, n=10):
        """Get results for the query and present them."""
        self.present(self.query(query_text, n))


class UnixConsultant(IRSystem):
    """A trivial IR system over a small collection of Unix man pages."""
    def __init__(self):
        IRSystem.__init__(self, stopwords="how do i the a of")

        import os
        aima_root = os.path.dirname(__file__)
        mandir = os.path.join(aima_root, 'aima-data/MAN/')
        man_files = [mandir + f for f in os.listdir(mandir) if f.endswith('.txt')]

        self.index_collection(man_files)


# Example usage
if __name__ == "__main__":
    ir_system = UnixConsultant()
    ir_system.present_results("gzip cat", n=5)
