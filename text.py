import heapq
import os
import re
from collections import defaultdict
import numpy as np
import search
from probabilistic_learning import CountingProbDist
from utils import hashabledict

class IRSystem:
    """A simple Information Retrieval System with frequency-based scoring."""
    
    def __init__(self, stopwords='the a of'):
        """Create an IR System. Optionally specify stopwords."""
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
        if not qwords:
            return []
        docids = {docid for word in qwords if word in self.index for docid in self.index[word]}
        return heapq.nlargest(n, ((self.total_score(qwords, docid), docid) for docid in docids))

    def score(self, word, docid):
        """Compute a score based on the frequency of this word in the document."""
        return self.index[word][docid]

    def total_score(self, words, docid):
        """Compute the sum of the scores of these words on the document with this docid."""
        return sum(self.score(word, docid) for word in words if word in self.index)

    def present(self, results):
        """Present the results as a list."""
        for (score, docid) in results:
            doc = self.documents[docid]
            print("{:5.2}|{:25} | {}".format(100 * score, doc.url, doc.title[:45].expandtabs()))

    def present_results(self, query_text, n=10):
        """Get results for the query and present them."""
        self.present(self.query(query_text, n))

class Document:
    """Metadata for a document: title and url; maybe add others later."""

    def __init__(self, title, url, nwords):
        self.title = title
        self.url = url
        self.nwords = nwords

def words(text, reg=re.compile('[a-z0-9]+')):
    """Return a list of the words in text, ignoring punctuation and converting everything to lowercase."""
    return reg.findall(text.lower())
