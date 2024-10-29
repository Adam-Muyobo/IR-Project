import heapq 
import os
import re
from collections import defaultdict
import numpy as np
from probabilistic_learning import CountingProbDist
from utils import hashabledict
from text import words, Document, IRSystem

class ExtendedIRSystem(IRSystem):
    """An extended Information Retrieval System that calculates BM25 scores."""

    def __init__(self, stopwords='the a of'):
        super().__init__(stopwords)
        self.doc_lengths = []  # Store the length of each document for BM25
        self.avg_doc_length = 0  # Average document length

    def index_document(self, text, url):
        """Index the text of a document and compute document length."""
        title = text[:text.index('\n')].strip()
        docwords = words(text)
        docid = len(self.documents)
        self.documents.append(Document(title, url, len(docwords)))
        
        self.doc_lengths.append(len(docwords))  # Store document length
        self.avg_doc_length = np.mean(self.doc_lengths)  # Update average length
        
        for word in docwords:
            if word not in self.stopwords:
                self.index[word][docid] += 1

    def score(self, word, docid):
        """Compute a BM25 score for this word on the document with this docid."""
        # BM25 parameters
        k1 = 1.5
        b = 0.75

        # Get term frequency (TF) and document frequency (DF)
        tf = self.index[word][docid]  # Term frequency in the document
        df = len(self.index[word])  # Document frequency for the term
        doc_length = self.doc_lengths[docid]  # Length of the document

        # BM25 formula
        numerator = tf * (k1 + 1)
        denominator = tf + k1 * (1 - b + b * (doc_length / self.avg_doc_length))
        score = (numerator / denominator) * np.log((len(self.documents) - df + 0.5) / (df + 0.5))

        return score

    def present_individual_scores(self, query_text, n=10):
        """Present the individual BM25 scores for the words in the query."""
        qwords = [w for w in words(query_text) if w not in self.stopwords]
        for word in qwords:
            if word in self.index:
                scores = [(docid, self.score(word, docid)) for docid in self.index[word]]
                print(f"Scores for word '{word}':")
                for docid, score in scores:
                    print(f"Document ID: {docid}, Score: {score:.4f}")
            else:
                print(f"Word '{word}' not found in index.")

# Example usage
if __name__ == "__main__":
    # Create an instance of ExtendedIRSystem
    ir_system = ExtendedIRSystem(stopwords='the a of')

    # Index some documents (example)
    ir_system.index_document("Document 1 content goes here.", "doc1")
    ir_system.index_document("Document 2 content is different.", "doc2")

    # Present individual BM25 scores for a query
    ir_system.present_individual_scores("content different")
