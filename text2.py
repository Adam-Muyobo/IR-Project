"""
202104060 Muyobo, AM Computer Science
202102742 Mndolo, BK Computer Science
"""
import numpy as np
from collections import Counter

class Document:
    def __init__(self, text):
        self.text = text
        self.words = text.split()
        self.nwords = len(self.words)
        self.word_count = Counter(self.words)

class IRSystem:
    def __init__(self, documents):
        self.documents = documents

    def score(self, word):
        """Calculate the score of a given word in all documents."""
        scores = []
        for doc in self.documents:
            freq = doc.word_count[word]
            score = np.log(1 + freq) / np.log(1 + doc.nwords)
            scores.append(score)
        return scores

    def get_relevant_documents(self, query, expert_relevant_indexes):
        """Return the list of relevant document indexes based on expert judgment."""
        relevant_docs = []
        for index in expert_relevant_indexes:
            if index < len(self.documents):
                relevant_docs.append(index)
        return relevant_docs

    def retrieve_documents(self, query):
        """Retrieve documents that contain terms from the query."""
        query_terms = query.split()
        retrieved_docs = []

        for i, doc in enumerate(self.documents):
            # Check if any query term is in the document's words
            if any(term in doc.word_count for term in query_terms):
                retrieved_docs.append(i)

        return retrieved_docs

    def evaluate(self, relevant_docs, retrieved_docs):
        """Calculate Precision, Recall, FP, FN, and F-measure."""
        TP = len(set(relevant_docs) & set(retrieved_docs))  # True Positives
        FP = len(set(retrieved_docs) - set(relevant_docs))  # False Positives
        FN = len(set(relevant_docs) - set(retrieved_docs))  # False Negatives

        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f_measure = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        return precision, recall, FP, FN, f_measure

# Example usage
if __name__ == "__main__":
    # Sample documents
    documents = [
        Document("text of document one discussing various topics"),
        Document("another text of document two with different content"),
        Document("document three contains relevant information about text"),
    ]

    # Create an IR system with the documents
    ir_system = IRSystem(documents)

    # Score the word "text"
    word_to_score = "text"
    scores = ir_system.score(word_to_score)
    print("Scores for word '{}':".format(word_to_score), scores)

    # Define expert judgments for relevant documents
    expert_relevant_indexes = [0, 2]  # Let's say documents 0 and 2 are relevant based on expert judgment

    # Get relevant documents based on the expert judgments
    relevant_docs = ir_system.get_relevant_documents("text", expert_relevant_indexes)

    # Retrieve documents based on a query
    query = "text"
    retrieved_docs = ir_system.retrieve_documents(query)
    print("Retrieved documents:", retrieved_docs)

    # Evaluate the system
    precision, recall, fp, fn, f_measure = ir_system.evaluate(relevant_docs, retrieved_docs)
    print(f"Precision: {precision:.2f}, Recall: {recall:.2f}, FP: {fp}, FN: {fn}, F-measure: {f_measure:.2f}")
