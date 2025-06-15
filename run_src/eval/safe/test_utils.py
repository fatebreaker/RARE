from utils import RetrievalSystem
retrieval_system = RetrievalSystem("BM25", "MedCorp", "/home/hieutran/MedRAG/corpus")
retrieved_snippets, scores = retrieval_system.retrieve("Obama", k=3)
import ipdb; ipdb.set_trace()