from langchain_community.llms import Ollama
import bm25s
import heapq
# Hybrid Search - Reciprocal Rank Fusion
class ReciprocalRankFusion:
    def __init__(self, k: float = 60.0):
        self.k = k

    def fuse(self, ranked_lists, top_n: int = 3):
        item_ranks = {}
        for lst in ranked_lists:
            for rank, (item, score) in enumerate(lst, start=1):
                if item not in item_ranks:
                    item_ranks[item] = [len(ranked_lists) + 1] * len(ranked_lists)
                item_ranks[item][ranked_lists.index(lst)] = rank

        fused_scores = []
        for item, ranks in item_ranks.items():
            fused_score = sum(1 / (rank + self.k) for rank in ranks)
            heapq.heappush(fused_scores, (-fused_score, item))

        # Return top-n results
        return [(item, -score) for score, item in sorted(fused_scores, reverse=True)[:top_n]]
    
def bm25s_search(query, retriever, stemmer, corpus):
    query_tokens = bm25s.tokenize(query, stemmer=stemmer)
    
    # Adjust k based on the number of query tokens
    k = min(2, len(query_tokens[0]))  # Ensure k is <= number of tokens
    
    # Retrieve top-k results
    results,scores = retriever.retrieve(query_tokens, corpus=corpus, k=k)
    
    # Combine the results with the scores
    scored_results = [(results[i][i]['text'], scores[i][i]) for i in range(len(results))]
    
    return scored_results


# Qdrant similarity search returning scores
def qdrant_search(query, qdrant):
    search_results = qdrant.similarity_search_with_score(query)[0:3]
    return [(doc.page_content, score) for doc, score in search_results]

from pydantic import BaseModel

# KO class definition
class KO(BaseModel):
    short_description: str
    symptoms: str
    long_description: str
    causes: str
    resolution_note: str
