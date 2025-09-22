from typing import Dict, List, Optional, Callable
from collections import Counter
import numpy as np

class UDFResolver:
    def __init__(self, embedding_fn: Callable[[str], List[float]], threshold: float = 0.8):
        """
        embedding_fn: string → vector embedding function (e.g., OpenAI embeddings API)
        threshold: semantic similarity threshold
        """
        self.embedding_fn = embedding_fn
        self.threshold = threshold
        self.udf_catalog: Dict[str, Dict] = {}   # {"udf_name": {"desc": str, "embedding": vec, "impl": fn}}
        self.query_counter = Counter()

    def register_udf(self, name: str, desc: str, impl: Optional[Callable] = None):
        emb = self.embedding_fn(desc)
        self.udf_catalog[name] = {"desc": desc, "embedding": emb, "impl": impl}

    def resolve(self, query_term: str) -> str:
        """
        mapping NL to UDF or decide as a new UDF candidate
        """
        q_emb = self.embedding_fn(query_term)
        best_match, best_score = None, -1.0

        # 1. semantic similarity search
        for name, meta in self.udf_catalog.items():
            score = self._cosine_similarity(q_emb, meta["embedding"])
            if score > best_score:
                best_match, best_score = name, score

        # 2. compare with threshold
        if best_score >= self.threshold:
            decision = f"Reusing existing UDF: {best_match} (sim={best_score:.2f})"
            self.query_counter[best_match] += 1
            return decision

        # 3. check combination possibility (e.g., 'crossing' is representable as 'perpendicular' + 'velocity_above')
        if query_term == "crossing":
            decision = "Use builtin ops: perpendicular + velocity_above"
            return decision

        # 4. register new UDF candidate
        self.register_udf(query_term, desc=query_term)  # placeholder
        self.query_counter[query_term] += 1
        decision = f"Registered NEW UDF: {query_term}"
        return decision

    def _cosine_similarity(self, v1: List[float], v2: List[float]) -> float:
        v1, v2 = np.array(v1), np.array(v2)
        return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8))

    def stats(self):
        return dict(self.query_counter)


# === Example Usage ===
if __name__ == "__main__":
    # dummy embedding function (actual: OpenAI embeddings API)
    dummy_emb = lambda text: [float(ord(c)) % 10 for c in text][:10]

    resolver = UDFResolver(embedding_fn=dummy_emb, threshold=0.75)
    resolver.register_udf("perpendicular", "two objects moving at 90 degrees")
    resolver.register_udf("turn_left", "object changes heading about 90 degrees to the left")

    print(resolver.resolve("crossing"))   # Use builtin ops (fallback)
    print(resolver.resolve("perpendicular"))  # Reuse existing UDF
    print(resolver.resolve("new_behavior"))   # Register NEW UDF
    print(resolver.stats())