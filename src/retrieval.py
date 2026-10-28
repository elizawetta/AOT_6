import re
from dataclasses import dataclass

from rank_bm25 import BM25Okapi

from .schemas import Claim


def tokenize(text: str) -> list[str]:
    return re.findall(r'[a-zа-яё0-9_]+', text.lower().replace('ё', 'е'))


@dataclass(frozen=True)
class SearchHit:
    claim: Claim
    score: float


class ClaimRetriever:
    def __init__(self, claims: list[Claim]):
        self.claims = sorted(claims, key=lambda c: c.claim_id)
        self.index = BM25Okapi([tokenize(c.text) for c in self.claims]) if self.claims else None

    def search(self, question: str, limit: int = 15) -> list[SearchHit]:
        if isinstance(limit, bool) or not isinstance(limit, int) or not 0 <= limit <= 15:
            raise ValueError('limit must be an integer between 0 and 15')
        if not self.index or not limit:
            return []
        scores = self.index.get_scores(tokenize(question))
        hits = [SearchHit(c, float(s)) for c, s in zip(self.claims, scores)]
        return sorted(hits, key=lambda h: (-h.score, h.claim.claim_id))[:limit]
