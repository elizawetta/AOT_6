import json
from dataclasses import dataclass
from typing import Literal

from .padic import group_claims
from .retrieval import ClaimRetriever, SearchHit
from .schemas import Claim, Dataset


@dataclass(frozen=True)
class BuiltContext:
    mode: str
    text: str
    claim_ids: tuple[str, ...]
    document_ids: tuple[str, ...]
    retrieval_scores: tuple[tuple[str, float], ...]


def claim_payload(claim: Claim, dataset: Dataset) -> dict:
    meta = dataset.metadata[claim.claim_id]
    doc = next(d for d in dataset.documents if d.document_id == claim.document_id)
    return dict(claim_id=claim.claim_id, text=claim.text, document_id=claim.document_id, evidence=claim.evidence, status=claim.status, source_type=doc.source_type, occurred_at=meta.occurred_at.isoformat() if meta.occurred_at else None, padic_path=list(meta.path))


def build_context(dataset: Dataset, question: str, mode: Literal['A', 'B', 'C'], hits: list[SearchHit] | None = None) -> BuiltContext:
    if mode not in {'A', 'B', 'C'}:
        raise ValueError('Unknown context mode')
    if mode == 'A':
        docs = [d.model_dump(mode='json') for d in dataset.documents]
        claims = sorted(dataset.claims, key=lambda c: c.claim_id)
        content = 'ДОКУМЕНТЫ\n' + '\n'.join(json.dumps(d, ensure_ascii=False) for d in docs)
        content += '\nРЕЕСТР УТВЕРЖДЕНИЙ ДЛЯ ССЫЛОК\n' + '\n'.join(json.dumps(claim_payload(c, dataset), ensure_ascii=False) for c in claims)
        return BuiltContext(mode, content, tuple(c.claim_id for c in claims), tuple(d.document_id for d in dataset.documents), ())
    selected = hits if hits is not None else ClaimRetriever(dataset.claims).search(question)
    if len(selected) > 15 or len({h.claim.claim_id for h in selected}) != len(selected):
        raise ValueError('Retrieval must return at most 15 unique claims')
    known = {c.claim_id: c for c in dataset.claims}
    if any(h.claim.claim_id not in known or known[h.claim.claim_id] != h.claim for h in selected):
        raise ValueError('Retrieval contains unknown or stale claims')
    claims = [h.claim for h in selected]
    render = lambda c: json.dumps(claim_payload(c, dataset), ensure_ascii=False)
    if mode == 'B':
        content = 'УТВЕРЖДЕНИЯ\n' + '\n'.join(render(c) for c in claims)
    else:
        sections = []
        for path, branch in group_claims(claims, dataset.metadata):
            sections.append('ВЕТВЬ ' + json.dumps(list(path)) + '\n' + '\n'.join(render(c) for c in branch))
        content = '\n\n'.join(sections)
    return BuiltContext(mode, content, tuple(c.claim_id for c in claims), tuple(d.document_id for d in dataset.documents), tuple((h.claim.claim_id, h.score) for h in selected))


def build_context_pair(dataset: Dataset, question: str) -> tuple[BuiltContext, BuiltContext]:
    hits = ClaimRetriever(dataset.claims).search(question)
    return build_context(dataset, question, 'B', hits), build_context(dataset, question, 'C', hits)
