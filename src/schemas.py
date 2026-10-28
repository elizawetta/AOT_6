import json
from datetime import datetime
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

LogicalStatus = Literal['entailed', 'contradicted', 'both', 'insufficient']
ClaimStatus = Literal['direct', 'reported', 'inferred', 'disputed']
Digit = Annotated[int, Field(strict=True, ge=0, le=2)]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)


class Document(StrictModel):
    document_id: str = Field(pattern=r'^D\d+$')
    title: str
    source_type: str
    created_at: datetime
    author: str
    text: str = Field(min_length=1)


class Claim(StrictModel):
    claim_id: str = Field(pattern=r'^C\d+$')
    text: str = Field(min_length=1)
    document_id: str
    evidence: str = Field(min_length=1)
    status: ClaimStatus


class Question(StrictModel):
    question_id: str = Field(pattern=r'^Q\d+$')
    question: str
    gold_status: LogicalStatus
    gold_answer: str | None
    gold_evidence_ids: list[str]
    gold_counterevidence_ids: list[str]

    @model_validator(mode='after')
    def check_both(self):
        if self.gold_status == 'both' and not (self.gold_evidence_ids and self.gold_counterevidence_ids):
            raise ValueError('Both requires supporting and opposing evidence')
        return self


class ClaimMetadata(StrictModel):
    path: tuple[Digit, Digit, Digit]
    occurred_at: datetime | None
    rationale: str


class ModelAnswer(StrictModel):
    question_id: str
    status: LogicalStatus
    answer: str | None
    evidence_ids: list[str]
    counterevidence_ids: list[str]
    confidence: float = Field(ge=0, le=1, allow_inf_nan=False)
    explanation: str = Field(min_length=1)

    @model_validator(mode='after')
    def check_evidence(self):
        if self.status == 'both' and not (self.evidence_ids and self.counterevidence_ids):
            raise ValueError('Both requires supporting and opposing evidence')
        if len(set(self.evidence_ids)) != len(self.evidence_ids) or len(set(self.counterevidence_ids)) != len(self.counterevidence_ids):
            raise ValueError('Evidence lists must not contain duplicates')
        if set(self.evidence_ids) & set(self.counterevidence_ids):
            raise ValueError('Supporting and opposing evidence must be disjoint')
        return self


class Dataset(StrictModel):
    documents: list[Document]
    claims: list[Claim]
    questions: list[Question]
    metadata: dict[str, ClaimMetadata]

    @model_validator(mode='after')
    def check_references(self):
        for field, key in [(self.documents, 'document_id'), (self.claims, 'claim_id'), (self.questions, 'question_id')]:
            if len({getattr(x, key) for x in field}) != len(field):
                raise ValueError(f'Duplicate {key}')
        docs = {d.document_id: d for d in self.documents}
        ids = {c.claim_id for c in self.claims}
        if set(self.metadata) != ids:
            raise ValueError('Metadata must cover exactly the available claims')
        for claim in self.claims:
            if claim.document_id not in docs or claim.evidence not in docs[claim.document_id].text:
                raise ValueError(f'Invalid source evidence for {claim.claim_id}')
        for question in self.questions:
            if not set(question.gold_evidence_ids + question.gold_counterevidence_ids) <= ids:
                raise ValueError(f'Unknown gold evidence for {question.question_id}')
        return self


def load_jsonl(path: Path, schema):
    return [schema.model_validate_json(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def load_dataset(data_dir: str | Path, metadata_path: str | Path) -> Dataset:
    root = Path(data_dir)
    metadata = {cid: ClaimMetadata.model_validate_json(json.dumps(value)) for cid, value in json.loads(Path(metadata_path).read_text()).items()}
    return Dataset(documents=load_jsonl(root / 'documents.jsonl', Document), claims=load_jsonl(root / 'claims.jsonl', Claim), questions=load_jsonl(root / 'questions.jsonl', Question), metadata=metadata)


def validate_answer(raw: str, question_id: str, allowed_claim_ids: set[str]) -> ModelAnswer:
    answer = ModelAnswer.model_validate_json(raw)
    if answer.question_id != question_id:
        raise ValueError('Unexpected question_id')
    if not set(answer.evidence_ids + answer.counterevidence_ids) <= allowed_claim_ids:
        raise ValueError('Evidence refers to an unavailable claim')
    return answer
