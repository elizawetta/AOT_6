import json
from copy import deepcopy
from pathlib import Path

import pandas as pd

from .experiment import digest, write_json
from .schemas import Claim, ClaimMetadata, Dataset, Document, Question


def save_variant(dataset: Dataset, output_dir: Path, scenario_id: str, kind: str, details: dict):
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, records in [('documents', dataset.documents), ('claims', dataset.claims), ('questions', dataset.questions)]:
        (output_dir / f'{name}.jsonl').write_text(''.join(x.model_dump_json()+'\n' for x in records))
    write_json(output_dir / 'claims_metadata.json', {k:v.model_dump(mode='json') for k,v in dataset.metadata.items()})
    manifest = dict(scenario_id=scenario_id, kind=kind, human_reviewed=False, reviewed_dataset_sha256=None, dataset_sha256=digest(dataset.model_dump_json()), expected_statuses={q.question_id:q.gold_status for q in dataset.questions}, **details)
    write_json(output_dir / 'scenario.json', manifest)
    return manifest


def permute_documents(dataset: Dataset, random_state: int = 42) -> Dataset:
    value = dataset.model_dump(mode='json')
    order = pd.Series(range(len(dataset.documents))).sample(frac=1, random_state=random_state).tolist()
    value['documents'] = [value['documents'][i] for i in order]
    return Dataset.model_validate_json(json.dumps(value))


def remove_premise(dataset: Dataset, question_id: str, removed_ids: list[str], replacements: dict[str, list[tuple[str, str]]], updated_evidence: dict[str, str] | None = None) -> Dataset:
    value = dataset.model_dump(mode='json')
    for doc in value['documents']:
        for old, new in replacements.get(doc['document_id'], []):
            if old not in doc['text']:
                raise ValueError(f"Redaction missing from {doc['document_id']}: {old}")
            doc['text'] = doc['text'].replace(old, new)
    value['claims'] = [c for c in value['claims'] if c['claim_id'] not in removed_ids]
    for claim in value['claims']:
        if claim['claim_id'] in (updated_evidence or {}):
            claim['evidence'] = updated_evidence[claim['claim_id']]
    value['metadata'] = {k:v for k,v in value['metadata'].items() if k not in removed_ids}
    question = next(q for q in value['questions'] if q['question_id']==question_id)
    question.update(gold_status='insufficient', gold_answer='После удаления необходимой посылки определённый ответ по изменённому досье не следует; недостаёт удалённого факта, указанного в описании сценария.', gold_evidence_ids=[], gold_counterevidence_ids=[])
    value['questions'] = [question]
    return Dataset.model_validate_json(json.dumps(value))


def add_contradiction(dataset: Dataset, question_id: str, new_text: str, expected_status: str, evidence: list[str], counter: list[str], kind: str = 'system_log') -> Dataset:
    value = dataset.model_dump(mode='json')
    docid, cid = 'D11', 'C41'
    if any(d['document_id']==docid for d in value['documents']) or any(c['claim_id']==cid for c in value['claims']):
        raise ValueError('Scenario identifiers already exist')
    stamp = '2026-03-14T18:50:00+03:00'
    value['documents'].append(dict(document_id=docid,title='Подготовленный альтернативный источник',source_type=kind,created_at=stamp,author='Независимый архивный источник',text=new_text))
    value['claims'].append(dict(claim_id=cid,text=new_text,document_id=docid,evidence=new_text,status='disputed'))
    value['metadata'][cid]=dict(path=[2 if kind=='policy' else (1 if kind=='statement' else 0),2,2],occurred_at=None,rationale='Added contradictory source; no independent precedence over the original source is established. Event time is not assigned from document creation time.')
    question = next(q for q in value['questions'] if q['question_id']==question_id)
    question.update(gold_status=expected_status,gold_answer='После добавления источника сохраняются несовместимые основания; выбирать одну версию без дополнительных сведений нельзя.',gold_evidence_ids=evidence,gold_counterevidence_ids=counter)
    value['questions']=[question]
    return Dataset.model_validate_json(json.dumps(value))
