import hashlib
import json
from pathlib import Path

from src.schemas import load_dataset

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / 'stress' / 'scenarios'


def load_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def test_all_seventeen_scenario_datasets_are_consistent():
    index = load_json(FOLDER / 'index.json')
    assert len(index['scenarios']) == 17
    assert index['question_cases'] == 28
    total = 0
    for item in index['scenarios']:
        directory = FOLDER / item['scenario_id']
        manifest = load_json(directory / 'scenario.json')
        dataset = load_dataset(directory, directory / 'claims_metadata.json')
        actual = hashlib.sha256(dataset.model_dump_json().encode('utf-8')).hexdigest()
        assert manifest['dataset_sha256'] == actual
        assert manifest['scenario_id'] == item['scenario_id']
        assert {q.question_id:q.gold_status for q in dataset.questions} == manifest['expected_statuses']
        total += len(dataset.questions)
    assert total == 28


def test_permutation_changes_only_document_order():
    baseline = load_dataset(ROOT / 'docs', ROOT / 'config/claims_metadata.json')
    directory = FOLDER / 'permutation_42'
    alternate = load_dataset(directory, directory / 'claims_metadata.json')
    assert [d.document_id for d in baseline.documents] != [d.document_id for d in alternate.documents]
    assert {d.document_id:d for d in baseline.documents} == {d.document_id:d for d in alternate.documents}
    assert baseline.claims == alternate.claims
    assert baseline.questions == alternate.questions
    assert baseline.metadata == alternate.metadata


def test_removals_delete_required_claims_and_change_expected_status():
    baseline = load_dataset(ROOT / 'docs', ROOT / 'config/claims_metadata.json')
    original = {c.claim_id for c in baseline.claims}
    for qid in ['Q03', 'Q04', 'Q05', 'Q06']:
        directory = FOLDER / f'remove_{qid}'
        manifest = load_json(directory / 'scenario.json')
        variant = load_dataset(directory, directory / 'claims_metadata.json')
        removed = set(manifest['specification']['removed_claim_ids'])
        assert original - {c.claim_id for c in variant.claims} == removed
        assert len(variant.questions) == 1
        assert variant.questions[0].question_id == qid
        assert variant.questions[0].gold_status == 'insufficient'


def test_addition_scenarios_use_separate_new_source_without_overwriting_baseline():
    baseline = load_dataset(ROOT / 'docs', ROOT / 'config/claims_metadata.json')
    for item in load_json(FOLDER / 'index.json')['scenarios']:
        sid = item['scenario_id']
        if sid == 'permutation_42' or sid.startswith('remove_'):
            continue
        directory = FOLDER / sid
        variant = load_dataset(directory, directory / 'claims_metadata.json')
        old_docs = {d.document_id:d for d in variant.documents if d.document_id != 'D11'}
        old_claims = {c.claim_id:c for c in variant.claims if c.claim_id != 'C41'}
        assert old_docs == {d.document_id:d for d in baseline.documents}
        assert old_claims == {c.claim_id:c for c in baseline.claims}
        assert len(variant.documents) == len(baseline.documents)+1
        assert len(variant.claims) == len(baseline.claims)+1
        assert len(variant.questions) == 1
