import itertools
import json
from fractions import Fraction
from pathlib import Path

import pytest
from pydantic import ValidationError

from src.context_builder import build_context_pair
from src.evaluation import evaluate_record
from src.llm import LLMUnavailable, MLXClient
from src.padic import decode, distance, encode
from src.retrieval import ClaimRetriever
from src.schemas import load_dataset, validate_answer

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def dataset():
    return load_dataset(ROOT / 'docs', ROOT / 'config/claims_metadata.json')


def response(status='entailed', answer='Ответ', evidence=None, counter=None):
    return json.dumps({
        'question_id': 'Q01',
        'status': status,
        'answer': answer,
        'evidence_ids': ['C01'] if evidence is None else evidence,
        'counterevidence_ids': [] if counter is None else counter,
        'confidence': 0.8,
        'explanation': 'Проверяемая связь',
    }, ensure_ascii=False)


def test_same_padic_paths_have_zero_distance():
    assert distance((2, 1, 0), (2, 1, 0)) == Fraction(0)


def test_difference_at_first_padic_level():
    assert distance((0, 1, 1), (1, 1, 1)) == Fraction(1)


def test_difference_at_second_padic_level():
    assert distance((2, 0, 1), (2, 1, 1)) == Fraction(1, 3)


def test_difference_at_third_padic_level():
    assert distance((2, 1, 0), (2, 1, 1)) == Fraction(1, 9)


def test_encode_decode_roundtrip():
    for path in itertools.product(range(3), repeat=3):
        assert decode(encode(path)) == path


def test_ultrametric_inequality():
    paths = list(itertools.product(range(3), repeat=3))
    for a, b, c in itertools.product(paths, repeat=3):
        assert distance(a, c) <= max(distance(a, b), distance(b, c))


def test_invalid_json_rejected():
    with pytest.raises(ValidationError):
        validate_answer('{not a json}', 'Q01', {'C01'})


def test_unknown_evidence_id_rejected():
    with pytest.raises(ValueError, match='unavailable claim'):
        validate_answer(response(evidence=['C999']), 'Q01', {'C01'})


def test_known_but_unavailable_evidence_rejected():
    with pytest.raises(ValueError, match='unavailable claim'):
        validate_answer(response(evidence=['C02']), 'Q01', {'C01'})


def test_both_requires_counterevidence():
    with pytest.raises(ValidationError):
        validate_answer(response(status='both', counter=[]), 'Q01', {'C01'})


def test_boolean_answer_rejected():
    data = json.loads(response())
    data['answer'] = True
    with pytest.raises(ValidationError):
        validate_answer(json.dumps(data), 'Q01', {'C01'})


def test_insufficient_fabricated_answer_requires_manual_review():
    # Schema validation checks structure, not whether natural-language facts are true.
    # A nonempty answer under "insufficient" is flagged for human semantic review.
    text = response(status='insufficient', answer='Виновен неизвестный человек из другого города', evidence=[])
    result = validate_answer(text, 'Q01', {'C01'})
    gold = {'gold_status': 'insufficient', 'gold_evidence_ids': [], 'gold_counterevidence_ids': []}
    metrics = evaluate_record({'valid': True, 'answer': result.model_dump()}, gold)
    assert metrics['insufficient_answer_review_required'] is True
    assert metrics['valid'] is True  # Semantic concern must not be mislabeled as invalid JSON.

    no_claim = response(status='insufficient', answer=None, evidence=[])
    no_claim_result = validate_answer(no_claim, 'Q01', {'C01'})
    no_claim_metrics = evaluate_record({'valid': True, 'answer': no_claim_result.model_dump()}, gold)
    assert no_claim_metrics['insufficient_answer_review_required'] is False


def test_unavailable_local_llm_raises():
    with pytest.raises(LLMUnavailable):
        MLXClient({'model_path': 'missing', 'revision': 'missing', 'max_context_tokens': 100}, ROOT)



def test_unavailable_gigachat_is_recorded_without_network(monkeypatch, tmp_path):
    # Simulate a failed GigaChat call at the actual experiment-client boundary.
    # No real API, credentials, or network connection are used.
    from src import experiment

    class UnavailableGigaChat:
        def generate(self, system, user):
            raise LLMUnavailable('GigaChat API недоступен')

    monkeypatch.setattr(experiment, 'create_client', lambda config_path, root: UnavailableGigaChat())
    config = tmp_path / 'model.json'
    config.write_text(json.dumps({'backend': 'gigachat', 'model': 'GigaChat-2-Pro'}), encoding='utf-8')
    output = tmp_path / 'unavailable_gigachat'

    experiment.run_experiment(output, config, execute=True, limit=1)

    records = [json.loads(line) for line in (output / 'answers.jsonl').read_text(encoding='utf-8').splitlines()]
    assert len(records) == 1
    assert records[0]['valid'] is False
    assert records[0]['answer'] is None
    assert 'LLMUnavailable: GigaChat API недоступен' in records[0]['error']
    manifest = json.loads((output / 'manifest.json').read_text(encoding='utf-8'))
    assert manifest['completed'] is True
    assert manifest['completed_cases'] == 1
    assert manifest['valid_responses'] == 0


def test_bm25_returns_at_most_fifteen(dataset):
    retriever = ClaimRetriever(dataset.claims)
    for question in dataset.questions:
        hits = retriever.search(question.question)
        assert len(hits) <= 15
        assert len({h.claim.claim_id for h in hits}) == len(hits)


def test_b_and_c_have_identical_claim_ids_and_payloads(dataset):
    for question in dataset.questions:
        b, c = build_context_pair(dataset, question.question)
        assert set(b.claim_ids) == set(c.claim_ids)
        b_payload = {x['claim_id']: x for x in (json.loads(line) for line in b.text.splitlines() if line.startswith('{'))}
        c_payload = {x['claim_id']: x for x in (json.loads(line) for line in c.text.splitlines() if line.startswith('{'))}
        assert b_payload == c_payload
        assert len(b_payload) <= 15
