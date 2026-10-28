import argparse
import hashlib
import importlib.metadata
import json
import platform
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from .context_builder import build_context, build_context_pair
from .evaluation import summarize
from .llm import create_client
from .schemas import load_dataset, validate_answer

ROOT = Path(__file__).resolve().parents[1]


def digest(text: str) -> str:
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def prepare_cases(dataset):
    for question in dataset.questions:
        pair = build_context_pair(dataset, question.question)
        if set(pair[0].claim_ids) != set(pair[1].claim_ids):
            raise ValueError('B and C must receive the same claim IDs')
        for context in [build_context(dataset, question.question, 'A'), *pair]:
            user = 'КОНТЕКСТ ДЛЯ АНАЛИЗА\n' + context.text + '\nКОНЕЦ КОНТЕКСТА\n\nВОПРОС\n' + json.dumps({'question_id': question.question_id, 'question': question.question}, ensure_ascii=False) + '\nВерни только JSON с полями question_id, status, answer, evidence_ids, counterevidence_ids, confidence, explanation. Статус ответа: entailed, contradicted, both или insufficient.'
            yield question, context, user


def run_experiment(output_dir: str | Path, config_path: str | Path, data_dir: str | Path = ROOT / 'docs', metadata_path: str | Path = ROOT / 'config/claims_metadata.json', execute: bool = False, limit: int | None = None, scenario_manifest: str | Path | None = None):
    dataset = load_dataset(data_dir, metadata_path)
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise ValueError('Output directory must be empty to preserve previous runs')
    if scenario_manifest:
        scenario = json.loads(Path(scenario_manifest).read_text())
        expected_hash = digest(dataset.model_dump_json())
        if execute and (not scenario.get('human_reviewed') or scenario.get('reviewed_dataset_sha256') != expected_hash):
            raise ValueError('Stress dataset must be reviewed after its last change before running the model')
    else:
        if Path(data_dir).resolve() != (ROOT / 'docs').resolve():
            raise ValueError('A changed dataset requires its scenario manifest')
        scenario = {'scenario_id': 'baseline', 'human_reviewed': True, 'review_source': 'User confirmed dossier, claims and gold answers in this chat before implementation'}
    system = (ROOT / 'prompts/system.txt').read_text()
    config_path = Path(config_path).resolve()
    config = json.loads(config_path.read_text())
    cases = list(prepare_cases(dataset))
    if limit is not None:
        if limit <= 0:
            raise ValueError('limit must be positive')
        cases = cases[:limit]
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / 'dataset_snapshot.json', dataset.model_dump(mode='json'))
    versions = {}
    for package in ['pydantic', 'pandas', 'rank-bm25', 'mlx-lm', 'mlx']:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    manifest = dict(created_at=datetime.now(timezone.utc).isoformat(), python=sys.version, platform=platform.platform(), packages=versions, model_config=config, system_prompt_sha256=digest(system), dataset_sha256=digest(dataset.model_dump_json()), document_order=[d.document_id for d in dataset.documents], scenario=scenario, requested_cases=len(cases), execution_requested=execute, completed=False, semantic_response_review='not performed', timing='Per-request generation, excluding model load and file writes; input prefill included; no forced cold cache')
    write_json(output / 'manifest.json', manifest)
    (output / 'system.txt').write_text(system)
    coverage = []
    for question, context, user in cases:
        name = f'{question.question_id}_{context.mode}'
        (output / 'contexts').mkdir(exist_ok=True)
        (output / 'contexts' / f'{name}.txt').write_text(user)
        write_json(output / 'contexts' / f'{name}.json', asdict(context))
        expected = set(question.gold_evidence_ids + question.gold_counterevidence_ids)
        coverage.append(dict(question_id=question.question_id, mode=context.mode, selected_claim_ids=list(context.claim_ids), missing_gold_ids=sorted(expected - set(context.claim_ids)), gold_coverage=len(expected & set(context.claim_ids))/len(expected) if expected else None))
    write_json(output / 'retrieval_diagnostics.json', coverage)
    if not execute:
        manifest['preparation_completed'] = True
        write_json(output / 'manifest.json', manifest)
        return output
    client = create_client(config_path, ROOT)
    manifest['backend_info'] = getattr(client, 'backend_info', {})
    write_json(output / 'manifest.json', manifest)
    records = []
    for question, context, user in cases:
        record = dict(question_id=question.question_id, mode=context.mode, scenario_id=scenario['scenario_id'], allowed_claim_ids=list(context.claim_ids), prompt_sha256=digest(system+'\n'+user), valid=False, answer=None, raw_text=None, error=None, input_tokens=None, output_tokens=None, elapsed_seconds=None)
        start = time.perf_counter()
        try:
            generated = client.generate(system, user)
            record.update(asdict(generated))
            record['answer'] = validate_answer(generated.raw_text, question.question_id, set(context.claim_ids)).model_dump(mode='json')
            record['valid'] = True
        except Exception as exc:
            record['error'] = f'{type(exc).__name__}: {exc}'
            if record['elapsed_seconds'] is None:
                record['elapsed_seconds'] = time.perf_counter() - start
        with (output / 'answers.jsonl').open('a', encoding='utf-8') as file:
            file.write(json.dumps(record, ensure_ascii=False, allow_nan=False)+'\n')
            file.flush()
        records.append(record)
        print(f"{question.question_id} {context.mode}: {'valid' if record['valid'] else record['error']} ({record['elapsed_seconds']:.2f}s)", flush=True)
        write_json(output / 'progress.json', dict(completed=len(records), requested=len(cases)))
    if hasattr(client, 'close'):
        client.close()
    frame, summary = summarize(records, [q.model_dump(mode='json') for q in dataset.questions])
    frame.to_csv(output / 'per_question_metrics.csv', index=False)
    summary.to_csv(output / 'summary.csv', index=False)
    manifest['completed'] = True
    manifest['completed_cases'] = len(records)
    manifest['valid_responses'] = sum(r['valid'] for r in records)
    write_json(output / 'manifest.json', manifest)
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--config', default=str(ROOT / 'config/model.example.json'))
    parser.add_argument('--data-dir', default=str(ROOT / 'docs'))
    parser.add_argument('--metadata', default=str(ROOT / 'config/claims_metadata.json'))
    parser.add_argument('--scenario-manifest')
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    print(run_experiment(args.output, args.config, args.data_dir, args.metadata, args.run, args.limit, args.scenario_manifest))


if __name__ == '__main__':
    main()
