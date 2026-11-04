import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation import compare_runs
from src.experiment import run_experiment, digest
from src.schemas import load_dataset

SCENARIOS = ROOT / 'stress' / 'scenarios'
BASELINE = ROOT / 'results' / 'gigachat_baseline_v2'
CONFIG = ROOT / 'config' / 'model.local.json'


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def state(directory):
    m = read_json(directory / 'scenario.json')
    data = load_dataset(directory, directory / 'claims_metadata.json')
    actual_hash = digest(data.model_dump_json())
    current = actual_hash == m['dataset_sha256']
    reviewed = bool(m.get('human_reviewed')) and m.get('reviewed_dataset_sha256') == actual_hash
    return m, data, current, reviewed


def output_path(scenario_id):
    return ROOT / 'results' / f'stress_{scenario_id}_gigachat'


def summary_for(scenario_id, base_records, output):
    records = [json.loads(line) for line in (output / 'answers.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
    changed = read_json(output / 'dataset_snapshot.json')
    gold = {q['question_id']: q for q in changed['questions']}
    comp = compare_runs(base_records, records)
    statuses = {(r['question_id'], r['mode']): r['answer']['status'] for r in base_records if r['valid']}
    enriched = []
    for r in records:
        key = (r['question_id'], r['mode'])
        current_status = r['answer']['status'] if r['valid'] else None
        expected_status = gold[r['question_id']]['gold_status']
        enriched.append(dict(scenario_id=scenario_id, question_id=r['question_id'], mode=r['mode'], baseline_status=statuses.get(key), expected_status=expected_status, observed_status=current_status, schema_valid=r['valid'], expected_status_correct=(current_status == expected_status), baseline_status_changed=(current_status != statuses.get(key)) if current_status is not None and key in statuses else None))
    frame = pd.DataFrame(enriched)
    frame.to_csv(output / 'stress_comparison.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(comp).to_csv(output / 'stability.csv', index=False, encoding='utf-8-sig')
    return frame


def main():
    p = argparse.ArgumentParser(description='Run only human-reviewed stress scenarios, preserving existing results')
    p.add_argument('--run', action='store_true', help='Make paid API calls (without this option: list only)')
    p.add_argument('--scenario', help='Run just one scenario, e.g. permutation_42')
    p.add_argument('--max-scenarios', type=int, default=None, help='Stop after N runnable scenarios')
    args = p.parse_args()

    index = read_json(SCENARIOS / 'index.json')
    if args.scenario and args.scenario not in {s['scenario_id'] for s in index['scenarios']}:
        p.error('Unknown scenario')
    choices = [s for s in index['scenarios'] if not args.scenario or s['scenario_id'] == args.scenario]
    checks = []
    for s in choices:
        d = SCENARIOS / s['scenario_id']
        manifest, data, current, reviewed = state(d)
        out = output_path(s['scenario_id'])
        complete = (out / 'manifest.json').is_file() and read_json(out / 'manifest.json').get('completed', False)
        print(f"{s['scenario_id']}: cases={len(data.questions)*3}, data={'OK' if current else 'CHANGED'}, manual_review={'YES' if reviewed else 'NO'}, result={'DONE' if complete else 'MISSING'}")
        checks.append((s['scenario_id'], d, current, reviewed, out, complete))

    if not args.run:
        print('\nТолько проверка состояния. Запросов к GigaChat не было.')
        return
    if not CONFIG.is_file() or not BASELINE.is_dir():
        raise SystemExit('Сначала нужен config/model.local.json и результаты gigachat_baseline_v2')
    base_manifest = read_json(BASELINE / 'manifest.json')
    if not base_manifest.get('completed'):
        raise SystemExit('Исходный эксперимент не завершён')
    config_value = read_json(CONFIG)
    system_sha = digest((ROOT / 'prompts/system.txt').read_text(encoding='utf-8'))
    if base_manifest.get('model_config') != config_value or base_manifest.get('system_prompt_sha256') != system_sha:
        raise SystemExit('Конфигурация модели или промпт отличаются от исходного эксперимента. Нельзя корректно сравнивать результаты.')
    base_records = [json.loads(s) for s in (BASELINE / 'answers.jsonl').read_text(encoding='utf-8').splitlines() if s.strip()]
    completed_frames = []
    launched = 0
    for scenario_id, d, current, reviewed, out, complete in checks:
        if not current:
            print(f'SKIP {scenario_id}: data changed since scenario manifest')
            continue
        if not reviewed:
            print(f'SKIP {scenario_id}: human review required')
            continue
        if complete:
            print(f'SKIP {scenario_id}: run already completed')
            completed_frames.append(summary_for(scenario_id, base_records, out))
            continue
        if out.exists() and any(out.iterdir()):
            print(f'SKIP {scenario_id}: nonempty incomplete results directory; inspect manually')
            continue
        if args.max_scenarios is not None and launched >= args.max_scenarios:
            break
        print(f'\nSTART {scenario_id} ({len(load_dataset(d, d / "claims_metadata.json").questions)*3} API calls)')
        run_experiment(out, CONFIG, d, d / 'claims_metadata.json', execute=True, scenario_manifest=d / 'scenario.json')
        completed_frames.append(summary_for(scenario_id, base_records, out))
        print(f'FINISHED {scenario_id}: {out}')
        launched += 1
    if completed_frames:
        full = pd.concat(completed_frames, ignore_index=True)
        full.to_csv(ROOT / 'results' / 'stress_overview.csv', index=False, encoding='utf-8-sig')
        print('\nОбщий файл: results/stress_overview.csv')
        print(full.groupby('mode').agg(valid=('schema_valid','mean'), correct=('expected_status_correct','mean')).to_string())
    else:
        print('Ни один сценарий не запускался: сначала требуется ручная проверка.')


if __name__ == '__main__':
    main()
