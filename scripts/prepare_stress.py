import argparse
import difflib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.experiment import digest, write_json
from src.schemas import Dataset, load_dataset
from src.stress import add_contradiction, permute_documents, remove_premise, save_variant


def with_specification(dataset, spec):
    value = dataset.model_dump(mode='json')
    question = value['questions'][0]
    for name in ['gold_status', 'gold_answer', 'gold_evidence_ids', 'gold_counterevidence_ids']:
        question[name] = spec[name]
    if 'new_text' in spec:
        value['claims'][-1].update(text=spec['claim_text'], status=spec['claim_status'])
        value['metadata']['C41'] = spec['metadata']
        if spec['kind'] == 'add_unverified_control':
            value['documents'][-1].update(title='Неподтверждённое анонимное сообщение', author='Неустановленный автор')
    return Dataset.model_validate_json(json.dumps(value))


def write_snapshot(dataset, directory):
    directory.mkdir(parents=True, exist_ok=True)
    for name in ['documents', 'claims', 'questions']:
        (directory / f'{name}.jsonl').write_text(''.join(x.model_dump_json() + '\n' for x in getattr(dataset, name)))
    write_json(directory / 'claims_metadata.json', {k: v.model_dump(mode='json') for k, v in dataset.metadata.items()})


def write_diff(baseline, changed, path):
    before = {d.document_id: d for d in baseline.documents}
    after = {d.document_id: d for d in changed.documents}
    parts = []
    for did in sorted(before.keys() | after.keys()):
        a = (before[did].text + '\n').splitlines(keepends=True) if did in before else []
        b = (after[did].text + '\n').splitlines(keepends=True) if did in after else []
        parts.extend(difflib.unified_diff(a, b, fromfile=f'baseline/{did}', tofile=f'{path.parent.name}/{did}', n=1))
    path.write_text(''.join(parts))


def build(output):
    dataset = load_dataset(ROOT / 'docs', ROOT / 'config/claims_metadata.json')
    specs = json.loads((ROOT / 'stress/specifications.json').read_text())
    if output.exists() and any(output.iterdir()):
        raise ValueError('Output must be empty; previous scenarios and review history are preserved')
    output.mkdir(parents=True, exist_ok=True)
    manifests = []
    baseline_hash = digest(dataset.model_dump_json())
    write_snapshot(dataset, output / '_baseline')
    write_json(output / '_baseline/provenance.json', dict(dataset_sha256=baseline_hash, source='docs + config/claims_metadata.json', approval='User confirmed baseline dossier, claims and gold answers in this chat; no new scenario inherits that approval'))
    p = specs['permutation']
    changed = permute_documents(dataset, p['random_state'])
    manifest = save_variant(changed, output / p['scenario_id'], p['scenario_id'], 'permutation', dict(random_state=p['random_state'], baseline_sha256=baseline_hash, original_document_order=[d.document_id for d in dataset.documents], changed_document_order=[d.document_id for d in changed.documents], factor='Изменён только порядок документов; тексты, утверждения, метаданные и эталоны сохранены.', assistant_reviewed=False))
    write_diff(dataset, changed, output / p['scenario_id'] / 'changes.patch')
    manifests.append(manifest)
    for spec in specs['removals'] + specs['contradictions'] + specs['controls']:
        if spec['kind'] == 'remove_premise':
            changed = remove_premise(dataset, spec['question_id'], spec['removed_claim_ids'], spec['replacements'], spec['updated_evidence'])
        else:
            changed = add_contradiction(dataset, spec['question_id'], spec['new_text'], spec['gold_status'], spec['gold_evidence_ids'], spec['gold_counterevidence_ids'], spec['source_type'])
        changed = with_specification(changed, spec)
        directory = output / spec['scenario_id']
        manifest = save_variant(changed, directory, spec['scenario_id'], spec['kind'], dict(baseline_sha256=baseline_hash, specification=spec, assistant_reviewed=False, review_note='Generated from specification; generation alone does not constitute semantic or human review'))
        write_diff(dataset, changed, directory / 'changes.patch')
        manifests.append(manifest)
    counts = {'permutation_question_cases': 12, 'removal_question_cases': 4, 'contradiction_question_cases': 10, 'unverified_control_question_cases': 2}
    write_json(output / 'index.json', dict(version=specs['version'], baseline_sha256=baseline_hash, scenario_count=len(manifests), question_cases=sum(counts.values()), calls_for_three_modes=3 * sum(counts.values()), counts=counts, scenarios=manifests))
    rows = []
    for q in dataset.questions:
        qid = q.question_id
        rows.append(dict(question_id=qid, permutation='permutation_42', remove_premise='remove_' + qid if qid in {'Q03', 'Q04', 'Q05', 'Q06'} else None, removal_note=None if qid in {'Q03', 'Q04', 'Q05', 'Q06'} else 'Not applicable: assignment specifies multi-hop and temporal questions', addition='contradiction_' + qid if int(qid[1:]) <= 10 else 'unverified_' + qid, addition_role='contradiction' if int(qid[1:]) <= 10 else 'unverified_control'))
    write_json(output / 'applicability.json', rows)
    intro = '''# Стресс-сценарии для передачи Максиму

Это сгенерированный комплект данных, версия 2. Генерация сама по себе не подтверждает логику эталонов; статус проверки фиксируется отдельно. Проверка человеком и запуск LLM не заявляются. Предыдущие черновики в stress/drafts сохранены для истории; для продолжения используется этот комплект.

Всего 17 наборов, 28 пар «вариант–вопрос», 84 вызова при трёх режимах. Перестановка относится ко всем 12 вопросам; удаление — к четырём multi-hop и временным вопросам. Десять добавлений создают конфликт, ещё два — контроль неподтверждённых сведений для исходно insufficient вопросов. Эти два контроля не выдаются за создание противоречия и не включаются в метрику перехода к both.

Сравнивать каждый вариант только с исходным досье, не применять изменения последовательно друг к другу. Порядок остальных документов сохранён. C41/D11 существуют отдельно в каждом добавлении, а не обозначают один общий источник для всех сценариев. Во всех режимах запрещено передавать модели эталоны, спецификации изменений или скрытую авторскую хронологию.

Изменение считается одним смысловым фактором: один факт скрывается вместе со всеми раскрывающими его повторами и производными выводами либо добавляется одно несовместимое сообщение. Удаление исходного C40 вместе с временной посылкой не является вторым независимым воздействием. Изменение ожидаемого ответа служит разметкой и не входит в контекст LLM.

## Таблица ожидаемых результатов

| Набор | Вопрос | Исходный статус | Новый статус | Изменение |
|---|---|---|---|---|
'''
    original = {q.question_id: q for q in dataset.questions}
    lines = [intro]
    for manifest in manifests:
        for qid, status in manifest['expected_statuses'].items():
            lines.append(f"| {manifest['scenario_id']} | {qid} | {original[qid].gold_status} | {status} | {manifest.get('factor', manifest.get('specification', {}).get('factor', ''))} |\n")
    lines.append('''
## Как проверять

В каждой папке находятся documents.jsonl, claims.jsonl, questions.jsonl, claims_metadata.json, scenario.json и changes.patch. В questions.jsonl указан конкретный новый ответ, включая недостающую информацию для insufficient. Снимок исходного набора находится в _baseline.

Для подтверждения фактической проверки человеком: `.venv/bin/python scripts/review_scenario.py stress/scenarios/ИМЯ --reviewer "Имя участника"`. Команду выполняют после просмотра изменений и эталона. Она сохраняет хеш проверенного набора; любая последующая правка требует повторной проверки. Ассистент не выставляет human_reviewed=true за человека.

Максиму нужно выполнить исходные и изменённые наборы одной моделью API, с теми же параметрами и промптом. Ответы сохранить до ручного анализа. Для перестановки сравнить долю сменившихся статусов и Jaccard доказательств; для удаления — долю ответов с ожидаемым insufficient; для добавления противоречия — долю ожидаемых both. Отдельно показать, что Q09 уже был both, поэтому здесь оценивается сохранение конфликта, а не переход в него. Для двух неподтверждённых сообщений считать сохранение insufficient отдельно.

У задания есть неоднозначность: общая фраза требует три теста для всех 12 вопросов, но удаление явно ограничено multi-hop и временными; из исходного insufficient нельзя автоматически получить both добавлением одного сообщения. Здесь применимость зафиксирована честно. Контроли Q11–Q12 дают проверку добавления информации для всех вопросов, но не подменяют формальное требование о несовместимом утверждении. Если преподаватель требует именно конфликт для каждого insufficient-вопроса, это потребует отдельной постановки, а не выдуманного эталона both.

## Эталоны и объяснения
''')
    for manifest in manifests:
        directory = output / manifest['scenario_id']
        changed = load_dataset(directory, directory / 'claims_metadata.json')
        if manifest['kind'] == 'permutation':
            continue
        q = changed.questions[0]
        lines.append(f"\n### {manifest['scenario_id']}\n\n{q.question}\n\n**Ответ:** {q.gold_answer}\n\n**Статус:** {q.gold_status}. **Основания:** {', '.join(q.gold_evidence_ids) or 'нет'}. **Контроснования:** {', '.join(q.gold_counterevidence_ids) or 'нет'}.\n")
    (output / 'README.md').write_text(''.join(lines))
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=ROOT / 'stress/scenarios')
    args = parser.parse_args()
    print(build(args.output))
