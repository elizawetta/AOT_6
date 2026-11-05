import json
from pathlib import Path

import pandas as pd
import streamlit as st

from src.context_builder import build_context, build_context_pair
from src.evaluation import evaluate_record
from src.experiment import prepare_cases
from src.llm import create_client
from src.schemas import load_dataset, validate_answer

ROOT = Path(__file__).resolve().parent
STATUS_LABELS = {
    'entailed': 'Подтверждено',
    'contradicted': 'Опровергнуто',
    'both': 'Есть противоречие',
    'insufficient': 'Недостаточно данных',
}
KIND_LABELS = {
    'permutation': 'Перестановка',
    'remove_premise': 'Удаление посылки',
    'add_contradiction': 'Противоречие',
    'add_unverified_control': 'Контрольный сценарий',
}
MODE_LABELS = {
    'A': 'Полное досье',
    'B': 'BM25 · топ-15',
    'C': 'p-adic · топ-15',
}

st.set_page_config(page_title='AI-детектив | Контур', page_icon='🔎', layout='wide', initial_sidebar_state='expanded')
st.markdown('''
<style>
.block-container {max-width: 1360px; padding-top: 2rem;}
[data-testid="stMetric"] {border: 1px solid rgba(130,148,165,.24); border-radius: 14px; padding: 14px 18px;}
[data-testid="stSidebar"] {border-right: 1px solid rgba(130,148,165,.17);}
h1, h2, h3 {letter-spacing: -.025em;}
div.stButton > button[kind="primary"] {border-radius: 10px;}
</style>
''', unsafe_allow_html=True)


@st.cache_resource(show_spinner='Загружаем досье')
def get_dataset():
    return load_dataset(ROOT / 'docs', ROOT / 'config/claims_metadata.json')


@st.cache_data(show_spinner=False)
def get_answers(path_str):
    path = Path(path_str) / 'answers.jsonl'
    if not path.exists():
        return {}
    rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
    return {(row['question_id'], row['mode']): row for row in rows}



def stress_baseline_comparison(baseline, changed, original_gold, expected_statuses):
    """Pair baseline and stress answers by (question_id, mode)."""
    rows = []
    for (question_id, mode), current in sorted(changed.items()):
        original = baseline.get((question_id, mode))
        if original is None:
            continue
        old_answer = original.get('answer') if original.get('valid') else None
        new_answer = current.get('answer') if current.get('valid') else None
        old_status = old_answer.get('status') if old_answer else None
        new_status = new_answer.get('status') if new_answer else None
        target = expected_statuses.get(question_id)
        if target is None:
            continue
        old_correct = old_status == original_gold.get(question_id) if old_status else False
        new_correct = new_status == target if new_status else False
        old_evidence = (set(old_answer.get('evidence_ids', [])), set(old_answer.get('counterevidence_ids', []))) if old_answer else None
        new_evidence = (set(new_answer.get('evidence_ids', [])), set(new_answer.get('counterevidence_ids', []))) if new_answer else None
        rows.append({
            'Вопрос': question_id,
            'Режим': mode,
            'Статус до': STATUS_LABELS.get(old_status, old_status or 'Ошибка формата'),
            'Статус после': STATUS_LABELS.get(new_status, new_status or 'Ошибка формата'),
            'Правильно до': old_correct,
            'Правильно после': new_correct,
            'Статус изменился': old_status != new_status,
            'Доказательства изменились': old_evidence != new_evidence,
        })
    detail = pd.DataFrame(rows)
    if detail.empty:
        return detail, detail
    summary = []
    for mode, frame in detail.groupby('Режим', sort=True):
        n = len(frame)
        old_ok = int(frame['Правильно до'].sum())
        new_ok = int(frame['Правильно после'].sum())
        summary.append({
            'Режим': mode,
            'Правильно до': f'{old_ok}/{n} ({old_ok/n:.1%})',
            'Правильно после': f'{new_ok}/{n} ({new_ok/n:.1%})',
            'Изменилось статусов': int(frame['Статус изменился'].sum()),
            'Стало правильнее': int(((~frame['Правильно до']) & frame['Правильно после']).sum()),
            'Стало неправильнее': int((frame['Правильно до'] & (~frame['Правильно после'])).sum()),
            'Изменились доказательства': int(frame['Доказательства изменились'].sum()),
        })
    return pd.DataFrame(summary), detail


def show_record(record, question):
    if not record:
        st.info('Для этого вопроса ответа в выбранном прогоне нет.')
        return
    if not record.get('valid'):
        st.error('Ответ не прошёл проверку формата или ссылок')
        st.code(record.get('error') or 'Неизвестная ошибка')
        if record.get('raw_text'):
            with st.expander('Исходный ответ модели'):
                st.code(record['raw_text'], language='json')
        return
    answer = record['answer']
    correct = answer['status'] == question.gold_status
    left, middle, right = st.columns([1.3, 1, 1])
    left.metric('Логический статус', STATUS_LABELS.get(answer['status'], answer['status']))
    middle.metric('Совпадение с эталоном', 'Да' if correct else 'Нет')
    right.metric('Уверенность модели', f"{answer['confidence']:.0%}")
    st.markdown('**Краткий ответ:** ' + str(answer['answer'] if answer['answer'] is not None else 'Не установлен'))
    st.write(answer['explanation'])
    col1, col2 = st.columns(2)
    col1.write('**Доказательства:** ' + (', '.join(answer['evidence_ids']) or 'нет'))
    col2.write('**Контрдоказательства:** ' + (', '.join(answer['counterevidence_ids']) or 'нет'))
    metric = evaluate_record(record, question.model_dump(mode='json'))
    a, b, c = st.columns(3)
    a.metric('Evidence F1', f"{metric['evidence_f1']:.1%}")
    b.metric('Время ответа', f"{record['elapsed_seconds']:.2f} с" if record.get('elapsed_seconds') is not None else '—')
    c.metric('Выходные токены', str(record.get('output_tokens') if record.get('output_tokens') is not None else '—'))
    with st.expander('Исходный JSON модели'):
        st.code(record.get('raw_text') or json.dumps(answer, ensure_ascii=False, indent=2), language='json')


st.title('AI-детектив')
st.caption('Исследовательский стенд · вымышленный турнир «Контур» · сравнение контекстов A / B / C')
try:
    dataset = get_dataset()
except Exception as exc:
    st.error(f'Не удалось загрузить досье: {exc}')
    st.stop()

available_runs = sorted(
    [p for p in (ROOT / 'results').iterdir() if p.is_dir() and (p / 'answers.jsonl').is_file()],
    key=lambda p: (p.name != 'gigachat_baseline_v2', p.name),
) if (ROOT / 'results').exists() else []

st.sidebar.header('Параметры стенда')
if available_runs:
    selected_run = st.sidebar.selectbox('Сохранённый эксперимент', available_runs, format_func=lambda p: p.name)
    saved_answers = get_answers(str(selected_run))
else:
    selected_run = None
    saved_answers = {}
    st.sidebar.info('Нет сохранённых экспериментов')

config_path = ROOT / 'config/model.local.json'
if config_path.exists():
    try:
        model_info = json.loads(config_path.read_text(encoding='utf-8'))
        st.sidebar.caption(f"Модель: {model_info.get('model', 'не указана')}")
    except (OSError, ValueError):
        st.sidebar.caption('Конфигурация модели не читается')
else:
    st.sidebar.caption('Конфигурация API не обнаружена')

intro, results, dossier, stress = st.tabs(['Исследование', 'Результаты', 'Досье', 'Стресс-тесты'])

with intro:
    st.subheader('Проверка вопроса')
    qids = [q.question_id for q in dataset.questions]
    selected_qid = st.selectbox('Вопрос', qids, format_func=lambda qid: f"{qid} — {next(q.question for q in dataset.questions if q.question_id == qid)}")
    question = next(q for q in dataset.questions if q.question_id == selected_qid)
    st.write(question.question)
    st.caption(f'Эталонный статус: {STATUS_LABELS[question.gold_status]}')
    pair = build_context_pair(dataset, question.question)
    contexts = {
        'A': build_context(dataset, question.question, 'A'),
        'B': pair[0],
        'C': pair[1],
    }
    if set(contexts['B'].claim_ids) != set(contexts['C'].claim_ids):
        st.error('Ошибка: состав утверждений B и C различается')
        st.stop()
    if 'live_results' not in st.session_state:
        st.session_state.live_results = {}
    for pane, mode in zip(st.tabs([f'{mode} · {MODE_LABELS[mode]}' for mode in 'ABC']), 'ABC'):
        with pane:
            context = contexts[mode]
            col1, col2, col3 = st.columns(3)
            col1.metric('Утверждения', len(context.claim_ids))
            col2.metric('Размер контекста', f'{len(context.text):,} символов'.replace(',', ' '))
            col3.metric('Источники', len(context.document_ids) if mode == 'A' else len({c.document_id for c in dataset.claims if c.claim_id in context.claim_ids}))
            if mode == 'C':
                from src.padic import group_claims
                selected = [c for c in dataset.claims if c.claim_id in context.claim_ids]
                branches = group_claims(selected, dataset.metadata)
                st.write('**Ветви p-адической классификации**')
                for path, claims in branches:
                    st.caption(f"{'.'.join(map(str, path))} · {', '.join(c.claim_id for c in claims)}")
            with st.expander('Показать передаваемый контекст'):
                st.text_area('Контекст', value=context.text, height=320, disabled=True, key=f'context_{selected_qid}_{mode}')
            with st.expander('Показать отобранные факты и цитаты'):
                by_id = {c.claim_id: c for c in dataset.claims}
                for cid in context.claim_ids:
                    claim = by_id[cid]
                    st.markdown(f'**{cid}** · {claim.text}')
                    st.caption(f'{claim.document_id}: {claim.evidence}')
            st.markdown('#### Ответ модели')
            saved = saved_answers.get((selected_qid, mode))
            show_record(saved, question)
            st.divider()
            if st.button('Запросить новый ответ GigaChat', key=f'ask_{selected_qid}_{mode}', type='primary', disabled=not config_path.exists()):
                client = None
                raw = None
                try:
                    case = next(user for q, c, user in prepare_cases(dataset) if q.question_id == selected_qid and c.mode == mode)
                    system = (ROOT / 'prompts/system.txt').read_text(encoding='utf-8')
                    with st.spinner('Модель обрабатывает контекст'):
                        client = create_client(config_path, ROOT)
                        generated = client.generate(system, case)
                    raw = generated.raw_text
                    answer = validate_answer(raw, selected_qid, set(context.claim_ids)).model_dump(mode='json')
                    entry = {
                        'question_id': selected_qid,
                        'mode': mode,
                        'valid': True,
                        'answer': answer,
                        'raw_text': raw,
                        'input_tokens': generated.input_tokens,
                        'output_tokens': generated.output_tokens,
                        'elapsed_seconds': generated.elapsed_seconds,
                    }
                except Exception as exc:
                    entry = {'valid': False, 'raw_text': raw, 'error': f'{type(exc).__name__}: {exc}'}
                finally:
                    if client is not None:
                        client.close()
                st.session_state.live_results[(selected_qid, mode)] = entry
            live = st.session_state.live_results.get((selected_qid, mode))
            if live is not None:
                st.markdown('#### Новый ответ (не сохранён в baseline)')
                show_record(live, question)

with results:
    st.subheader('Сводка по эксперименту')
    if not selected_run:
        st.info('Нет сохранённых результатов')
    else:
        summary_path = selected_run / 'summary.csv'
        metrics_path = selected_run / 'per_question_metrics.csv'
        if summary_path.is_file():
            summary = pd.read_csv(summary_path)
            cols = ['mode', 'status_accuracy', 'evidence_f1', 'mean_seconds', 'valid_response_rate', 'completed_questions']
            st.dataframe(summary[[c for c in cols if c in summary.columns]], hide_index=True, width="stretch")
            if {'status_accuracy', 'evidence_f1'} <= set(summary.columns):
                chart_data = summary[['mode', 'status_accuracy', 'evidence_f1']].melt(
                    id_vars='mode', var_name='Метрика', value_name='Значение'
                )
                chart_data['Метрика'] = chart_data['Метрика'].map({
                    'status_accuracy': 'Точность статуса',
                    'evidence_f1': 'Evidence F1',
                })
                import altair as alt
                chart = alt.Chart(chart_data).mark_bar().encode(
                    x=alt.X('mode:N', title='Режим'),
                    xOffset='Метрика:N',
                    y=alt.Y('Значение:Q', title='Доля', scale=alt.Scale(domain=[0, 1])),
                    color=alt.Color('Метрика:N', title='Метрика'),
                    tooltip=['mode:N', 'Метрика:N', alt.Tooltip('Значение:Q', format='.1%')],
                )
                st.altair_chart(chart, width='stretch')
        st.markdown('#### По каждому вопросу')
        if metrics_path.is_file():
            metrics = pd.read_csv(metrics_path)
            st.dataframe(metrics, hide_index=True, width="stretch")
        with st.expander('Данные эксперимента'):
            manifest_path = selected_run / 'manifest.json'
            if manifest_path.exists():
                manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
                st.json({k: manifest.get(k) for k in ['created_at', 'model_config', 'requested_cases', 'completed_cases', 'valid_responses', 'system_prompt_sha256']})

with dossier:
    st.subheader('Документы и утверждения')
    selected_document = st.selectbox('Документ', dataset.documents, format_func=lambda d: f'{d.document_id} · {d.title}')
    st.markdown(f'**{selected_document.title}**')
    st.caption(f'{selected_document.document_id} · {selected_document.source_type} · {selected_document.author}')
    st.text_area('Текст документа', value=selected_document.text, height=380, disabled=True)
    st.markdown('#### Выделенные утверждения')
    relevant_claims = [c for c in dataset.claims if c.document_id == selected_document.document_id]
    for claim in relevant_claims:
        st.markdown(f'**{claim.claim_id}** · {claim.text}')
        st.caption('Цитата: ' + claim.evidence)

with stress:
    from datetime import datetime, timezone
    from src.experiment import digest, run_experiment, write_json

    st.subheader('Стресс-тесты')
    index_path = ROOT / 'stress/scenarios/index.json'
    if not index_path.exists():
        st.info('Сценарии не найдены')
    else:
        index = json.loads(index_path.read_text(encoding='utf-8'))
        scenarios = index.get('scenarios', [])
        manifests = {s['scenario_id']: json.loads((ROOT / 'stress/scenarios' / s['scenario_id'] / 'scenario.json').read_text(encoding='utf-8')) for s in scenarios}
        checked = sum(m.get('human_reviewed', False) for m in manifests.values())
        col1, col2, col3 = st.columns(3)
        col1.metric('Сценарии', len(scenarios))
        col2.metric('Пары «сценарий–вопрос»', index.get('question_cases', '—'))
        col3.metric('Проверено', checked)
        scenario = st.selectbox('Сценарий', scenarios, format_func=lambda s: f"{s['scenario_id']} · {KIND_LABELS.get(s.get('kind'), s.get('kind', ''))}")
        scenario_id = scenario['scenario_id']
        directory = ROOT / 'stress/scenarios' / scenario_id
        manifest_path = directory / 'scenario.json'
        manifest = manifests[scenario_id]
        st.markdown('**Изменение:** ' + (manifest.get('factor') or manifest.get('specification', {}).get('factor') or manifest.get('kind','')))
        st.markdown('**Эталонные статусы**')
        st.dataframe(pd.DataFrame([{'Вопрос': qid, 'Статус': STATUS_LABELS.get(status, status)} for qid, status in manifest['expected_statuses'].items()]), hide_index=True, width='stretch')
        if manifest['kind'] == 'permutation':
            a, b = st.columns(2)
            a.markdown('**Исходный порядок:** ' + ' → '.join(manifest['original_document_order']))
            b.markdown('**Новый порядок:** ' + ' → '.join(manifest['changed_document_order']))
        else:
            spec = manifest.get('specification', {})
            st.write('**Вопрос:** ' + next((q.question for q in dataset.questions if q.question_id == spec.get('question_id')), '—'))
            st.write('**Ожидаемый ответ:** ' + spec.get('gold_answer', '—'))
            patch_path = directory / 'changes.patch'
            if patch_path.exists():
                with st.expander('Изменения в документах', expanded=True):
                    st.code(patch_path.read_text(encoding='utf-8') or 'Изменений текста не найдено', language='diff')
            with st.expander('Изменения утверждений'):
                new_claims = [json.loads(line) for line in (directory / 'claims.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
                old_claims = {c.claim_id:c for c in dataset.claims}
                ids = {c['claim_id'] for c in new_claims}
                removed = [c for cid,c in old_claims.items() if cid not in ids]
                added = [c for c in new_claims if c['claim_id'] not in old_claims]
                for c in removed:
                    st.write('Удалено: ' + c.claim_id + ' — ' + c.text)
                for c in added:
                    st.write('Добавлено: ' + c['claim_id'] + ' — ' + c['text'])
                if not removed and not added:
                    st.write('Изменены только тексты документов либо существующие утверждения.')
        try:
            updated_dataset = load_dataset(directory, directory / 'claims_metadata.json')
            actual_hash = digest(updated_dataset.model_dump_json())
            intact = actual_hash == manifest.get('dataset_sha256')
            review_valid = bool(manifest.get('human_reviewed')) and manifest.get('reviewed_dataset_sha256') == actual_hash
        except Exception as exc:
            intact = review_valid = False
            st.error(f'Ошибка набора данных: {exc}')
        if not intact:
            st.error('Ошибка целостности сценария')
        elif not review_valid:
            if st.button('Подтвердить проверку', key=f'confirm_{scenario_id}', disabled=not intact):
                manifest.update(
                    human_reviewed=True,
                    reviewed_dataset_sha256=actual_hash,
                    review_source='Проверка оператором в интерфейсе',
                    reviewed_at=datetime.now(timezone.utc).isoformat(),
                )
                write_json(manifest_path, manifest)
                st.rerun()
        output = ROOT / 'results' / f'stress_{scenario_id}_gigachat'
        existing_manifest = output / 'manifest.json'
        completed = existing_manifest.exists() and json.loads(existing_manifest.read_text(encoding='utf-8')).get('completed', False)
        if completed:
            st.success('Эксперимент завершён')
            comparison_baseline = ROOT / 'results' / 'gigachat_baseline_v2'
            if (comparison_baseline / 'answers.jsonl').exists() and (output / 'answers.jsonl').exists():
                baseline_answers = get_answers(str(comparison_baseline))
                stress_answers = get_answers(str(output))
                summary_frame, detail_frame = stress_baseline_comparison(
                    baseline_answers, stress_answers,
                    {q.question_id: q.gold_status for q in dataset.questions},
                    manifest['expected_statuses'],
                )
                if not summary_frame.empty:
                    st.markdown('#### Сравнение с исходным экспериментом')
                    st.dataframe(summary_frame, hide_index=True, width='stretch')
                    total_changed = int(detail_frame['Статус изменился'].sum())
                    total_answers = len(detail_frame)
                    if manifest['kind'] == 'permutation':
                        if total_changed == 0:
                            st.write(f'**Вывод:** после перестановки статусы не изменились ни в одном из {total_answers} ответов.')
                        else:
                            st.write(f'**Вывод:** после перестановки изменились статусы в {total_changed} из {total_answers} ответов.')
                    st.markdown('#### По каждому вопросу: до и после')
                    st.dataframe(detail_frame, hide_index=True, width='stretch')
                else:
                    st.warning('Нет сопоставимых ответов в исходном и изменённом экспериментах.')
            else:
                st.warning('Для сравнения нужны файлы answers.jsonl исходного и изменённого экспериментов.')
            with st.expander('Метрики стресс-эксперимента'):
                metrics_path = output / 'per_question_metrics.csv'
                if metrics_path.exists():
                    st.dataframe(pd.read_csv(metrics_path), hide_index=True, width='stretch')
        else:
            if output.exists() and any(output.iterdir()):
                st.error('Обнаружены незавершённые результаты эксперимента')
            else:
                n_calls = len(updated_dataset.questions)*3 if intact else 0
                if st.button(f'Запустить эксперимент · {n_calls} запросов', type='primary', disabled=not (review_valid and config_path.is_file())):
                    try:
                        with st.spinner('Выполняем запросы к GigaChat и сохраняем результаты'):
                            run_experiment(output, config_path, directory, directory / 'claims_metadata.json', execute=True, scenario_manifest=manifest_path)
                        st.success('Сценарий завершён. Итоги записаны в ' + str(output.relative_to(ROOT)))
                        st.rerun()
                    except Exception as exc:
                        st.error(f'Ошибка запуска: {type(exc).__name__}: {exc}')
