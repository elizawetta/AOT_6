# Передача функций для стенда и тестов

В корне проекта пока нет app.py: стенд ещё предстоит сделать.

## Загрузка

```python
from src.schemas import load_dataset

dataset = load_dataset('docs', 'config/claims_metadata.json')
question = next(q for q in dataset.questions if q.question_id == 'Q01')
```

Dataset содержит documents, claims, questions и metadata. Загрузка проверяет уникальность идентификаторов, источники цитат и ссылки эталонов. Строки JSON читаются Pydantic со строгими полями. Не загружать авторский файл истины и Markdown-дубликаты в модель.

## Retrieval и ветви

```python
from src.retrieval import ClaimRetriever
from src.context_builder import build_context, build_context_pair
from src.padic import encode, decode, distance, group_claims

hits = ClaimRetriever(dataset.claims).search(question.question)
flat, grouped = build_context_pair(dataset, question.question)
full = build_context(dataset, question.question, 'A')
branches = group_claims([hit.claim for hit in hits], dataset.metadata)
```

SearchHit содержит claim и score. BuiltContext содержит mode, text, claim_ids, document_ids и retrieval_scores. document_ids — порядок полного исходного набора, а не список полностью переданных документов в B/C. В этих режимах передаются только записи утверждений.

distance принимает два трёхзначных пути и возвращает точную Fraction. Числовые коды можно преобразовать decode перед расчётом расстояния. Полное совпадение категорий даёт 0, даже если claim_id разные.

## Контракт API и пример локального клиента

```python
import json
from pathlib import Path
from src.llm import create_client
from src.schemas import validate_answer

project_root = Path.cwd()
client = create_client(project_root / 'config/model.example.json', project_root)
context = build_context(dataset, question.question, 'B')
system = (project_root / 'prompts/system.txt').read_text()
from src.experiment import prepare_cases
case = next((q, c, user) for q, c, user in prepare_cases(dataset)
            if q.question_id == question.question_id and c.mode == context.mode)
generation = client.generate(system, case[2])
answer = validate_answer(generation.raw_text, question.question_id, set(context.claim_ids))
```

prepare_cases — общий шаблон фактического запроса; стенд должен использовать его, чтобы не создавать четвёртый формат. Для режима C показывать ветви с тем же набором hits. Клиент модели можно сохранять в ресурсе сессии; при завершении вызвать close, если метод существует. Вызовы одного MLXProcessClient должны быть последовательными: worker не поддерживает конкурентные запросы. Ошибки LLMUnavailable и ValidationError нужно отображать отдельно от содержательно неверного ответа.

Для API нужно реализовать клиента с методом generate(system, user), возвращающим src.llm.Generation: raw_text, input_tokens, output_tokens, elapsed_seconds, token_measurement. Учёт токенов берётся из фактического usage API. Новый backend подключается в create_client и фиксируется в конфигурации с именем модели и параметрами; секреты в конфигурационный JSON не включаются. Существующий клиент MLX выше — предварительный локальный пример, а не реализация API.

Основной baseline и все стресс-варианты нужно выполнить заново через одну выбранную модель API с одинаковыми настройками. Не сравнивать API-стресс с локальным baseline Qwen и не менять промпт между режимами.

Для сохранения полного воспроизводимого эксперимента вызывается `src.experiment.run_experiment`. Для сравнения сохранённых ответов предусмотрены `src.evaluation.summarize` и `compare_runs`. Исходный ответ сохранять до ручной оценки и не заменять исправленным текстом.

## Изменения данных

Функции находятся в src.stress: permute_documents, remove_premise, add_contradiction. Они возвращают отдельный Dataset. Подготовленные случаи лежат в stress/scenarios; загрузка такая же, но metadata берётся из папки варианта. Удалять только строку Cxx в интерфейсе недостаточно для режима A: факт может оставаться в документе или в производном выводе. Для исследовательских прогонов использовать проверенные варианты со scenario.json и хешем проверки.

## Обязательные проверки

Покрыть одинаковые пути и все уровни различия, ультраметрическое неравенство, невалидный JSON, неизвестный или не переданный evidence_id, both без контрдоказательств, недоступность модели, ограничение top-15 и равенство содержимого B/C. Для insufficient с выдуманным фактом требуется содержательная проверка; текущий validate_answer проверяет форму и ссылки, а не истинность произвольного русского текста. Нельзя выдать проверку JSON за детектор любых галлюцинаций.

Результаты разовой проверки алгоритмов находятся в results/core_verification.json. Это не замена вашему pytest-набору. Полный набор тестов и проверка интерфейса ещё не выполнены.

## Готовый комплект стресс-сценариев

Подключать stress/scenarios/index.json; он содержит 17 наборов и 28 пар «вариант–вопрос». Для трёх режимов это 84 стресс-запроса, плюс 36 исходных запросов. Полная сводка — stress/scenarios/README.md; каждый набор содержит свои данные, эталон, метаданные, manifest и changes.patch. Общий замороженный исходный набор сохранён в stress/scenarios/_baseline.

Для Q11–Q12 добавлены отдельные add_unverified_control с сохранением insufficient. Они не входят в показатель перехода к both. Q09 изначально both; его добавление проверяет сохранение конфликта, а не появление нового статуса. Удаление применимо только к Q03–Q06 по явному условию задания. Неприменимые пары не считать успешными тестами.

Проверка ассистентом завершена, отметка проверки человеком намеренно не выставлена. До запуска получить фактическую проверку изменённых эталонов и сохранить её через scripts/review_scenario.py. Подтверждение фиксируется по хешу набора; исходные данные в docs не менять.
