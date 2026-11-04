"""Проверка подготовленных стресс-сценариев перед пакетным запуском GigaChat.

Поместить в AOT_6/scripts/review_remaining_stress.py.
Без --review: только проверяет целостность и показывает список.
С --review: показывает изменения всех ещё не подтверждённых сценариев
и записывает подтверждения ТОЛЬКО после одного явного ответа человека "да".
Запросов к GigaChat не делает.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.schemas import load_dataset  # noqa: E402

SCENARIOS = ROOT / "stress" / "scenarios"

KIND = {
    "permutation": "перестановка",
    "remove_premise": "удаление посылки",
    "add_contradiction": "добавление противоречия",
    "add_unverified_control": "контроль неподтверждённого сообщения",
}
RU_STATUS = {
    "entailed": "Подтверждено",
    "contradicted": "Опровергнуто",
    "both": "Есть противоречие",
    "insufficient": "Недостаточно данных",
}


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def fail(message: str) -> None:
    raise ValueError(message)


def sha(dataset) -> str:
    return hashlib.sha256(dataset.model_dump_json().encode("utf-8")).hexdigest()


def id_map(objects, field: str) -> dict:
    return {getattr(obj, field): obj.model_dump(mode="json") for obj in objects}


def verify_scenario(base, directory: Path):
    manifest = load_json(directory / "scenario.json")
    variant = load_dataset(directory, directory / "claims_metadata.json")
    sid = directory.name
    if manifest.get("scenario_id") != sid or sha(variant) != manifest.get("dataset_sha256"):
        fail(f"{sid}: идентификатор или контрольная сумма не совпадают")
    got = {q.question_id: q.gold_status for q in variant.questions}
    if got != manifest.get("expected_statuses"):
        fail(f"{sid}: эталоны не совпадают с описанием сценария")

    kind = manifest["kind"]
    bdocs = id_map(base.documents, "document_id")
    vdocs = id_map(variant.documents, "document_id")
    bclaims = id_map(base.claims, "claim_id")
    vclaims = id_map(variant.claims, "claim_id")
    bmeta = {k: v.model_dump(mode="json") for k, v in base.metadata.items()}
    vmeta = {k: v.model_dump(mode="json") for k, v in variant.metadata.items()}

    if kind == "permutation":
        if bdocs != vdocs or bclaims != vclaims or bmeta != vmeta or base.questions != variant.questions:
            fail(f"{sid}: кроме порядка документов изменились и другие данные")
        if [d.document_id for d in base.documents] == [d.document_id for d in variant.documents]:
            fail(f"{sid}: порядок документов не изменился")
    elif kind == "remove_premise":
        spec = manifest["specification"]
        removed = set(spec["removed_claim_ids"])
        if removed != set(bclaims) - set(vclaims):
            fail(f"{sid}: список удалённых утверждений неверный")
        if set(bclaims) - removed != set(vclaims):
            fail(f"{sid}: неожиданно добавились утверждения")
        expected = {k: dict(v) for k, v in bclaims.items() if k not in removed}
        for claim_id, evidence in spec.get("updated_evidence", {}).items():
            expected[claim_id]["evidence"] = evidence
        if expected != vclaims or {k: v for k, v in bmeta.items() if k not in removed} != vmeta:
            fail(f"{sid}: помимо описанных удалений изменились утверждения или метаданные")
        expected_docs = {k: dict(v) for k, v in bdocs.items()}
        for did, substitutions in spec.get("replacements", {}).items():
            for old, new in substitutions:
                if old not in expected_docs[did]["text"]:
                    fail(f"{sid}: не найден исходный фрагмент: {did}")
                expected_docs[did]["text"] = expected_docs[did]["text"].replace(old, new)
        if expected_docs != vdocs:
            fail(f"{sid}: документы изменены не так, как описано в сценарии")
    elif kind in {"add_contradiction", "add_unverified_control"}:
        if not (set(vdocs) - set(bdocs) == {"D11"} and set(vclaims) - set(bclaims) == {"C41"}):
            fail(f"{sid}: должно добавиться ровно D11 и C41")
        if {k: v for k, v in vdocs.items() if k != "D11"} != bdocs:
            fail(f"{sid}: вместе с D11 изменились исходные документы")
        if {k: v for k, v in vclaims.items() if k != "C41"} != bclaims:
            fail(f"{sid}: вместе с C41 изменились исходные утверждения")
        if {k: v for k, v in vmeta.items() if k != "C41"} != bmeta:
            fail(f"{sid}: вместе с C41 изменились метаданные старых утверждений")
        if vclaims["C41"]["evidence"] not in vdocs["D11"]["text"]:
            fail(f"{sid}: цитата C41 отсутствует в новом D11")
    else:
        fail(f"{sid}: неизвестный тип сценария: {kind}")

    if kind != "permutation":
        if len(variant.questions) != 1:
            fail(f"{sid}: ожидается ровно один проверяемый вопрос")
        before = {q.question_id for q in base.questions}
        if variant.questions[0].question_id not in before:
            fail(f"{sid}: неизвестный вопрос")
    return manifest, variant


def print_details(base, manifest: dict, variant) -> None:
    sid = manifest["scenario_id"]
    print("\n" + "=" * 70)
    print(f"{sid} — {KIND.get(manifest['kind'], manifest['kind'])}")
    original = {q.question_id: q for q in base.questions}
    for question in variant.questions:
        o = original[question.question_id]
        print(f"{question.question_id}: {o.gold_status} → {question.gold_status}")
        print("Вопрос:", question.question)
        print("Ожидаемый ответ:", question.gold_answer)
        print("Ожидаемые доказательства:", question.gold_evidence_ids)
        print("Контрдоказательства:", question.gold_counterevidence_ids)
    spec = manifest.get("specification", {})
    if manifest["kind"] == "permutation":
        print("Изменён порядок:", " → ".join(manifest.get("changed_document_order", [])))
    elif manifest["kind"] == "remove_premise":
        print("Удалены утверждения:", ", ".join(spec.get("removed_claim_ids", [])))
        for did, swaps in spec.get("replacements", {}).items():
            print("  ", did)
            for old, new in swaps:
                print("    -", old)
                print("    +", new or "[удалено]")
    else:
        print("Добавленный документ D11:", spec.get("new_text", "нет в описании"))
        print("Добавленное утверждение C41:", spec.get("claim_text", "нет в описании"))
    if sid == "contradiction_Q09":
        print("ПРИМЕЧАНИЕ: Q09 уже имел статус BOTH. Это проверка сохранения конфликта, не перехода к BOTH.")
    if manifest["kind"] == "add_unverified_control":
        print("ПРИМЕЧАНИЕ: это контрольный сценарий, а не тест создания противоречия.")
    print("Автоматическая проверка структуры: пройдена. Смысл эталона требуется оценить самостоятельно.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--review", action="store_true", help="После просмотра каждой правки предложить её подтвердить")
    args = parser.parse_args()

    base = load_dataset(ROOT / "docs", ROOT / "config" / "claims_metadata.json")
    index = load_json(SCENARIOS / "index.json")
    done = 0
    waiting = []
    for entry in index["scenarios"]:
        directory = SCENARIOS / entry["scenario_id"]
        manifest, variant = verify_scenario(base, directory)
        valid_review = bool(manifest.get("human_reviewed")) and manifest.get("reviewed_dataset_sha256") == sha(variant)
        if valid_review:
            done += 1
            print("Уже подтверждён:", entry["scenario_id"])
            continue
        waiting.append((directory, manifest, variant))
        if args.review:
            print_details(base, manifest, variant)
        else:
            print("Ожидает проверки:", entry["scenario_id"], "—", KIND.get(manifest["kind"], manifest["kind"]))

    print(f"\nВсего: {len(index['scenarios'])}; ранее подтверждено: {done}; осталось проверить: {len(waiting)}.")
    if not args.review:
        print("Для просмотра и подтверждения: python -X utf8 scripts/review_remaining_stress.py --review")
        return
    if not waiting:
        print("Все сценарии уже подтверждены.")
        return
    print("\nПодтверждение не означает, что ответы GigaChat будут правильными.")
    print("Оно означает только, что вы лично проверили изменения и ожидаемые эталоны.")
    choice = input(f"Вы прочитали и согласовали ВСЕ {len(waiting)} показанных сценариев? [да/нет]: ").strip().casefold()
    if choice not in ("да", "д", "yes", "y"):
        print("Подтверждения не сохранены. Сценарии остались без ручной проверки.")
        return
    for directory, manifest, variant in waiting:
        manifest.update({
            "human_reviewed": True,
            "reviewed_dataset_sha256": sha(variant),
            "reviewed_at": datetime.now(timezone.utc).isoformat(),
            "review_method": "interactive_terminal_bulk_confirmation",
        })
        (directory / "scenario.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Подтверждено сейчас: {len(waiting)}.\nСледующий шаг: python -X utf8 scripts/stress_batch.py --run")


if __name__ == "__main__":
    main()
