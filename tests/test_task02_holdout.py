import json
from pathlib import Path


def test_manual_holdout_has_required_size_and_categories() -> None:
    path = Path(__file__).parent / "fixtures" / "task02_manual_holdout.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) >= 150
    assert len({row["case_id"] for row in rows}) == len(rows)
    required = {
        "valid",
        "sentence_copy",
        "unanswerable_similar",
        "wrong_number",
        "wrong_entity",
        "too_general",
        "other_sentence",
        "answer_leak",
        "judge_disagreement_candidate",
    }
    assert required <= {row["category"] for row in rows}
    assert all(row["human_answerability"] in (0, 1) for row in rows)
