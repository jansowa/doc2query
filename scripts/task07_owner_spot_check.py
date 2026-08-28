#!/usr/bin/env python3
"""Ślepy spot-check właściciela na 50 parach v3 — kontrola operacyjna, nie panel.

Amendment `task06_v3_groq_role_amendment_2026-08-27.md` §2.3 przewiduje przed
treningiem ślepą kontrolę 50 par z rzeczywistego rozkładu. **Nie jest to panel
AGENTS.md §9.3** i nie wolno tego tak raportować: to sanity check, nie evidence.

Dwa tryby:

* `export` — losuje próbkę zamrożonym seedem, zapisuje arkusz Markdown, w którym
  strony są **losowo** przypisane do A i B, oraz osobny plik klucza. Arkusz nie
  zawiera niczego, co ujawnia stronę wybraną przez selektor: ani kolejności, ani
  identyfikatorów kandydatów, ani wyniku głosowania.
* `score` — czyta wypełnione odpowiedzi (`<numer><tab lub spacja><A|B|=>`) i
  liczy zgodność z selektorem wraz z dokładnym dwustronnym przedziałem
  Cloppera-Pearsona. Nie ma tu żadnego progu: amendment nie zamroził bramki, a
  dopisanie progu po zobaczeniu wyniku byłoby dorabianiem kryterium.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from pathlib import Path
from typing import Any

from doc2query.preferences.defect_pair_gate_v2_1 import clopper_pearson_bounds
from doc2query.utils.records import read_records, write_json

SHEET_NAME = "spot_check_sheet.md"
KEY_NAME = "spot_check_key.json"
ANSWER_NAME = "spot_check_answers.txt"
ANSWER_PATTERN = re.compile(r"^\s*(\d+)\s*[\t ,:;]\s*([ABab=])\s*$")


def _side_for_a(preference_id: str, seed: int) -> str:
    """Deterministyczne, niezależne od kolejności losowanie, która strona jest A."""
    digest = hashlib.sha256(f"{seed}:spot-check-side:{preference_id}".encode()).digest()
    return "chosen" if digest[0] % 2 == 0 else "rejected"


def _passage(prompt: str) -> str:
    marker = "Pasaż:\n"
    if marker not in prompt:
        return prompt.strip()
    tail = prompt.split(marker, 1)[1]
    return tail.split("\n\nZapytanie:", 1)[0].strip()


def _export(args: argparse.Namespace) -> None:
    if args.output_dir.exists():
        raise SystemExit(f"wyjście już istnieje: {args.output_dir}")
    rows: list[dict[str, Any]] = []
    for path in args.pairs:
        rows.extend(dict(row) for row in read_records(path))
    if len(rows) < args.count:
        raise SystemExit(f"za mało par: {len(rows)} < {args.count}")
    rows.sort(key=lambda row: str(row["preference_id"]))
    sample = random.Random(args.seed).sample(rows, args.count)

    lines = [
        "# Ślepy spot-check 50 par (kontrola operacyjna, nie panel §9.3)",
        "",
        "Dla każdej pozycji: przeczytaj pasaż i dwa zapytania. Zapisz w",
        f"`{ANSWER_NAME}` numer i literę zapytania, które **lepiej nadaje się jako",
        "zapytanie wyszukiwawcze do tego pasażu** (ugruntowane, odpowiadalne, nie",
        "kopiujące pasażu, nie zdradzające odpowiedzi). `=` oznacza brak różnicy.",
        "",
        "Przykład wiersza odpowiedzi: `7 B`",
        "",
    ]
    key: list[dict[str, Any]] = []
    for index, row in enumerate(sample, start=1):
        preference_id = str(row["preference_id"])
        a_side = _side_for_a(preference_id, args.seed)
        b_side = "rejected" if a_side == "chosen" else "chosen"
        lines += [
            f"## {index}",
            "",
            f"**Pasaż:** {_passage(str(row['prompt']))}",
            "",
            f"- **A:** {row[a_side]}",
            f"- **B:** {row[b_side]}",
            "",
        ]
        key.append(
            {
                "item": index,
                "preference_id": preference_id,
                "a_side": a_side,
                "b_side": b_side,
                "chosen_letter": "A" if a_side == "chosen" else "B",
            }
        )

    args.output_dir.mkdir(parents=True)
    (args.output_dir / SHEET_NAME).write_text("\n".join(lines) + "\n", encoding="utf-8")
    (args.output_dir / ANSWER_NAME).write_text(
        "# jeden wiersz na pozycję: numer i litera (A/B/=)\n", encoding="utf-8"
    )
    write_json(
        args.output_dir / KEY_NAME,
        {
            "schema_version": 1,
            "contract": "task07-owner-spot-check-v1",
            "role": "operational_sanity_check_not_panel_9_3",
            "seed": args.seed,
            "count": args.count,
            "sources": [str(path) for path in args.pairs],
            "items": key,
            "final_tests_used": [],
        },
    )
    print(
        json.dumps(
            {
                "sheet": str(args.output_dir / SHEET_NAME),
                "answers_template": str(args.output_dir / ANSWER_NAME),
                "key": str(args.output_dir / KEY_NAME),
                "count": args.count,
            },
            indent=2,
            sort_keys=True,
        )
    )


def _score(args: argparse.Namespace) -> None:
    key = json.loads((args.output_dir / KEY_NAME).read_text(encoding="utf-8"))
    letters: dict[int, str] = {}
    for line in (args.output_dir / ANSWER_NAME).read_text(encoding="utf-8").split("\n"):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        match = ANSWER_PATTERN.match(line)
        if match is None:
            raise SystemExit(f"nie rozumiem wiersza odpowiedzi: {line!r}")
        item = int(match.group(1))
        if item in letters:
            raise SystemExit(f"pozycja {item} ma dwie odpowiedzi")
        letters[item] = match.group(2).upper()
    expected = {int(row["item"]) for row in key["items"]}
    unknown = sorted(set(letters) - expected)
    if unknown:
        raise SystemExit(f"odpowiedzi do nieistniejących pozycji: {unknown}")

    agree = ties = disagree = 0
    disagreements: list[int] = []
    for row in key["items"]:
        letter = letters.get(int(row["item"]))
        if letter is None:
            continue
        if letter == "=":
            ties += 1
        elif letter == row["chosen_letter"]:
            agree += 1
        else:
            disagree += 1
            disagreements.append(int(row["item"]))
    decided = agree + disagree
    # Dwustronny 95%: każda granica jednostronnie na 0,025.
    low, high = clopper_pearson_bounds(agree, decided, 0.025) if decided else (0.0, 1.0)
    summary = {
        "schema_version": 1,
        "contract": "task07-owner-spot-check-result-v1",
        "role": "operational_sanity_check_not_panel_9_3",
        "answered": len(letters),
        "count": int(key["count"]),
        "agreements": agree,
        "disagreements": disagree,
        "ties": ties,
        "decided": decided,
        "agreement_rate": round(agree / decided, 4) if decided else None,
        "agreement_ci95": [round(low, 4), round(high, 4)],
        "disagreement_items": disagreements,
        "no_frozen_threshold": True,
        "final_tests_used": [],
    }
    write_json(args.output_dir / "spot_check_result.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("export", "score"))
    parser.add_argument(
        "--pairs",
        type=Path,
        action="append",
        default=None,
        help="pliki par; domyślnie spakowany train i dev handoffu v3",
    )
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/task07/handoff_v3_bottom/owner_spot_check"),
    )
    args = parser.parse_args()
    if args.pairs is None:
        packaged = Path("artifacts/task07/handoff_v3_bottom/packaged")
        args.pairs = [packaged / "preference_train.jsonl", packaged / "preference_dev.jsonl"]
    if args.mode == "export":
        _export(args)
    else:
        _score(args)


if __name__ == "__main__":
    main()
