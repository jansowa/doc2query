"""Polityka par v3: turniej sędziego jako selektor, z podziałem na trzy etapy.

Kontrakt zamraża ADR `reports/decisions/task06_judge_selected_pair_policy_v3.md`, a
próg agregacji amendment `task06_v3_selector_aggregation_amendment_2026-08-27.md`
(**jednomyślność 6/6**, zakres ważności: ugruntowanie, kopiowanie, ogólność).

Podział na etapy wynika z tego, że serwer sędziego stoi na innej maszynie niż
artefakty kohort:

1. **eksport pakietu** (tutaj) — chudy plik z pasażem i kandydatami, bez score'ów i
   bez niczego, co zdradza wybór automatu; 380 MB artefaktów zostaje na miejscu;
2. **turniej** (na maszynie z serwerem) — porównania parami, wynik do journala;
3. **złożenie par** (tutaj) — dołączenie pełnej proweniencji, fingerprintów i
   guardów, z werdyktów odczytanych z journala.

Journal etapu 2 jest **cache'em porównań** kluczowanym parą kandydatów, nie numerem
kroku. Dzięki temu turniej jest deterministyczną funkcją nad cache'em: wznowienie
odtwarza przebieg bez ani jednego zbędnego wywołania, a powtórny run jest darmowy.
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from doc2query.preferences.pair_selector_v3 import (
    RANKING_RUBRIC,
    RUBRICS,
    JudgeApiError,
    JudgeEndpoint,
    PairwiseItem,
    Transport,
    http_transport,
    journal_key,
    load_journal,
)
from doc2query.preferences.pair_selector_v3 import (
    _append as _append_journal,
)
from doc2query.utils.records import read_records, write_json

BUNDLE_CONTRACT = "task06-v3-tournament-bundle-v1"
TOURNAMENT_CONTRACT = "task06-v3-tournament-v1"
# Zamrożone amendmentem: sześć zgodnych głosów z sześciu (3 rubryki x 2 kolejności).
REQUIRED_UNANIMOUS_VOTES = 6
RejectedVariant = Literal["bottom", "near_miss"]


@dataclass(frozen=True)
class BundleCandidate:
    candidate_id: str
    candidate_index: int
    query: str
    admissible_as_chosen: bool
    admissible_as_rejected: bool

    def __post_init__(self) -> None:
        # Czystość zawiera w sobie dopuszczalność formatem, więc „czysty, ale
        # niedopuszczalny formatem" jest sprzecznością i oznacza błąd eksportu.
        if self.admissible_as_chosen and not self.admissible_as_rejected:
            raise ValueError(
                f"{self.candidate_id}: kandydat czysty musi być dopuszczalny formatem"
            )


@dataclass(frozen=True)
class BundleGroup:
    """Jedna grupa same-prompt gotowa do turnieju; bez score'ów i bez etykiet."""

    group_id: str
    cohort_id: str
    passage: str
    candidates: tuple[BundleCandidate, ...]

    def _sorted(self, rows: list[BundleCandidate]) -> list[BundleCandidate]:
        return sorted(rows, key=lambda row: (row.candidate_index, row.candidate_id))

    def chosen_pool(self) -> list[BundleCandidate]:
        """Kandydaci spełniający PEŁNY kontrakt czystości — tylko z nich może być chosen.

        ADR §5 wymaga po stronie `chosen` formatu, round-tripu @20 i braku `copy_risk`.
        Rankingowanie lidera po szerszej puli oznaczałoby, że para wypada dopiero przy
        składaniu, już po wydaniu wywołań.
        """
        return self._sorted([row for row in self.candidates if row.admissible_as_chosen])

    def rejected_pool(self) -> list[BundleCandidate]:
        """Kandydaci dopuszczalni formatem; strona `rejected` nie wymaga czystości."""
        return self._sorted([row for row in self.candidates if row.admissible_as_rejected])

    def ranked_pool(self) -> list[BundleCandidate]:
        """Zachowane dla raportowania planu: pula, w której cokolwiek się rozgrywa."""
        return self.rejected_pool()


def load_bundle(path: Path) -> list[BundleGroup]:
    """Odczytaj pakiet turniejowy; kolejność grup jest częścią kontraktu."""
    groups: list[BundleGroup] = []
    for row in read_records(path):
        groups.append(
            BundleGroup(
                group_id=str(row["group_id"]),
                cohort_id=str(row["cohort_id"]),
                passage=str(row["passage"]),
                candidates=tuple(
                    BundleCandidate(
                        candidate_id=str(item["candidate_id"]),
                        candidate_index=int(item["candidate_index"]),
                        query=str(item["query"]),
                        admissible_as_chosen=bool(item["admissible_as_chosen"]),
                        admissible_as_rejected=bool(item["admissible_as_rejected"]),
                    )
                    for item in row["candidates"]
                ),
            )
        )
    if not groups:
        raise ValueError(f"{path}: pakiet turniejowy jest pusty")
    return groups


def comparison_item(
    group: BundleGroup, first: BundleCandidate, second: BundleCandidate
) -> PairwiseItem:
    """Zbuduj porównanie w kanonicznej kolejności (pierwszy, drugi)."""
    return PairwiseItem(
        item_id=f"{group.group_id}|{first.candidate_id}|{second.candidate_id}",
        passage=group.passage,
        query_first=first.query,
        query_second=second.query,
        metadata={"group_id": group.group_id, "cohort_id": group.cohort_id},
    )


Comparator = Callable[[BundleGroup, BundleCandidate, BundleCandidate], str]


def _pairwise_winner(votes: Mapping[str, str]) -> str:
    """Zgoda obu kolejności albo `tie`; rozbieżność pozycyjna nie wygrywa niczego."""
    forward = votes.get("ab")
    reverse = votes.get("ba")
    if forward is None or reverse is None or forward != reverse:
        return "tie"
    return forward


def run_group_tournament(
    group: BundleGroup, compare: Comparator
) -> dict[str, Any]:
    """Wyłoń najlepszego, najgorszego i drugiego od końca; remisy nie przesuwają lidera.

    Turniej jest pojedynczą eliminacją po deterministycznej kolejności kandydatów, więc
    liczba porównań to `k-1` na lidera, `k-2` na najgorszego i `k-3` na drugiego od
    końca. Przy remisie utrzymuje się dotychczasowy lider — inaczej wynik zależałby od
    kolejności losowo.
    """
    chosen_pool = group.chosen_pool()
    rejected_pool = group.rejected_pool()
    if not chosen_pool:
        # Grupa bez ani jednego czystego kandydata nie może dać pary; turniej na niej
        # byłby wydatkiem bez możliwego wyniku.
        return {
            "group_id": group.group_id,
            "cohort_id": group.cohort_id,
            "paired": False,
            "reason": "no_admissible_chosen",
        }
    if len(rejected_pool) < 2:
        return {
            "group_id": group.group_id,
            "cohort_id": group.cohort_id,
            "paired": False,
            "reason": "too_few_admissible",
        }
    comparisons = 0

    def duel(left: BundleCandidate, right: BundleCandidate) -> str:
        nonlocal comparisons
        comparisons += 1
        return compare(group, left, right)

    best = chosen_pool[0]
    for challenger in chosen_pool[1:]:
        if duel(best, challenger) == "B":
            best = challenger

    rest = [row for row in rejected_pool if row.candidate_id != best.candidate_id]
    if not rest:
        return {
            "group_id": group.group_id,
            "cohort_id": group.cohort_id,
            "paired": False,
            "reason": "no_rejected_candidate",
        }
    worst = rest[0]
    for challenger in rest[1:]:
        if duel(worst, challenger) == "A":
            worst = challenger

    remaining = [row for row in rest if row.candidate_id != worst.candidate_id]
    second_worst: BundleCandidate | None = None
    if remaining:
        second_worst = remaining[0]
        for challenger in remaining[1:]:
            if duel(second_worst, challenger) == "A":
                second_worst = challenger

    return {
        "group_id": group.group_id,
        "cohort_id": group.cohort_id,
        "paired": True,
        "comparisons": comparisons,
        "best": best.candidate_id,
        "worst": worst.candidate_id,
        "second_worst": second_worst.candidate_id if second_worst else None,
        "chosen_pool_size": len(chosen_pool),
        "rejected_pool_size": len(rejected_pool),
    }


def confirmation_votes(votes_by_rubric: Mapping[str, Mapping[str, str]]) -> dict[str, Any]:
    """Policz głosy potwierdzenia finałowej pary; próg 6/6 z amendmentu."""
    for rubric in RUBRICS:
        if rubric not in votes_by_rubric:
            return {
                "votes_for_first": 0,
                "votes_for_second": 0,
                "unanimous": False,
                "complete": False,
            }
    first = second = 0
    flips = 0
    for rubric in RUBRICS:
        winner = _pairwise_winner(votes_by_rubric[rubric])
        if winner == "A":
            first += 2
        elif winner == "B":
            second += 2
        else:
            flips += 1
    return {
        "votes_for_first": first,
        "votes_for_second": second,
        "position_flips": flips,
        "complete": True,
        "unanimous": first >= REQUIRED_UNANIMOUS_VOTES,
        "unanimous_against": second >= REQUIRED_UNANIMOUS_VOTES,
    }


class JudgeCache:
    """Cache porównań oparty na journalu: klucz to para kandydatów, nie numer kroku."""

    def __init__(
        self,
        journal_path: Path,
        endpoint: JudgeEndpoint,
        transport: Transport | None = None,
    ) -> None:
        self.journal_path = journal_path
        self.endpoint = endpoint
        self.call = transport or http_transport(endpoint)
        self.lock = threading.Lock()
        self.rows = load_journal(journal_path)
        self.calls_made = 0
        self.failures = 0

    def verdicts(self, item: PairwiseItem, rubric: str) -> dict[str, str]:
        """Zwróć werdykty obu kolejności, dociągając brakujące."""
        result: dict[str, str] = {}
        for order in ("ab", "ba"):
            key = journal_key(item.item_id, rubric, order)
            with self.lock:
                cached = self.rows.get(key)
            if cached is not None:
                result[order] = str(cached["canonical_verdict"])
                continue
            row = self._judge(item, rubric, order, key)
            if row is None:
                continue
            result[order] = str(row["canonical_verdict"])
        return result

    def _judge(
        self, item: PairwiseItem, rubric: str, order: str, key: str
    ) -> dict[str, Any] | None:
        from typing import cast as _cast

        from doc2query.preferences.pair_selector_v3 import (
            _canonical,
            chat_payload,
            parse_verdict,
        )

        payload = chat_payload(item, rubric, _cast(Any, order), self.endpoint)
        try:
            response = self.call(payload)
            choices = list(response.get("choices") or [])
            if not choices:
                raise ValueError("odpowiedź bez pola choices")
            content = choices[0]["message"]["content"]
            verdict, confidence = parse_verdict(str(content))
        except (JudgeApiError, ValueError, KeyError, TypeError) as exc:
            with self.lock:
                self.failures += 1
                _append_journal(
                    self.journal_path,
                    {"event": "failure", "key": key, "reason": str(exc)[:500]},
                )
            return None
        row = {
            "event": "judgment",
            "key": key,
            "item_id": item.item_id,
            "rubric": rubric,
            "order": order,
            "verdict": verdict,
            "canonical_verdict": _canonical(verdict, _cast(Any, order)),
            "confidence": confidence,
            "metadata": dict(item.metadata),
        }
        with self.lock:
            _append_journal(self.journal_path, row)
            self.rows[key] = row
            self.calls_made += 1
        return row


def run_tournaments(
    *,
    groups: Sequence[BundleGroup],
    endpoint: JudgeEndpoint,
    journal_path: Path,
    output_path: Path,
    transport: Transport | None = None,
    concurrency: int = 8,
    progress_every: int = 25,
) -> dict[str, Any]:
    """Rozegraj turnieje wszystkich grup; równolegle po grupach, sekwencyjnie w grupie."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    cache = JudgeCache(journal_path, endpoint, transport)
    started = time.perf_counter()

    def ranking_compare(
        group: BundleGroup, left: BundleCandidate, right: BundleCandidate
    ) -> str:
        item = comparison_item(group, left, right)
        return _pairwise_winner(cache.verdicts(item, RANKING_RUBRIC))

    def process(group: BundleGroup) -> dict[str, Any]:
        outcome = run_group_tournament(group, ranking_compare)
        if not outcome.get("paired"):
            return outcome
        by_id = {row.candidate_id: row for row in group.candidates}
        best = by_id[str(outcome["best"])]
        confirmations: dict[str, Any] = {}
        for variant, candidate_id in (
            ("bottom", outcome["worst"]),
            ("near_miss", outcome["second_worst"]),
        ):
            if candidate_id is None:
                continue
            rejected = by_id[str(candidate_id)]
            item = comparison_item(group, best, rejected)
            votes = {rubric: cache.verdicts(item, rubric) for rubric in RUBRICS}
            confirmations[variant] = confirmation_votes(votes) | {
                "rejected_candidate_id": rejected.candidate_id
            }
        outcome["confirmations"] = confirmations
        return outcome

    outcomes: list[dict[str, Any]] = []
    pool = ThreadPoolExecutor(max_workers=max(1, concurrency))
    try:
        futures = [pool.submit(process, group) for group in groups]
        for index, future in enumerate(as_completed(futures), start=1):
            outcomes.append(future.result())
            if progress_every and index % progress_every == 0:
                elapsed = time.perf_counter() - started
                rate = cache.calls_made / elapsed if elapsed else 0.0
                print(
                    f"[turniej] {index}/{len(groups)} grup | {cache.calls_made} wywołań, "
                    f"{rate:.2f}/s | porażki {cache.failures}",
                    flush=True,
                )
    except BaseException:
        for future in futures:
            future.cancel()
        pool.shutdown(wait=False)
        raise
    else:
        pool.shutdown(wait=True)

    outcomes.sort(key=lambda row: str(row["group_id"]))
    with output_path.open("w", encoding="utf-8") as handle:
        for row in outcomes:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    unanimous = sum(
        1
        for row in outcomes
        if row.get("paired")
        and (row.get("confirmations") or {}).get("bottom", {}).get("unanimous")
    )
    summary = {
        "contract": TOURNAMENT_CONTRACT,
        "groups": len(groups),
        "paired_groups": sum(1 for row in outcomes if row.get("paired")),
        "unanimous_bottom_pairs": unanimous,
        "calls_made": cache.calls_made,
        "failures": cache.failures,
        "elapsed_seconds": round(time.perf_counter() - started, 1),
        "observed_calls_per_second": (
            round(cache.calls_made / (time.perf_counter() - started), 3)
            if cache.calls_made
            else None
        ),
        "required_unanimous_votes": REQUIRED_UNANIMOUS_VOTES,
        "task07_training_authorized": False,
        "final_tests_used": [],
    }
    write_json(output_path.parent / "tournament_summary.json", summary)
    return summary


__all__ = [
    "BUNDLE_CONTRACT",
    "REQUIRED_UNANIMOUS_VOTES",
    "TOURNAMENT_CONTRACT",
    "BundleCandidate",
    "BundleGroup",
    "JudgeCache",
    "comparison_item",
    "confirmation_votes",
    "load_bundle",
    "run_group_tournament",
    "run_tournaments",
]
