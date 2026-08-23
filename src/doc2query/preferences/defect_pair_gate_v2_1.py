"""Reguła decyzyjna bramki V2.1-05 — przedziałowa, fail-closed, bez własnych progów.

Bramka v2.0 przegrała „o jedną parę”, bo porównywała **punktowe** udziały z progami
5% i 3,1% przy n=500, gdzie półszerokość przedziału to ~2 pp. ADR v2.1 zastąpił to
regułą przedziałową: liczony jest dokładny jednostronny przedział Cloppera-Pearsona i
porównywany z **niezmienionym** progiem, a werdykt jest trójwartościowy:

* ``pass`` — cały przedział po właściwej stronie progu;
* ``fail`` — cały przedział po złej stronie progu (naruszenie dowiedzione);
* ``inconclusive`` — przedział zawiera próg; **fail-closed**, czyli bramka nie jest
  zdana, ale polityka nie jest sfalsyfikowana.

P1 (nieodpowiadalne `chosen`) jest **guardrailem**, nie predykcją konfirmacyjną:
zapala się dopiero, gdy naruszenie jest dowiedzione. Przy prawdzie 3,90% i progu 5%
test konfirmacyjny wymagałby n≈2310 dla mocy 0,80, czyli więcej par, niż jest w
autoryzowanej podaży — szczegóły i cała arytmetyka w §4.3 ADR.

Ten moduł **nie zna żadnego progu**: wszystkie czyta z zamrożonego configu polityki.
Nie zmienia artefaktów audytu, nie dotyka testów finalnych i nie autoryzuje treningu.

Dwie własności fail-closed są tu celowe i przetestowane:

1. **odmowa odczytu niedokończonego audytu** — dopóki `status` nie jest `complete` i
   dopóki każda para nie ma oceny obu sędziów, moduł odmawia policzenia czegokolwiek.
   Bramki nie wolno podejrzeć po pierwszym oknie budżetu i „zobaczyć, jak idzie”;
2. **odmowa niezgodnego kontraktu** — eksport i polityka muszą być tymi, które
   zamrożono; inaczej liczby dotyczyłyby innej populacji niż predykcje.
"""

from __future__ import annotations

import json
import random
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal, cast

from scipy.stats import beta

from doc2query.preferences.pair_audit_export_v2_1 import (
    EXPORT_CONTRACT,
    load_defect_blind_export_manifest_v2_1,
)
from doc2query.preferences.pair_policy_v2_1 import (
    DefectPairPolicyV21,
    load_defect_pair_policy_v2_1,
)
from doc2query.utils.records import read_records

GATE_CONTRACT = "task06-defect-pair-gate-v2-1"
DECIDED_PREFERENCES = frozenset({"A", "B"})
Verdict = Literal["pass", "fail", "inconclusive"]


def clopper_pearson_bounds(successes: int, total: int, alpha: float) -> tuple[float, float]:
    """Dokładne jednostronne granice Cloppera-Pearsona, każda na poziomie `alpha`."""
    if total <= 0:
        raise ValueError("Clopper-Pearson wymaga niepustej próby")
    if not 0 <= successes <= total:
        raise ValueError("liczba sukcesów musi mieścić się w próbie")
    if not 0.0 < alpha < 0.5:
        raise ValueError("alpha musi leżeć w (0, 0.5)")
    lower = 0.0 if successes == 0 else float(beta.ppf(alpha, successes, total - successes + 1))
    upper = (
        1.0
        if successes == total
        else float(beta.ppf(1 - alpha, successes + 1, total - successes))
    )
    return lower, upper


def verdict_at_most(successes: int, total: int, threshold: float, alpha: float) -> Verdict:
    """Predykcja typu „udział ≤ próg”: zdana, gdy CAŁY przedział leży pod progiem."""
    lower, upper = clopper_pearson_bounds(successes, total, alpha)
    if upper <= threshold:
        return "pass"
    if lower > threshold:
        return "fail"
    return "inconclusive"


def verdict_at_least(successes: int, total: int, threshold: float, alpha: float) -> Verdict:
    """Predykcja typu „udział ≥ próg”: zdana, gdy CAŁY przedział leży nad progiem."""
    lower, upper = clopper_pearson_bounds(successes, total, alpha)
    if lower >= threshold:
        return "pass"
    if upper < threshold:
        return "fail"
    return "inconclusive"


def guardrail_fired(successes: int, total: int, threshold: float, alpha: float) -> bool:
    """Guardrail zapala się wyłącznie przy DOWIEDZIONYM naruszeniu progu."""
    lower, _upper = clopper_pearson_bounds(successes, total, alpha)
    return bool(lower > threshold)


def paired_contrast(
    observations: Sequence[tuple[bool, bool]],
    *,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    """Sparowany kontrast (lewa minus prawa) z percentylowym bootstrapem po parach."""
    if not observations:
        raise ValueError("kontrast wymaga co najmniej jednej pary")
    count = len(observations)
    point = sum(left for left, _ in observations) / count - sum(
        right for _, right in observations
    ) / count
    rng = random.Random(seed)
    estimates: list[float] = []
    for _ in range(replicates):
        sample = [observations[rng.randrange(count)] for _ in range(count)]
        estimates.append(
            sum(left for left, _ in sample) / count - sum(right for _, right in sample) / count
        )
    estimates.sort()

    def percentile(fraction: float) -> float:
        index = min(len(estimates) - 1, max(0, round(fraction * (len(estimates) - 1))))
        return estimates[index]

    return {
        "pair_count": count,
        "difference": point,
        "ci95_low": percentile(0.025),
        "ci95_high": percentile(0.975),
        "replicates": replicates,
        "seed": seed,
    }


def _prediction(policy: DefectPairPolicyV21, identifier: str) -> Any:
    for row in policy.decision_rule.predictions:
        if row.id == identifier:
            return row
    raise KeyError(f"polityka nie definiuje predykcji {identifier}")


def _role_answerability(rating: Mapping[str, Any], automatic: str) -> tuple[bool, bool]:
    """Zwróć (chosen_unanswerable, rejected_unanswerable) po odślepieniu orientacji."""
    chosen_suffix, rejected_suffix = ("a", "b") if automatic == "A" else ("b", "a")
    return (
        not bool(rating[f"answerable_{chosen_suffix}"]),
        not bool(rating[f"answerable_{rejected_suffix}"]),
    )


def _require_complete_audit(analysis: Mapping[str, Any], expected_pairs: int) -> None:
    if str(analysis.get("status")) != "complete":
        raise ValueError(
            "bramki V2.1-05 nie wolno liczyć na niedokończonym audycie "
            f"(status={analysis.get('status')!r}); dokończ okna budżetu i wróć"
        )
    if int(analysis.get("rated_pair_count", -1)) != expected_pairs:
        raise ValueError(
            f"audyt ocenił {analysis.get('rated_pair_count')} par, a próbka ma {expected_pairs}"
        )


def measure_gate(
    *, export_dir: Path, audit_dir: Path, policy_path: Path
) -> dict[str, Any]:
    """Policz bramkę V2.1-05 na ukończonym audycie; progi pochodzą tylko z polityki."""
    policy = load_defect_pair_policy_v2_1(policy_path)
    manifest = load_defect_blind_export_manifest_v2_1(export_dir / "manifest.json")
    if manifest.contract != EXPORT_CONTRACT or manifest.policy_id != policy.policy_id:
        raise ValueError("eksport audytu nie należy do zamrożonej polityki v2.1")
    analysis = json.loads((audit_dir / "analysis.json").read_text(encoding="utf-8"))
    if str(analysis.get("export_policy_id")) != policy.policy_id:
        raise ValueError("analiza audytu opisuje inną politykę")
    _require_complete_audit(analysis, manifest.sampled_pair_count)

    rows = list(read_records(audit_dir / "pair_verdicts.jsonl"))
    if len(rows) != manifest.sampled_pair_count:
        raise ValueError("liczba werdyktów par nie zgadza się z próbką")
    models = sorted({model for row in rows for model in (row.get("ratings") or {})})
    if len(models) != 2:
        raise ValueError(f"bramka wymaga dokładnie dwóch sędziów, jest {len(models)}")

    total = len(rows)
    alpha = policy.decision_rule.alpha
    unanswerable_chosen = dict.fromkeys(models, 0)
    ties = dict.fromkeys(models, 0)
    contrast_inputs: dict[str, list[tuple[bool, bool]]] = {model: [] for model in models}
    consensus = {"supports": 0, "contradicts": 0, "abstained": 0, "disagreement": 0}
    for row in rows:
        ratings = cast(Mapping[str, Any], row.get("ratings") or {})
        if set(ratings) != set(models):
            raise ValueError(f"para {row.get('pair_id')} nie ma ocen obu sędziów")
        automatic = str(row["automatic_chosen_option"])
        for model in models:
            rating = cast(Mapping[str, Any], ratings[model])
            chosen_bad, rejected_bad = _role_answerability(rating, automatic)
            unanswerable_chosen[model] += int(chosen_bad)
            contrast_inputs[model].append((rejected_bad, chosen_bad))
            if str(rating["preference"]) not in DECIDED_PREFERENCES:
                ties[model] += 1
        label = str(row.get("consensus"))
        if label == "consensus_supports_automatic":
            consensus["supports"] += 1
        elif label == "consensus_contradicts_automatic":
            consensus["contradicts"] += 1
        elif label == "disagreement":
            consensus["disagreement"] += 1
        else:
            consensus["abstained"] += 1

    p1 = _prediction(policy, "P1")
    p2 = _prediction(policy, "P2")
    p3 = _prediction(policy, "P3")
    p4 = _prediction(policy, "P4_prime")

    p1_rows: dict[str, Any] = {}
    for model in models:
        successes = unanswerable_chosen[model]
        lower, upper = clopper_pearson_bounds(successes, total, alpha)
        p1_rows[model] = {
            "count": successes,
            "share": successes / total,
            "ci95_low": lower,
            "ci95_high": upper,
            "guardrail_fired": guardrail_fired(successes, total, float(p1.threshold), alpha),
            "confirmatory_verdict_if_it_were_one": verdict_at_most(
                successes, total, float(p1.threshold), alpha
            ),
        }
    p2_verdict = verdict_at_least(consensus["supports"], total, float(p2.threshold), alpha)
    p3_verdict = verdict_at_most(consensus["contradicts"], total, float(p3.threshold), alpha)
    p4_rows: dict[str, Any] = {}
    for model in models:
        stats = paired_contrast(
            contrast_inputs[model],
            replicates=int(p4.bootstrap_replicates or 10000),
            seed=int(p4.bootstrap_seed or policy.audit_sample.seed),
        )
        stats["verdict"] = (
            "pass"
            if stats["ci95_low"] >= float(p4.threshold)
            else "fail"
            if stats["ci95_high"] < float(p4.threshold)
            else "inconclusive"
        )
        p4_rows[model] = stats

    confirmatory = [p2_verdict, p3_verdict, *(row["verdict"] for row in p4_rows.values())]
    guardrails = [row["guardrail_fired"] for row in p1_rows.values()]
    gate_passed = all(value == "pass" for value in confirmatory) and not any(guardrails)
    blocking: list[str] = []
    if p2_verdict != "pass":
        blocking.append(f"P2:{p2_verdict}")
    if p3_verdict != "pass":
        blocking.append(f"P3:{p3_verdict}")
    blocking.extend(
        f"P4_prime:{model}:{row['verdict']}"
        for model, row in sorted(p4_rows.items())
        if row["verdict"] != "pass"
    )
    blocking.extend(
        f"P1_guardrail:{model}" for model, row in sorted(p1_rows.items()) if row["guardrail_fired"]
    )

    return {
        "schema_version": 1,
        "contract": GATE_CONTRACT,
        "policy_id": policy.policy_id,
        "export_contract": manifest.contract,
        "audit_ids_fingerprint": manifest.audit_ids_fingerprint,
        "pair_count": total,
        "alpha": alpha,
        "interval": policy.decision_rule.interval,
        "inconclusive_is_fail_closed": True,
        "multiplicity_correction": policy.decision_rule.multiplicity_correction,
        "predictions": {
            "P1": {
                "role": "guardrail",
                "threshold": p1.threshold,
                "per_model": p1_rows,
                "note": (
                    "guardrail: FAIL wyłącznie przy dowiedzionym naruszeniu. Brak zapalenia "
                    "NIE znaczy 'P1 przeszła' — przy n=800 moc konfirmacyjna to 0,39/0,76."
                ),
            },
            "P2": {
                "role": "confirmatory",
                "threshold": p2.threshold,
                "count": consensus["supports"],
                "share": consensus["supports"] / total,
                "ci95": clopper_pearson_bounds(consensus["supports"], total, alpha),
                "verdict": p2_verdict,
            },
            "P3": {
                "role": "confirmatory",
                "threshold": p3.threshold,
                "count": consensus["contradicts"],
                "share": consensus["contradicts"] / total,
                "ci95": clopper_pearson_bounds(consensus["contradicts"], total, alpha),
                "verdict": p3_verdict,
            },
            "P4_prime": {
                "role": "confirmatory",
                "threshold": p4.threshold,
                "per_model": p4_rows,
            },
            "P5": {
                "role": "reported_only",
                "tie_share": {model: ties[model] / total for model in models},
            },
        },
        "consensus_counts": consensus,
        "gate": {
            "id": "V2.1-05",
            "passed": gate_passed,
            "blocking": blocking,
            "decision": (
                "pairs_may_proceed_to_a_separate_training_authorization"
                if gate_passed
                else "pairs_do_not_proceed_policy_returns_to_design"
            ),
        },
        "human_evidence_claimed": False,
        "task07_training_authorized": False,
        "final_tests_used": [],
    }


__all__ = [
    "GATE_CONTRACT",
    "clopper_pearson_bounds",
    "guardrail_fired",
    "measure_gate",
    "paired_contrast",
    "verdict_at_least",
    "verdict_at_most",
]
