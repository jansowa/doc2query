#!/usr/bin/env python3
"""Podgląd postępu trzech ramion Task 07: kroki, strata, tempo, ETA i stan GPU.

Czyta wyłącznie to, co runy same zapisują, więc nie dotyka treningu i można go
uruchamiać oraz zabijać dowolnie. Dwa źródła, bo mają różną świeżość:
`history.jsonl` jest dopisywany dopiero na checkpointach (co 25 kroków), a log
runnera drukuje krok co 10 — bierzemy większy z nich, żeby podgląd nie wyglądał na
zawieszony między checkpointami.

Tempo mierzy się na żywo, między odświeżeniami, a nie od startu procesu: po
wznowieniu z checkpointu ETA liczone od startu kłamałoby o krokach, których ten
proces nie policzył.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

ARMS = (
    ("dpo", "T07-V3-DPO-S42"),
    ("continued_sft", "T07-V3-CSFT-S42"),
    ("score_weighted_continued_sft", "T07-V3-WSFT-S42"),
)
BAR_WIDTH = 28


def _rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").split("\n"):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            break
    return rows


def _log_progress(log_path: Path, arm: str) -> tuple[int, float, int] | None:
    """Ostatni krok wydrukowany przez runner: świeższy niż historia między checkpointami."""
    if not log_path.is_file():
        return None
    pattern = re.compile(rf"\[{re.escape(arm)}\] krok (\d+)/\d+ loss ([0-9.]+) tokeny (\d+)")
    found = None
    for match in pattern.finditer(log_path.read_text(encoding="utf-8", errors="replace")):
        found = (int(match.group(1)), float(match.group(2)), int(match.group(3)))
    return found


def _target_steps(plan_path: Path, arm: str) -> int:
    try:
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        return int(plan["arms"][arm]["target_optimizer_steps"])
    except (OSError, KeyError, ValueError):
        return 0


def _bar(done: int, total: int) -> str:
    if total <= 0:
        return "─" * BAR_WIDTH
    filled = min(BAR_WIDTH, round(BAR_WIDTH * done / total))
    return "█" * filled + "·" * (BAR_WIDTH - filled)


def _duration(seconds: float) -> str:
    if math.isnan(seconds) or seconds < 0:
        return "?"
    minutes, secs = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m" if hours else f"{minutes}m{secs:02d}s"


def _gpu() -> str:
    if shutil.which("nvidia-smi") is None:
        return "brak nvidia-smi"
    try:
        output = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.total,utilization.gpu,power.draw,"
                "power.limit,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stdout.strip()
    except (subprocess.SubprocessError, OSError) as error:
        return f"nvidia-smi nie odpowiada: {error}"
    used, total, util, power, limit, temp = (part.strip() for part in output.split(","))
    return f"GPU {util}% · {used}/{total} MiB · {power}/{limit} W · {temp}°C"


def _arm_line(
    runs_dir: Path,
    plan_path: Path,
    arm: str,
    name: str,
    seen: dict[str, tuple[int, float]],
    log_path: Path,
) -> list[str]:
    directory = runs_dir / name
    manifest_path = directory / "run_manifest.json"
    history_path = directory / "history.jsonl"
    target = _target_steps(plan_path, arm)

    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        dev = manifest.get("dev", {})
        start = dev.get("start", {})
        end = dev.get("end", {})
        lines = [
            f"  {arm:<30} GOTOWE  {_bar(1, 1)} "
            f"{manifest['completed_optimizer_steps']}/{manifest['target_optimizer_steps']} "
            f"w {_duration(float(manifest.get('seconds_total', 0)))}",
            f"  {'':<30}   dev NLL/token {start.get('mean_chosen_nll_per_token', float('nan')):.4f}"
            f" → {end.get('mean_chosen_nll_per_token', float('nan')):.4f}"
            f" · margin acc {start.get('policy_margin_accuracy', float('nan')):.4f}"
            f" → {end.get('policy_margin_accuracy', float('nan')):.4f}",
        ]
        if "implicit_reward_accuracy" in end:
            lines.append(
                f"  {'':<30}   implicit reward acc (vs start) {end['implicit_reward_accuracy']:.4f}"
            )
        return lines

    rows = _rows(history_path)
    from_log = _log_progress(log_path, arm)
    if not rows and from_log is None:
        state = "czeka" if not directory.exists() else "start / pomiar dev"
        return [f"  {arm:<30} {state:<7} {_bar(0, target)} 0/{target}"]

    step = int(rows[-1]["step"]) if rows else 0
    loss = float(rows[-1]["loss"]) if rows else float("nan")
    tokens = int(rows[-1]["tokens_consumed"]) if rows else 0
    if from_log is not None and from_log[0] > step:
        step, loss, tokens = from_log
    now = time.time()
    # Tempo mierzone na żywo, między odświeżeniami: historia nie zapisuje czasu, a
    # liczenie go od startu procesu kłamałoby po wznowieniu z checkpointu.
    anchor = seen.get(name)
    pace = "tempo: mierzę…"
    if anchor is None or step < anchor[0]:
        seen[name] = (step, now)
    elif step > anchor[0]:
        per_step = (now - anchor[1]) / (step - anchor[0])
        remaining = max(0, target - step) * per_step
        pace = f"{per_step:.1f} s/krok · ETA {_duration(remaining)}"
    else:
        held = now - anchor[1]
        if held > 300:
            pace = f"⚠ bez nowego kroku od {_duration(held)}"
    silent_for = now - history_path.stat().st_mtime
    checkpoint = "  (checkpoint jest)" if (directory / "checkpoint").is_dir() else ""
    return [
        f"  {arm:<30} liczy   {_bar(step, target)} {step}/{target} · loss {loss:.4f} · {pace}",
        f"  {'':<30}   tokeny {tokens:,}".replace(",", " ")
        + f" · zapis historii {_duration(silent_for)} temu{checkpoint}",
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    parser.add_argument(
        "--plan",
        type=Path,
        default=Path("artifacts/task07/handoff_v3_bottom/plan/dpo_plan.json"),
    )
    parser.add_argument("--log", type=Path, default=Path("logs/task07_arms.log"))
    parser.add_argument("--interval", type=float, default=20.0, help="0 = jeden zrzut i wyjście")
    args = parser.parse_args()

    seen: dict[str, tuple[int, float]] = {}
    while True:
        lines = [
            f"Task 07 — trzy ramiona na planie {json.loads(args.plan.read_text())['plan_id']}",
            f"{time.strftime('%H:%M:%S')} · {_gpu()}",
            "",
        ]
        for arm, name in ARMS:
            lines.extend(_arm_line(args.runs_dir, args.plan, arm, name, seen, args.log))
        lines += ["", "Ctrl+C kończy podgląd; treningu to nie dotyka."]
        if args.interval > 0:
            print("\033[2J\033[H" + "\n".join(lines), flush=True)
            time.sleep(args.interval)
        else:
            print("\n".join(lines), flush=True)
            return


if __name__ == "__main__":
    main()
