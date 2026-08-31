"""Behaviour tests for the unattended compute window scripts.

These guard a multi-day window in which nobody watches the machine, so the
supervisor and the poweroff guardian are exercised end to end with a fake
`nvidia-smi` on PATH.  Every guardian invocation runs in dry-run mode: the
guardian must never actually power off a machine from the test suite.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
SUPERVISOR = REPO / "scripts/run_unattended_queue.sh"
GUARDIAN = REPO / "scripts/poweroff_when_queue_drained.sh"
TIMEOUT_EXIT_CODE = 124


def _fake_gpu(tmp_path: Path, *, busy: bool = False, broken: bool = False) -> Path:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    if broken:
        body = "#!/bin/sh\nexit 9\n"
    elif busy:
        body = "#!/bin/sh\necho 4242\n"
    else:
        body = "#!/bin/sh\nexit 0\n"
    script = bin_dir / "nvidia-smi"
    script.write_text(body, encoding="utf-8")
    script.chmod(0o755)
    return bin_dir


def _env(tmp_path: Path, bin_dir: Path, **extra: Any) -> dict[str, str]:
    env = dict(os.environ)
    env["PATH"] = f"{bin_dir}{os.pathsep}{env['PATH']}"
    env["DOC2QUERY_QUEUE_ROOT"] = str(tmp_path / "state")
    env.update({key: str(value) for key, value in extra.items()})
    return env


def _queue_file(tmp_path: Path, rows: Sequence[tuple[Any, ...]]) -> Path:
    path = tmp_path / "queue.tsv"
    path.write_text(
        "".join("\t".join(str(field) for field in row) + "\n" for row in rows), encoding="utf-8"
    )
    return path


def _run_supervisor(
    tmp_path: Path, rows: Sequence[tuple[Any, ...]], *, gpu_busy: bool = False, **extra: Any
) -> subprocess.CompletedProcess[str]:
    bin_dir = _fake_gpu(tmp_path, busy=gpu_busy)
    env = _env(
        tmp_path,
        bin_dir,
        DOC2QUERY_QUEUE_MIN_FREE_GB=1,
        DOC2QUERY_QUEUE_RETRY_SLEEP=0,
        DOC2QUERY_QUEUE_POLL=1,
        DOC2QUERY_QUEUE_KILL_GRACE=1,
        DOC2QUERY_QUEUE_GPU_WAIT=0,
        **extra,
    )
    return subprocess.run(
        ["bash", str(SUPERVISOR), str(_queue_file(tmp_path, rows))],
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
        cwd=REPO,
        check=False,
    )


def _run_guardian(
    tmp_path: Path,
    rows: Sequence[tuple[Any, ...]],
    *,
    gpu_busy: bool = False,
    gpu_broken: bool = False,
    **extra: Any,
) -> subprocess.CompletedProcess[str]:
    bin_dir = _fake_gpu(tmp_path, busy=gpu_busy, broken=gpu_broken)
    env = _env(
        tmp_path,
        bin_dir,
        DOC2QUERY_POWEROFF_DRY_RUN=1,  # never power off a machine from the test suite
        DOC2QUERY_POWEROFF_STABLE_SECONDS=extra.pop("stable_seconds", 0),
        **extra,
    )
    return subprocess.run(
        ["bash", str(GUARDIAN), str(_queue_file(tmp_path, rows))],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        cwd=REPO,
        check=False,
    )


def _events(tmp_path: Path) -> list[dict[str, Any]]:
    path = tmp_path / "state/queue.events.jsonl"
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _done(tmp_path: Path) -> set[str]:
    directory = tmp_path / "state/done"
    return {path.name for path in directory.iterdir()} if directory.is_dir() else set()


def _summary(tmp_path: Path) -> dict[str, Any]:
    payload = json.loads((tmp_path / "state/queue.summary.json").read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def test_supervisor_completes_a_job_and_records_provenance(tmp_path: Path) -> None:
    result = _run_supervisor(tmp_path, [("only_job", 60, 1, "echo hello")])
    assert result.returncode == 0
    assert _done(tmp_path) == {"only_job"}
    events = _events(tmp_path)
    assert [(row["job"], row["exit_code"], row["outcome"]) for row in events] == [
        ("only_job", 0, "completed")
    ]
    summary = _summary(tmp_path)
    assert summary["state"] == "finished"
    assert summary["completed_jobs"] == ["only_job"]
    assert summary["failed_attempts"] == []
    assert summary["final_tests_used"] == []
    assert "hello" in (tmp_path / "state/logs/only_job.log").read_text(encoding="utf-8")


def test_failing_job_is_retried_and_never_stops_the_queue(tmp_path: Path) -> None:
    result = _run_supervisor(
        tmp_path, [("broken", 60, 2, "false"), ("later", 60, 1, "echo still running")]
    )
    assert result.returncode == 0
    assert _done(tmp_path) == {"later"}
    attempts = [row for row in _events(tmp_path) if row["job"] == "broken"]
    assert [row["attempt"] for row in attempts] == [1, 2]
    assert {row["outcome"] for row in attempts} == {"failed"}
    assert _summary(tmp_path)["state"] == "finished"


def test_supervisor_refuses_forbidden_commands_but_keeps_going(tmp_path: Path) -> None:
    result = _run_supervisor(
        tmp_path,
        [
            ("evil_poweroff", 60, 1, "sudo poweroff"),
            ("evil_final_test", 60, 1, "cat data/final_test.jsonl"),
            ("legit", 60, 1, "true"),
        ],
    )
    assert result.returncode == 0
    assert _done(tmp_path) == {"legit"}
    refused = {row["job"] for row in _events(tmp_path) if row["outcome"] == "refused"}
    assert refused == {"evil_poweroff", "evil_final_test"}
    assert "REFUSE evil_poweroff" in result.stdout


def test_supervisor_kills_a_job_that_exceeds_its_time_limit(tmp_path: Path) -> None:
    started = time.monotonic()
    # Unikalny czas snu: `pgrep -f "sleep 120"` łapał cudze procesy (np. retry
    # `sleep 120` w skryptach kolejek uruchomionych obok testów).
    result = _run_supervisor(tmp_path, [("hung", 1, 1, "sleep 1207")])
    assert result.returncode == 0
    assert time.monotonic() - started < 60
    assert _done(tmp_path) == set()
    events = _events(tmp_path)
    assert [row["exit_code"] for row in events] == [TIMEOUT_EXIT_CODE]
    assert subprocess.run(["pgrep", "-f", "sleep 1207"], check=False).returncode != 0


def test_supervisor_skips_completed_jobs_and_malformed_rows(tmp_path: Path) -> None:
    (tmp_path / "state/done").mkdir(parents=True)
    (tmp_path / "state/done/already").touch()
    result = _run_supervisor(
        tmp_path,
        [("already", 60, 1, "exit 3"), ("broken_row",), ("# comment",), ("fresh", 60, 1, "true")],
    )
    assert result.returncode == 0
    assert "SKIP already: already completed" in result.stdout
    assert "SKIP broken_row: malformed queue row" in result.stdout
    assert _done(tmp_path) == {"already", "fresh"}
    assert [row["job"] for row in _events(tmp_path)] == ["fresh"]


def test_supervisor_skips_a_job_when_the_gpu_never_frees_up(tmp_path: Path) -> None:
    result = _run_supervisor(tmp_path, [("needs_gpu", 60, 1, "true")], gpu_busy=True)
    assert result.returncode == 0
    assert _done(tmp_path) == set()
    assert [row["outcome"] for row in _events(tmp_path)] == ["gpu_busy"]


def test_second_supervisor_refuses_while_the_first_holds_the_lock(tmp_path: Path) -> None:
    (tmp_path / "state").mkdir(parents=True)
    lock = tmp_path / "state/queue.lock"
    lock.touch()
    holder = subprocess.Popen(["flock", str(lock), "-c", "sleep 3"])
    try:
        time.sleep(0.5)
        result = _run_supervisor(tmp_path, [("blocked", 60, 1, "true")])
        assert result.returncode == 3
        assert "another unattended queue is already running" in result.stderr
    finally:
        holder.wait(timeout=15)


def test_guardian_refuses_while_work_is_pending(tmp_path: Path) -> None:
    (tmp_path / "state/done").mkdir(parents=True)
    (tmp_path / "state/done/done_job").touch()
    result = _run_guardian(tmp_path, [("done_job", 60, 1, "true"), ("todo_job", 60, 1, "true")])
    assert result.returncode == 0
    assert "pending work" in result.stdout
    assert '"todo_job"' in result.stdout
    assert not (tmp_path / "state/drained_since.txt").exists()


def test_guardian_refuses_when_the_gpu_is_busy_or_unreadable(tmp_path: Path) -> None:
    (tmp_path / "state/done").mkdir(parents=True)
    (tmp_path / "state/done/job").touch()
    rows = [("job", 60, 1, "true")]
    busy = _run_guardian(tmp_path, rows, gpu_busy=True)
    assert "busy: GPU compute processes present" in busy.stdout
    broken = _run_guardian(tmp_path, rows, gpu_broken=True)
    assert "cannot query the GPU" in broken.stdout
    assert not (tmp_path / "state/drained_since.txt").exists()


def test_guardian_requires_a_stability_window_before_powering_off(tmp_path: Path) -> None:
    (tmp_path / "state/done").mkdir(parents=True)
    (tmp_path / "state/done/job").touch()
    rows = [("job", 60, 1, "true")]
    first = _run_guardian(tmp_path, rows, stable_seconds=3600)
    assert "starting the 3600s confirmation window" in first.stdout
    assert (tmp_path / "state/drained_since.txt").is_file()
    second = _run_guardian(tmp_path, rows, stable_seconds=3600)
    assert "waiting for 3600s" in second.stdout
    assert "powering off" not in second.stdout


def test_guardian_powers_off_once_the_queue_is_drained(tmp_path: Path) -> None:
    (tmp_path / "state/done").mkdir(parents=True)
    (tmp_path / "state/done/job").touch()
    rows = [("job", 60, 1, "true")]
    _run_guardian(tmp_path, rows)
    result = _run_guardian(tmp_path, rows)
    assert "powering off" in result.stdout
    assert "dry run: would have run 'sudo -n systemctl poweroff'" in result.stdout


def test_guardian_abandons_a_job_only_after_repeated_failures(tmp_path: Path) -> None:
    (tmp_path / "state/done").mkdir(parents=True)
    (tmp_path / "state/done/good").touch()
    events = tmp_path / "state/queue.events.jsonl"
    rows = [("good", 60, 1, "true"), ("cursed", 60, 1, "false")]
    with events.open("w", encoding="utf-8") as handle:
        for attempt in range(1, 4):
            handle.write(json.dumps({"job": "cursed", "attempt": attempt, "outcome": "failed"}))
            handle.write("\n")
    assert "pending work" in _run_guardian(tmp_path, rows).stdout
    with events.open("a", encoding="utf-8") as handle:
        for attempt in range(4, 7):
            handle.write(json.dumps({"job": "cursed", "attempt": attempt, "outcome": "failed"}))
            handle.write("\n")
    result = _run_guardian(tmp_path, rows)
    assert "queue drained" in result.stdout
    assert '"abandoned": ["cursed"]' in result.stdout


def test_guardian_honours_the_manual_hold_file(tmp_path: Path) -> None:
    (tmp_path / "state/done").mkdir(parents=True)
    (tmp_path / "state/done/job").touch()
    (tmp_path / "state/no_poweroff").touch()
    result = _run_guardian(tmp_path, [("job", 60, 1, "true")])
    assert result.returncode == 0
    assert "hold:" in result.stdout
    assert "powering off" not in result.stdout


def test_real_queue_file_is_well_formed_and_free_of_forbidden_commands() -> None:
    """The live queue must stay parseable: a malformed row silently skips real work."""
    rows = [
        line.split("\t")
        for line in (REPO / "configs/unattended_queue_2026-08-14.tsv")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    assert rows, "the queue file has no jobs"
    assert all(len(row) == 4 for row in rows)
    names = [row[0] for row in rows]
    assert len(names) == len(set(names))
    assert all(row[1].isdigit() and row[2].isdigit() for row in rows)
    forbidden = ("poweroff", "shutdown", "reboot", "final_test", "test_native_pl")
    assert not [row[0] for row in rows if any(marker in row[3] for marker in forbidden)]
    output_dirs = []
    for row in rows:
        fields = row[3].split()
        if "--output-dir" in fields:
            output_dirs.append(fields[fields.index("--output-dir") + 1])
        else:  # the cohort runner takes <config> <output-dir> positionally
            output_dirs.append(fields[-1])
    assert len(output_dirs) == len(rows)
    assert len(output_dirs) == len(set(output_dirs)), "two jobs would write to the same directory"
    assert not [path for path in output_dirs if not path.startswith(("runs/", "artifacts/"))]
