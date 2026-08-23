"""Wspólne narzędzia testów: rozdzielenie testów CI od testów stanu lokalnego.

Część testów tego repozytorium świadomie sprawdza **rzeczywiste** artefakty runów
(`runs/`, `data/processed/`, `artifacts/`) — pinowanie fingerprintów, preflighty
fail-closed, zgodność manifestów. Te dane nigdy nie trafiają do gita (AGENTS.md
§17: brak dużych danych i wag w Git), więc w CI nie istnieją i takie testy nie
mogą tam przejść z definicji.

`require_local_artifacts()` pomija je wtedy jawnie, z podaniem brakującej
ścieżki, zamiast pozwalać im padać na `FileNotFoundError`. Na maszynie z
artefaktami testy wykonują się normalnie i nadal wykrywają dryf — pominięcie
dotyczy wyłącznie środowiska, w którym nie ma czego weryfikować.
"""

from __future__ import annotations

from pathlib import Path

import pytest

# Katalogi stanu lokalnego: obecne na maszynie roboczej, nieobecne w CI.
LOCAL_ARTIFACT_ROOTS: tuple[Path, ...] = (
    Path("runs"),
    Path("data/processed"),
    Path("artifacts"),
)


def require_local_artifacts(*extra: str | Path) -> None:
    """Pomiń test, jeżeli brakuje lokalnego stanu runów, którego CI nie ma."""
    candidates = [*LOCAL_ARTIFACT_ROOTS, *(Path(value) for value in extra)]
    missing = [str(path) for path in candidates if not path.exists()]
    if missing:
        pytest.skip(
            "test wymaga lokalnych artefaktów runów, których nie ma w tym środowisku: "
            + ", ".join(missing)
        )


def project_cache_env() -> dict[str, str]:
    """Env dla runnerów powłoki, które wymagają cache'ów na partycji projektu.

    Skrypty kampanii blokują się, gdy `HF_HOME` albo `UV_CACHE_DIR` leżą poza
    katalogiem projektu — to celowy guard (jedna partycja, przewidywalne miejsce na
    wagi i cache). Testy nie mogą jednak zależeć od tego, co operator ma w swoim
    shellu ani od tego, że `setup-uv` w CI ustawia własny `UV_CACHE_DIR` poza
    workspace'em, więc podają te dwie ścieżki jawnie.
    """
    root = Path(__file__).resolve().parents[1]
    return {
        "HF_HOME": str(root / ".cache" / "huggingface"),
        "UV_CACHE_DIR": str(root / ".uv-cache"),
    }
