#!/usr/bin/env bash
# Kalibracja selektora v3 na zdalnym serwerze sędziego — jedna krótka komenda.
#
# Użycie:
#   bash scripts/run_v3_calibration.sh <BASE_URL> <MODEL> <API_KEY> [CONCURRENCY]
#
# Skrypt istnieje po to, żeby nie wklejać wieloliniowych łańcuchów z cudzysłowami:
# urwany wklej zostawiał bash w oczekiwaniu na domknięcie stringa. Tu wszystkie
# argumenty są pozycyjne, a cała logika siedzi w pliku.
set -euo pipefail

if [ "$#" -lt 3 ]; then
  echo "Użycie: bash scripts/run_v3_calibration.sh <BASE_URL> <MODEL> <API_KEY> [CONCURRENCY]" >&2
  exit 64
fi

BASE_URL="$1"
MODEL="$2"
API_KEY="$3"
CONCURRENCY="${4:-8}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

SCRIPT="scripts/run_task06_v3_selector_calibration.py"
CORPUS="artifacts/task06/reward_validation_corpus_v1/corpus.jsonl"
[ -f "$SCRIPT" ] || { echo "Brak $SCRIPT — czy to katalog repozytorium?" >&2; exit 66; }
[ -f "$CORPUS" ] || {
  echo "Brak korpusu kalibracyjnego: $CORPUS" >&2
  echo "Rozpakuj tam v3_calibration_inputs.tar.gz i uruchom ponownie." >&2
  exit 66
}
command -v uv >/dev/null || { echo "Brak 'uv' w PATH." >&2; exit 69; }

echo "== commit: $(git log --oneline -1 2>/dev/null || echo 'brak informacji z gita')"
echo "== model: $MODEL | endpoint: $BASE_URL | równolegle: $CONCURRENCY"

# Klucz idzie zmienną środowiskową, więc nie widać go w liście procesów.
export QWEN_API_KEY="$API_KEY"
export PYTHONPATH=src

# Domyślnie NIE ruszamy istniejącego katalogu: journal jest wznawialny, a odkładanie
# go na bok kasowałoby postęp przerwanego runu (np. przy zmianie równoległości).
# Świeży start wymaga jawnego FRESH=1.
STALE="artifacts/task06/v3_selector_calibration_v1"
if [ -d "$STALE" ]; then
  if [ "${FRESH:-0}" = "1" ]; then
    BACKUP="$STALE.stare-$(date -u +%Y%m%dT%H%M%SZ)"
    mv "$STALE" "$BACKUP"
    echo "== FRESH=1: odłożyłem poprzedni katalog kalibracji na $BACKUP"
  else
    echo "== wznawiam istniejący katalog $STALE (FRESH=1 wymusza świeży start)"
  fi
fi

run_arm() {
  echo
  echo "== ramię: $1"
  shift
  uv run --no-project --with pydantic --python 3.11 python "$SCRIPT" \
    --base-url "$BASE_URL" --model "$MODEL" --concurrency "$CONCURRENCY" "$@" --execute
}

run_arm "bez rozumowania"
run_arm "z rozumowaniem" --reasoning --max-completion-tokens 1024

echo
echo "== gotowe. Przywieź katalog artifacts/task06/v3_selector_calibration_v1/"
