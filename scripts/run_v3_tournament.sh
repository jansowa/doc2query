#!/usr/bin/env bash
# Turniej selektora v3 na maszynie z serwerem sędziego — jedna krótka komenda.
#
# Użycie:
#   bash scripts/run_v3_tournament.sh <BASE_URL> <MODEL> <API_KEY> [CONCURRENCY]
set -euo pipefail

if [ "$#" -lt 3 ]; then
  echo "Użycie: bash scripts/run_v3_tournament.sh <BASE_URL> <MODEL> <API_KEY> [CONCURRENCY]" >&2
  exit 64
fi

BASE_URL="$1"; MODEL="$2"; API_KEY="$3"; CONCURRENCY="${4:-8}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

SCRIPT="scripts/run_v3_tournament.py"
BUNDLE="artifacts/task06/v3_tournament_bundle_v1/tournament_bundle.jsonl"
[ -f "$SCRIPT" ] || { echo "Brak $SCRIPT — czy to katalog repozytorium?" >&2; exit 66; }
[ -f "$BUNDLE" ] || {
  echo "Brak pakietu turniejowego: $BUNDLE" >&2
  echo "Rozpakuj v3_tournament_bundle.tar.gz w artifacts/task06/ i uruchom ponownie." >&2
  exit 66
}
command -v uv >/dev/null || { echo "Brak 'uv' w PATH." >&2; exit 69; }

echo "== commit: $(git log --oneline -1 2>/dev/null || echo 'brak informacji z gita')"
echo "== model: $MODEL | endpoint: $BASE_URL | równolegle: $CONCURRENCY"
export QWEN_API_KEY="$API_KEY"
export PYTHONPATH=src

uv run --no-project --with pydantic --python 3.11 python "$SCRIPT" \
  --base-url "$BASE_URL" --model "$MODEL" --concurrency "$CONCURRENCY" --execute

echo
echo "== gotowe. Przywieź katalog artifacts/task06/v3_tournament_v1/"
