#!/usr/bin/env bash
# Zadania nocne na serwerze inferencji — jedna komenda, wartości tylko tutaj.
#
#   scripts/run_night_jobs.sh <BASE_URL> <MODEL> <API_KEY> [CONCURRENCY]
#
# Wznawialne (journal per element); FRESH=1 zaczyna od zera. Kolejność zadań jest
# ustawiona wg wartości: najpierw to, co buduje nową klasę par, na końcu to, co
# tylko mierzy. Przerwanie w dowolnym momencie zostawia gotowe wcześniejsze
# zadania.
set -euo pipefail

[ -f pyproject.toml ] || cd doc2query
[ -f pyproject.toml ] || { echo "uruchom z katalogu repozytorium (albo katalog wyżej)" >&2; exit 2; }

BASE_URL="${1:?podaj BASE_URL, np. http://host:8000/v1}"
MODEL="${2:?podaj nazwę modelu}"
API_KEY="${3:?podaj API key}"
CONCURRENCY="${4:-12}"

IN="artifacts/task06/night_jobs_v1/input"
OUT="artifacts/task06/night_jobs_v1/verdicts"
[ -d "$IN" ] || { echo "brak $IN — rozpakuj night_jobs_input.tar.gz w katalogu repo" >&2; exit 2; }

if [ "${FRESH:-0}" = "1" ] && [ -d "$OUT" ]; then
  mv "$OUT" "${OUT}.przerwane-$(date +%Y%m%dT%H%M%S)"
fi

exec uv run python scripts/task06_night_jobs_remote.py \
  --jobs lexical_mutation,answer_leak_v2,label_purity,chosen_recheck \
  --input-dir "$IN" --output-dir "$OUT" \
  --base-url "$BASE_URL" --model "$MODEL" --api-key "$API_KEY" \
  --concurrency "$CONCURRENCY"
