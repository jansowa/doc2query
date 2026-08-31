#!/usr/bin/env bash
# Paczka v3: potwierdzenia par bez S4 + powtórka osi językowej.
#
#   scripts/run_night_jobs_v3.sh <BASE_URL> <MODEL> <API_KEY> [CONCURRENCY]
set -euo pipefail

[ -f pyproject.toml ] || cd doc2query
[ -f pyproject.toml ] || { echo "uruchom z katalogu repozytorium (albo katalog wyżej)" >&2; exit 2; }

BASE_URL="${1:?podaj BASE_URL}"
MODEL="${2:?podaj nazwę modelu}"
API_KEY="${3:?podaj API key}"
CONCURRENCY="${4:-16}"

INSERT_CA="${INSERT_CA:-/mnt/c/Users/Public/insert-ca.pem}"
if [ -z "${SSL_CERT_FILE:-}" ] && [ -f "$INSERT_CA" ]; then
  for candidate in /etc/ssl/certs/ca-certificates.crt /etc/pki/tls/certs/ca-bundle.crt; do
    if [ -f "$candidate" ]; then
      COMBINED="$(pwd)/artifacts/combined-ca.pem"
      cat "$candidate" "$INSERT_CA" > "$COMBINED"
      export SSL_CERT_FILE="$COMBINED"
      echo "[ssl] używam połączonego bundle CA: $COMBINED"
      break
    fi
  done
fi

IN="artifacts/task06/night_jobs_v3/input"
OUT="artifacts/task06/night_jobs_v3/verdicts"
[ -d "$IN" ] || { echo "brak $IN — rozpakuj night_jobs_v3_input.tar.gz w katalogu repo" >&2; exit 2; }

exec uv run python scripts/task06_night_jobs_remote.py \
  --jobs confirm_pairs,polish_recheck,sft_full_audit \
  --input-dir "$IN" --output-dir "$OUT" \
  --base-url "$BASE_URL" --model "$MODEL" --api-key "$API_KEY" \
  --concurrency "$CONCURRENCY"
