#!/usr/bin/env bash
# Podgląd kolejki probe embedderów Task 07 (odśwież: watch -n 60 scripts/watch_task07_probe.sh).
set -uo pipefail
cd "$(dirname "$0")/.."

echo "=== GPU ==="
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,power.draw,power.limit \
  --format=csv,noheader 2>/dev/null || echo "brak nvidia-smi"

echo
echo "=== Ramiona (runs/task07_probe) ==="
ARMS="start defect_dpo rpo beta02 divch bottom_dpo nearmiss_dpo defect_csft defect_wsft nearmiss_csft nearmiss_wsft bottom_csft bottom_wsft"
for arm in $ARMS; do
  dir="runs/task07_probe/$arm"
  if [ -f "$dir/result.json" ]; then
    python3 - "$dir" "$arm" <<'PY'
import json, sys
summary = json.load(open(f"{sys.argv[1]}/corpus_retrieval_summary.json"))
m = summary.get("metrics", summary)
r10 = m.get("corpus_recall_at_10")
nd = m.get("corpus_ndcg_at_10")
print(f"  {sys.argv[2]:14s} GOTOWE  recall@10={r10:.4f}  ndcg@10={nd:.4f}")
PY
  elif [ -d "$dir/corpus_embedding_cache" ]; then
    chunks=$(ls "$dir/corpus_embedding_cache"/*.pt 2>/dev/null | wc -l)
    echo "  $(printf '%-14s' "$arm") EWALUACJA  kodowanie korpusu: chunk $chunks/~100"
  elif [ -f "$dir/train_summary.json" ]; then
    echo "  $(printf '%-14s' "$arm") po treningu, przed ewaluacją"
  elif [ -d "$dir" ]; then
    echo "  $(printf '%-14s' "$arm") TRENING"
  else
    echo "  $(printf '%-14s' "$arm") w kolejce"
  fi
done

echo
echo "=== Log (ostatnie zdarzenia) ==="
grep "\[probe\]" logs/task07_probe_embedders.log 2>/dev/null | tail -5

echo
echo "=== Procesy ==="
ps -eo pid,etime,args | awk '/venv-gpu.*train_probe|queue_task07_probe/ && !/awk/ {printf "  %s %s %.90s\n", $1, $2, substr($0, index($0,$3))}'
done_count=$(ls runs/task07_probe/*/result.json 2>/dev/null | wc -l)
echo
echo "Gotowe ramiona: $done_count/13 (~3 h/ramię przy limicie 160 W)"
