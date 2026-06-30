#!/usr/bin/env bash
# Usage: run_gpqa.sh <proxy_url> <out> <tag> [limit]
cd "$(dirname "$0")"; source env.sh 2>/dev/null
CODEX_HF=/home/minjae/_DEV/aeq_icml/codex/aeq-icml/.hf-cache
CONTAINER_URL="$1" EVAL_MODE=quality QUALITY_LIMIT="${4:-30}" NUM_CONCURRENT="${NUM_CONCURRENT:-8}" \
  HF_HOME="$CODEX_HF" HF_DATASETS_CACHE="$CODEX_HF/datasets" \
  "$PYTHON" -c "
import eval_local as e
e.QUALITY_TASKS=[t for t in e.QUALITY_TASKS if t[1]=='gpqa_diamond']
r=e.run_quality_eval(); print('$3','=',r['gpqa_diamond']['score'],r['gpqa_diamond']['passed'])
" > "$2" 2>&1
echo DONE >> "$2"
