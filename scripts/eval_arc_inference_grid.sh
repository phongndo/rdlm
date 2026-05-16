#!/usr/bin/env bash
set -euo pipefail

# Factorial ARC inference smoke/eval harness.
# Required: RESUME=/path/to/checkpoint.pt EVAL_DIR=/path/to/arc/evaluation scripts/eval_arc_inference_grid.sh
# Optional knobs: DEVICE, DIM, MAX_EXAMPLES, EVAL_LIMIT, SAMPLE_STEPS, NUM_CANDIDATES, SHAPE_TOP_K, REPORT_DIR

: "${RESUME:?Set RESUME to a structured_encoder checkpoint path}"
: "${EVAL_DIR:?Set EVAL_DIR to ARC eval JSON directory}"

DEVICE="${DEVICE:-cuda}"
DIM="${DIM:-512}"
MAX_EXAMPLES="${MAX_EXAMPLES:-2}"
EVAL_LIMIT="${EVAL_LIMIT:-20}"
SAMPLE_STEPS="${SAMPLE_STEPS:-64}"
NUM_CANDIDATES="${NUM_CANDIDATES:-8}"
SHAPE_TOP_K="${SHAPE_TOP_K:-5}"
REPORT_DIR="${REPORT_DIR:-artifacts/reports/eval_grid}"
mkdir -p "$REPORT_DIR"

COMMON=(
  uv run python src/rdlm/train_arc.py
  --arch structured_encoder
  --eval-only
  --device "$DEVICE"
  --dim "$DIM"
  --max-examples "$MAX_EXAMPLES"
  --eval-dir "$EVAL_DIR"
  --resume "$RESUME"
  --infer-shape
  --shape-top-k "$SHAPE_TOP_K"
  --eval-limit "$EVAL_LIMIT"
  --sample-steps "$SAMPLE_STEPS"
  --inference-mode ensemble
  --num-candidates "$NUM_CANDIDATES"
  --use-shape-head
)

strategies=(confidence dos scanline border-first center-first)
temporal_flags=(off on)
calibration_flags=(off on)

for strategy in "${strategies[@]}"; do
  for temporal in "${temporal_flags[@]}"; do
    for calibration in "${calibration_flags[@]}"; do
      name="${strategy}_temporal-${temporal}_calibration-${calibration}"
      report="$REPORT_DIR/${name}.json"
      debug_dir="$REPORT_DIR/debug_${name}"
      cmd=("${COMMON[@]}" --sampling-strategy "$strategy" --eval-report "$report" --debug-dir "$debug_dir")
      if [[ "$temporal" == "on" ]]; then
        cmd+=(--temporal-vote)
      fi
      if [[ "$calibration" == "on" ]]; then
        cmd+=(--enable-calibration)
      fi
      echo "==> $name"
      "${cmd[@]}"
    done
  done
done

python - <<'PY' "$REPORT_DIR"
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
rows = []
for path in sorted(root.glob('*.json')):
    try:
        data = json.loads(path.read_text())
    except Exception:
        continue
    rows.append((
        path.name,
        data.get('exact', 0),
        data.get('cell_acc', 0),
        data.get('nonzero_iou', 0),
        data.get('shape_exact', 0),
        data.get('shape_topk_hit', 0),
    ))
print('report\texact\tcell_acc\tnonzero_iou\tshape_exact\tshape_topk_hit')
for row in rows:
    print('\t'.join(map(str, row)))
PY
