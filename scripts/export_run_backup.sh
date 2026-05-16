#!/usr/bin/env bash
set -Eeuo pipefail

# Create a portable backup archive for a disposable GPU training run.
#
# Default scope:
#   - Checkpoints and model weights from checkpoints/arc* when present.
#   - Run outputs: artifacts/, reports/, wandb/, runs/, logs/.
#   - Reproducibility files: source, scripts, tests, README, pyproject, uv.lock.
#   - A generated MANIFEST.txt with host, git, Python, uv, torch, and CUDA info.
#
# Usage:
#   bash scripts/export_run_backup.sh
#   bash scripts/export_run_backup.sh --checkpoint-dir checkpoints/arc --out-dir backups
#   bash scripts/export_run_backup.sh --include-data

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

OUT_DIR="backups"
ARCHIVE_NAME=""
INCLUDE_DATA=0
EXPLICIT_CHECKPOINT_DIRS=0
declare -a CHECKPOINT_DIRS=()

usage() {
  sed -n '2,18p' "$0" | sed 's/^# \{0,1\}//'
  cat <<'USAGE'

Options:
  --checkpoint-dir PATH   Include this checkpoint directory. May be passed multiple times.
  --out-dir PATH          Directory for the generated archive. Default: backups
  --name NAME             Archive filename. Default: rdlm_backup_<timestamp>.tar.gz
  --include-data          Include data/ in the archive.
  -h, --help              Show this help.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --checkpoint-dir)
      [[ $# -ge 2 ]] || { echo "error: --checkpoint-dir requires a path" >&2; exit 2; }
      EXPLICIT_CHECKPOINT_DIRS=1
      CHECKPOINT_DIRS+=("${2%/}")
      shift 2
      ;;
    --out-dir)
      [[ $# -ge 2 ]] || { echo "error: --out-dir requires a path" >&2; exit 2; }
      OUT_DIR="$2"
      shift 2
      ;;
    --name)
      [[ $# -ge 2 ]] || { echo "error: --name requires a filename" >&2; exit 2; }
      ARCHIVE_NAME="$2"
      shift 2
      ;;
    --include-data)
      INCLUDE_DATA=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "error: unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

timestamp="$(date -u +%Y%m%d_%H%M%S)"
if [[ -z "$ARCHIVE_NAME" ]]; then
  ARCHIVE_NAME="rdlm_backup_${timestamp}.tar.gz"
fi
case "$ARCHIVE_NAME" in
  *.tar.gz|*.tgz) ;;
  *) ARCHIVE_NAME="${ARCHIVE_NAME}.tar.gz" ;;
esac

mkdir -p "$OUT_DIR"
OUT_DIR="$(cd "$OUT_DIR" && pwd)"
ARCHIVE_PATH="$OUT_DIR/$ARCHIVE_NAME"
SHA_PATH="$ARCHIVE_PATH.sha256"

if [[ ${#CHECKPOINT_DIRS[@]} -eq 0 ]]; then
  shopt -s nullglob
  for candidate in checkpoints/arc*; do
    [[ -d "$candidate" ]] && CHECKPOINT_DIRS+=("$candidate")
  done
  shopt -u nullglob
fi

declare -a INCLUDE_PATHS=()
add_if_exists() {
  local path="$1"
  [[ -e "$path" ]] || return 0
  INCLUDE_PATHS+=("$path")
}

add_if_exists README.md
add_if_exists pyproject.toml
add_if_exists pyrightconfig.json
add_if_exists uv.lock
add_if_exists .python-version
add_if_exists src
add_if_exists scripts
add_if_exists tests
add_if_exists artifacts
add_if_exists reports
add_if_exists wandb
add_if_exists runs
add_if_exists logs

if [[ "$INCLUDE_DATA" == "1" ]]; then
  add_if_exists data
fi

for checkpoint_dir in "${CHECKPOINT_DIRS[@]+"${CHECKPOINT_DIRS[@]}"}"; do
  if [[ "$EXPLICIT_CHECKPOINT_DIRS" == "1" && ! -e "$checkpoint_dir" ]]; then
    echo "error: checkpoint path does not exist: $checkpoint_dir" >&2
    exit 1
  fi
  add_if_exists "$checkpoint_dir"
done

if [[ ${#CHECKPOINT_DIRS[@]} -eq 0 ]]; then
  echo "warning: no checkpoints/arc* directories found; archive will contain metadata and reports only" >&2
fi

if [[ ${#INCLUDE_PATHS[@]} -eq 0 ]]; then
  echo "error: no backup paths exist in $PROJECT_ROOT" >&2
  exit 1
fi

TMP_DIR="$(mktemp -d)"
cleanup() {
  rm -rf "$TMP_DIR"
}
trap cleanup EXIT

MANIFEST="$TMP_DIR/MANIFEST.txt"
{
  echo "rdlm backup manifest"
  echo "created_utc: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "project_root: $PROJECT_ROOT"
  echo "hostname: $(hostname 2>/dev/null || true)"
  echo "user: ${USER:-unknown}"
  echo
  echo "[git]"
  if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    echo "commit: $(git rev-parse HEAD 2>/dev/null || true)"
    echo "branch: $(git branch --show-current 2>/dev/null || true)"
    echo "status:"
    git status --short 2>/dev/null | sed 's/^/  /' || true
  else
    echo "not a git worktree"
  fi
  echo
  echo "[runtime]"
  if command -v python >/dev/null 2>&1; then
    echo "python: $(python --version 2>&1)"
  elif command -v python3 >/dev/null 2>&1; then
    echo "python: $(python3 --version 2>&1)"
  else
    echo "python: unavailable"
  fi
  echo "uv: $(uv --version 2>&1 || true)"
  if [[ -x .venv/bin/python ]]; then
    .venv/bin/python - <<'PY' 2>/dev/null || true
try:
    import torch
except Exception as exc:
    print(f"torch: unavailable ({exc})")
else:
    print(f"torch: {torch.__version__}")
    print(f"torch.version.cuda: {torch.version.cuda}")
    print(f"torch.cuda.is_available: {torch.cuda.is_available()}")
    print(f"torch.cuda.device_count: {torch.cuda.device_count()}")
    if torch.cuda.device_count():
        print(f"torch.cuda.device_0: {torch.cuda.get_device_name(0)}")
PY
  else
    echo "torch: skipped (.venv/bin/python not found)"
  fi
  if command -v nvidia-smi >/dev/null 2>&1; then
    echo
    echo "[nvidia-smi]"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>/dev/null || true
  fi
  echo
  echo "[included_paths]"
  printf '%s\n' "${INCLUDE_PATHS[@]}" | sed 's/^/  /'
  echo
  echo "[notes]"
  echo "Archive checksum is written next to the archive in $(basename "$SHA_PATH")."
} > "$MANIFEST"

declare -a TAR_ARGS=(
  --exclude .venv
  --exclude venv
  --exclude env
  --exclude __pycache__
  --exclude '.pytest_cache'
  --exclude '.ruff_cache'
  --exclude '.mypy_cache'
  --exclude '.DS_Store'
  --exclude 'backups/*.tar.gz'
  --exclude 'backups/*.tgz'
  --exclude 'backups/*.sha256'
)

tar -czf "$ARCHIVE_PATH" "${TAR_ARGS[@]}" -C "$TMP_DIR" MANIFEST.txt -C "$PROJECT_ROOT" "${INCLUDE_PATHS[@]}"

if command -v shasum >/dev/null 2>&1; then
  shasum -a 256 "$ARCHIVE_PATH" > "$SHA_PATH"
elif command -v sha256sum >/dev/null 2>&1; then
  sha256sum "$ARCHIVE_PATH" > "$SHA_PATH"
else
  echo "warning: no sha256 tool found; skipped checksum sidecar" >&2
fi

echo "backup archive: $ARCHIVE_PATH"
[[ -f "$SHA_PATH" ]] && echo "checksum: $SHA_PATH"
echo "pull from your Mac with:"
echo "  bash scripts/pull_remote_backup.sh user@host:$ARCHIVE_PATH ./rdlm-backups/"
