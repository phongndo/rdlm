#!/usr/bin/env bash
set -Eeuo pipefail

# Pull a backup archive from a remote GPU machine to the local computer.
#
# Usage:
#   bash scripts/pull_remote_backup.sh user@host:/path/to/rdlm/backups/rdlm_backup_*.tar.gz ./rdlm-backups/

usage() {
  cat <<'USAGE'
Usage:
  bash scripts/pull_remote_backup.sh REMOTE_ARCHIVE LOCAL_DIR

Examples:
  bash scripts/pull_remote_backup.sh user@host:/workspace/rdlm/backups/rdlm_backup_20260516_120000.tar.gz ./rdlm-backups/
  bash scripts/pull_remote_backup.sh user@host:'/workspace/rdlm/backups/rdlm_backup_*.tar.gz' ~/Downloads/rdlm-backups/
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ $# -ne 2 ]]; then
  usage
  exit 2
fi

REMOTE_ARCHIVE="$1"
LOCAL_DIR="$2"

mkdir -p "$LOCAL_DIR"

if command -v rsync >/dev/null 2>&1; then
  rsync -avh --progress "$REMOTE_ARCHIVE" "$LOCAL_DIR/"
  rsync -avh --progress "${REMOTE_ARCHIVE}.sha256" "$LOCAL_DIR/" 2>/dev/null || true
else
  scp "$REMOTE_ARCHIVE" "$LOCAL_DIR/"
  scp "${REMOTE_ARCHIVE}.sha256" "$LOCAL_DIR/" 2>/dev/null || true
fi

echo
echo "downloaded backup files:"
find "$LOCAL_DIR" -maxdepth 1 \( -name '*.tar.gz' -o -name '*.tgz' -o -name '*.sha256' \) -print | sort

latest_archive="$(find "$LOCAL_DIR" -maxdepth 1 \( -name '*.tar.gz' -o -name '*.tgz' \) -print | sort | tail -n 1)"
if [[ -n "$latest_archive" ]]; then
  extract_dir="${latest_archive%.tar.gz}"
  extract_dir="${extract_dir%.tgz}"
  echo
  if command -v shasum >/dev/null 2>&1; then
    echo "verify:"
    echo "  shasum -a 256 '$latest_archive'"
  elif command -v sha256sum >/dev/null 2>&1; then
    echo "verify:"
    echo "  sha256sum '$latest_archive'"
  fi
  echo "extract:"
  echo "  mkdir -p '$extract_dir'"
  echo "  tar -xzf '$latest_archive' -C '$extract_dir'"
fi
