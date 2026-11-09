#!/bin/bash
# Download PGPG demonstrations, rollouts and pretrained reference policies.
# Example:
#   bash bash_script/others/download_data.sh                 # all five tasks (~3.9G)
#   bash bash_script/others/download_data.sh square lift     # only these tasks
set -euo pipefail

REPO=${PGPG_HF_REPO:-tungluu2203/pgpg-data}
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
TASKS=${*:-"lift can square transport tool_hang"}
TMP="${ROOT}/.hf_download"

for T in $TASKS; do
  case "$T" in
    lift|can|square|transport|tool_hang) ;;
    *) echo "Unknown task '$T' (use: lift can square transport tool_hang)"; exit 1 ;;
  esac
done

command -v huggingface-cli >/dev/null || {
  echo "huggingface-cli not found. Activate the environment first: conda activate py39_dp"; exit 1; }

# One --include taking every pattern. The flag is nargs='*', so repeating it
# REPLACES the previous value instead of accumulating -- passing it per archive
# silently downloads only the last group.
PATTERNS=("SHA256SUMS")
for T in $TASKS; do
  PATTERNS+=("data/${T}.tar.gz" "videos/${T}.tar" "pretrain/${T}.tar")
done

echo "Downloading [${TASKS}] from ${REPO} ..."
mkdir -p "$TMP"
huggingface-cli download "$REPO" --repo-type dataset --local-dir "$TMP" \
  --include "${PATTERNS[@]}"

echo "Verifying checksums ..."
cd "$TMP"
for T in $TASKS; do
  grep -E "  (data/${T}\.tar\.gz|videos/${T}\.tar|pretrain/${T}\.tar)$" SHA256SUMS | sha256sum -c -
done

echo "Extracting into ${ROOT} ..."
for T in $TASKS; do
  for A in "data/${T}.tar.gz" "videos/${T}.tar" "pretrain/${T}.tar"; do
    tar -xf "$A" -C "$ROOT"
  done
done

cd "$ROOT" && rm -rf "$TMP"
echo "Done. Data is in logs/collect_data/, reference policies in logs/pretrain/."
