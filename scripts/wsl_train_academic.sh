#!/usr/bin/env bash
# Launch train_academic.py inside WSL2 (Ubuntu-22.04, docroi env) reading the TrainingSet LMDB
# from the WSL ext4 copy (fast random access). All extra arguments are passed through.
#
# From Windows PowerShell:
#   wsl -d Ubuntu-22.04 -- bash "/mnt/d/shivendra pratap singh/docroi_former/scripts/wsl_train_academic.sh" --run_name NAME [args...]
# Detached (keeps running after the terminal closes): add  --detach  as the FIRST argument.
# It runs inside tmux session "docroi"; watch it live with:
#   wsl -d Ubuntu-22.04 -- tmux attach -t docroi        (detach again with Ctrl+B then D)
#
# Self-healing: if the trainer's slowdown watchdog gives up (exit code 75), it has already saved a
# step-exact latest_ checkpoint; this script restarts the process and resumes from it
# (at most MAX_RESTARTS times). Any other exit code (done, Ctrl+C, divergence) ends the script.
set -euo pipefail
PY=/home/ogmailab/miniconda3/envs/docroi/bin/python
REPO="/mnt/d/shivendra pratap singh/docroi_former"
LMDB="$HOME/data/Doctamper/DocTamperV1-TrainingSet"
MAX_RESTARTS=${MAX_RESTARTS:-5}
cd "$REPO"
if [[ "${1:-}" == "--detach" ]]; then
  shift
  if tmux has-session -t docroi 2>/dev/null; then
    echo "tmux session 'docroi' already exists (a run may be active). Attach with: tmux attach -t docroi"; exit 1
  fi
  mkdir -p runs
  ARGS=$(printf ' %q' "$@")
  tmux new-session -d -s docroi \
    "bash $(printf %q "$0")$ARGS 2>&1 | tee -a $(printf %q "$REPO/runs/tmux_console.out"); echo; echo '[finished] press Enter to close'; read"
  echo "Started in tmux session 'docroi'. Watch live: wsl -d Ubuntu-22.04 -- tmux attach -t docroi"
  exit 0
fi

extra=()
for ((attempt = 0; attempt <= MAX_RESTARTS; attempt++)); do
  rc=0
  "$PY" -u train_academic.py --train_lmdb "$LMDB" "$@" "${extra[@]}" || rc=$?
  if [[ $rc -ne 75 ]]; then
    exit $rc
  fi
  echo "[launcher] trainer asked for a clean restart (exit 75); resuming from latest_ checkpoint ($((attempt + 1))/$MAX_RESTARTS)"
  extra=(--auto_resume 1)
  sleep 10
done
echo "[launcher] gave up after $MAX_RESTARTS restarts"; exit 75
