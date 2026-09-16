#!/bin/bash
# Build local Claude Code workspaces from the Deep4ge instances on the PC.
#   WS=/path/to/workspaces ./setup_workspaces.sh
set -eu
WS=${WS:?set WS}
HERE="$(cd "$(dirname "$0")" && pwd)"
rm -rf "$WS"
mkdir -p "$WS"
for name in $(ssh jarvispc 'ls ~/deep4ge_bench/instances'); do
  mkdir -p "$WS/$name"
  scp -q "jarvispc:deep4ge_bench/instances/$name/train.py" "jarvispc:deep4ge_bench/instances/$name/CustomCallback.py" "$WS/$name/"
  cp "$HERE/py" "$WS/$name/py"
  chmod +x "$WS/$name/py"
  ssh jarvispc "mkdir -p deep4ge_bench/runs_cc/$name"
done
cp "$HERE/launch.sh" "$WS/"
ls "$WS"
