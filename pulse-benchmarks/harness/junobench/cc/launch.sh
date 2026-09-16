#!/bin/bash
# Claude Code on the same 8 JunoBench cases, in parallel, on this machine.
# Answer files are unreadable on both machines for the duration of the run.
set -u
WS=${WS:-$HOME/junobench_cc}
CASES=${CASES:-"1 2 3 4 5 6 7 8"}
MODEL=${MODEL:-claude-opus-5}

PROMPT='This directory contains notebook.py, a Jupyter notebook exported as a script: cells are separated by "# %%", and cells marked "[cell not executed]" were never run in the notebook session. Running it crashes.

The Python environment and the data/ folder live on a remote machine, so run Python only through ./py, which syncs this directory there and runs python in it. Examples: ./py notebook.py   or   ./py -c "import pandas as pd; print(pd.read_csv('"'"'data/file.csv'"'"').columns)"

Find the root cause of the crash and fix notebook.py with the smallest correct change so the script runs to completion. Work only inside this directory.'

# Save exact modes, lock, and restore them on exit.
REMOTE_LOCK='benchmark/*/*.ipynb benchmark/*/README.md *.xlsx pulse_config.json baseline runs runs_1'
LOCAL_MODE=$(stat -c %a "$HOME/junobench")
ssh jarvispc "cd ~/junobench && stat -c '%a %n' $REMOTE_LOCK > .modes_before_cc"
restore() {
  chmod "$LOCAL_MODE" "$HOME/junobench"
  ssh jarvispc 'cd ~/junobench && while read -r mode name; do chmod "$mode" "$name"; done < .modes_before_cc && rm .modes_before_cc'
}
trap restore EXIT
chmod 000 "$HOME/junobench"
ssh jarvispc "cd ~/junobench && chmod 000 $REMOTE_LOCK"

for i in $CASES; do
  case_dir="$WS/NBspecific_$i"
  (
    cd "$case_dir" &&
    timeout 3600 claude -p "$PROMPT" --model "$MODEL" \
      --permission-mode acceptEdits \
      --allowedTools "Read" "Edit" "Write" "Glob" "Grep" "Bash(./py:*)" \
      --disallowedTools "WebSearch" "WebFetch" "Agent" \
      --strict-mcp-config --setting-sources project \
      --output-format stream-json --verbose \
      > "$WS/NBspecific_$i.claude.jsonl" 2> "$WS/NBspecific_$i.claude.err"
    echo "NBspecific_$i exit=$? $(date +%T)" >> "$WS/done.txt"
  ) &
done
wait
echo ALL_DONE >> "$WS/done.txt"
