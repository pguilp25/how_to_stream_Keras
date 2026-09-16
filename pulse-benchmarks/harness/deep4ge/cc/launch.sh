#!/bin/bash
# Claude Code on the Deep4ge instances, in parallel, on this machine.
# The correct programs and screening data are unreadable for the duration.
set -u
WS=${WS:?set WS to the workspace dir}
MODEL=${MODEL:-claude-opus-5}

PROMPT='This directory contains train.py, a Keras training script, and CustomCallback.py (logging only, not part of the task). The script has one injected bug that makes the model train worse than it should.

The Python environment lives on a remote machine, so run Python only through ./py, which syncs this directory there and runs python in it: ./py train.py   (one run trains the model and takes roughly a minute). CustomCallback.py writes a per-epoch CSV next to the script.

Find the root cause and fix train.py with the smallest correct change, so the model trains as well as it should. Work only inside this directory.'

REMOTE_LOCK='answers instances screen_report.json deep4ge/data runs'
ssh jarvispc "cd ~/deep4ge_bench && stat -c '%a %n' $REMOTE_LOCK > .modes_before_cc"
restore() {
  ssh jarvispc 'cd ~/deep4ge_bench && while read -r mode name; do chmod "$mode" "$name"; done < .modes_before_cc && rm -f .modes_before_cc'
}
trap restore EXIT
ssh jarvispc "cd ~/deep4ge_bench && chmod 000 $REMOTE_LOCK"

for case_dir in "$WS"/*/; do
  name=$(basename "$case_dir")
  (
    cd "$case_dir" &&
    timeout 3600 claude -p "$PROMPT" --model "$MODEL" \
      --permission-mode acceptEdits \
      --allowedTools "Read" "Edit" "Write" "Glob" "Grep" "Bash(./py:*)" \
      --disallowedTools "WebSearch" "WebFetch" "Agent" \
      --strict-mcp-config --setting-sources project \
      --output-format stream-json --verbose \
      > "$WS/$name.claude.jsonl" 2> "$WS/$name.claude.err"
    echo "$name exit=$? $(date +%T)" >> "$WS/done.txt"
  ) &
done
wait
echo ALL_DONE >> "$WS/done.txt"
