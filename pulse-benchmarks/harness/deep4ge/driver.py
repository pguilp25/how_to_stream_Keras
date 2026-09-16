"""Run one Deep4ge instance under Pulse, then ask Pulse's agent to fix it.

    driver.py <train.py> "<question>"

Deep4ge faults don't crash anything -- training just ends up worse -- so
there's no traceback to trigger Pulse. This reproduces what a user does in
Pulse's CLI: let the run finish with Pulse watching it, then ask the question
(the interactive prompt's "or ask AI" path, PulseCLI.ask_agent). Pulse keeps
its live training telemetry, and any fix it applies is written to train.py
and restarted by Pulse itself.
"""
import runpy
import sys
import time

import pulse.pulse_cli as pulse_cli

CRASH_RETRY_DELAYS = (5, 20, 60)

script, question = sys.argv[1], sys.argv[2]

instances = []
_original_init = pulse_cli.PulseCLI.__init__


def _init(self, *args, **kwargs):
    _original_init(self, *args, **kwargs)
    instances.append(self)


pulse_cli.PulseCLI.__init__ = _init

sys.argv = [script]
runpy.run_path(script, run_name="__main__")

if not instances:
    print("[driver] Pulse never started for this run -- nothing to ask.")
    sys.exit(2)

cli = instances[-1]
print(f"\n[driver] Training finished. Asking Pulse's agent:\n{question}\n", flush=True)
cli.ask_agent(question, include_code=True)
for delay in CRASH_RETRY_DELAYS:  # same foreground retry the crash hook uses
    if not cli._last_call_failed_transiently:
        break
    print(f"[driver] Agent request failed -- retrying in {delay}s...", flush=True)
    time.sleep(delay)
    cli.ask_agent(question, include_code=True)
