"""Benchmark-only tracing for Pulse runs (loaded via PYTHONPATH; changes no
behavior). Appends one JSON line per raw model call and per fix-apply result
to ./agent_trace.jsonl in the run directory, so failures Pulse doesn't print
(empty replies, rejected snippets) are visible afterwards."""
import builtins
import json
import os
import sys
import time

_TRACE = os.path.abspath("agent_trace.jsonl")


def _write(record):
    record["t"] = round(time.time(), 1)
    record["pid"] = os.getpid()
    with open(_TRACE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def _patch_pulse_cli(module):
    litellm = module.litellm
    real_completion = litellm.completion

    def completion(*args, **kwargs):
        base = {"kind": "llm", "model": kwargs.get("model"), "max_tokens": kwargs.get("max_tokens"),
                "prompt_chars": sum(len(str(m.get("content", ""))) for m in kwargs.get("messages", [])),
                "messages": kwargs.get("messages")}
        try:
            resp = real_completion(*args, **kwargs)
        except Exception as exc:
            _write({**base, "error": f"{type(exc).__name__}: {exc}"[:2000]})
            raise
        choice = resp.choices[0]
        msg = choice.message
        usage = getattr(resp, "usage", None)
        details = getattr(usage, "completion_tokens_details", None)
        _write({**base, "finish_reason": choice.finish_reason, "content": msg.content,
                "reasoning_chars": len(getattr(msg, "reasoning_content", None) or ""),
                "completion_tokens": getattr(usage, "completion_tokens", None),
                "reasoning_tokens": getattr(details, "reasoning_tokens", None)})
        return resp

    litellm.completion = completion

    real_apply = module.PulseCLI._apply_code_fix

    def _apply_code_fix(self, fix):
        result = real_apply(self, fix)
        _write({"kind": "apply_code_fix", "fix": fix, "result": result})
        return result

    module.PulseCLI._apply_code_fix = _apply_code_fix


_real_import = builtins.__import__


def _import(name, *args, **kwargs):
    module = _real_import(name, *args, **kwargs)
    target = sys.modules.get("pulse.pulse_cli")
    if (target is not None and not getattr(target, "_bench_traced", False)
            and hasattr(target, "PulseCLI") and hasattr(target, "litellm")):  # module fully loaded
        target._bench_traced = True
        _patch_pulse_cli(target)
    return module


builtins.__import__ = _import
