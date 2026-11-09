"""Turn a raw `swegemma eval` results dir into the paper's data contract (paper/data_contract.md).

    python -I harness/postprocess.py experiments/20261022-baseline-e4b --model e4b-qat-ollama \
        --backend ollama --subset dev20 --seed 0 [--parent-run-id ID] [--config-diff "..."]

Reads <run>/raw/ (task_results.jsonl, traces/trace_<id>.json in ATIF-v1.7) plus data/competition/tasks.jsonl,
and writes <run>/summary.json, <run>/tasks/<id>.json and <run>/traces/<id>.jsonl.
Agreed caveats: thinking_tokens is a chars/4 estimate (Ollama reports none); `seed` is a repeat index at T > 0.
"""

import argparse
import json
import re
import statistics
from collections import Counter
from datetime import date
from pathlib import Path

EDIT_TOOLS = {"edit_file", "write_file"}
VERIFY_CMD = re.compile(r"\b(pytest|python3?|unittest)\b")
PATH_IN_CMD = re.compile(r"(?:/workspace/)?([\w./-]+\.py)\b")
REPRO_CMD = re.compile(r"\bpython3?\s+(?:-\S+\s+)*/tmp/\S+\.py")  # a /tmp script run, not pytest
TRACEBACK = re.compile(r"Traceback \(most recent call last\)|\b\w+(?:Error|Exception|Warning): ")
GRAPH_TOOLS = {"get_code_neighbors", "get_code_subgraph", "search_similar_code"}
HUNK = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+\d+(?:,\d+)? @@ ?(.*)$")


def rel(path: str) -> str:
    path = path.strip()
    if path.startswith("/workspace/"):
        path = path[len("/workspace/"):]
    return path.lstrip("./")


def gold_files(patch: str) -> list[str]:
    return sorted({line[6:].strip() for line in patch.splitlines() if line.startswith("+++ b/")})


def hunks(patch: str) -> list[dict]:
    """Per-hunk file, old-side line range and git's enclosing-function context (the text after `@@`)."""
    out, path = [], None
    for line in patch.splitlines():
        if line.startswith("+++ b/"):
            path = line[6:].strip()
        elif (m := HUNK.match(line)) and path:
            start, length = int(m[1]), int(m[2] or 1)
            func = re.search(r"\b(?:def|class)\s+(\w+)", m[3])
            out.append({"file": path, "start": start, "end": start + max(length, 1) - 1,
                        "func": func[1] if func else None})
    return out


def tool_ok(observation: str) -> bool:
    try:
        return json.loads(observation).get("status") == "ok"
    except (json.JSONDecodeError, AttributeError):
        return "error" not in observation.lower()[:200]


def process_task(trace: dict, result: dict, task: dict) -> tuple[dict, list[dict]]:
    steps_out, files_read, files_edited = [], {}, {}
    tokens = {p: {"prompt": 0, "completion": 0} for p in ("explore", "edit", "verify", "submit")}
    peak, turn, phase, submitted, truncated = 0, 0, "explore", False, 0
    tool_counts = Counter()
    edit_calls = edit_fails = repro_runs = repro_after_edit = pytest_runs = 0
    reads = []  # (file, start, end) of successful read_file calls
    # A call's usage arrives on its first (thinking/text) step, before its tool calls. Hold it until we know
    # which phase the call's tool puts us in; a call with no tool call is charged to the current phase.
    pending_usage = None

    def charge(usage):
        tokens[phase]["prompt"] += usage.get("prompt_tokens", 0)
        tokens[phase]["completion"] += usage.get("completion_tokens", 0)

    for s in trace.get("steps", []):
        if s.get("source") == "user" and "tool_call" in (s.get("message") or "") and "truncat" in s["message"].lower():
            truncated += 1
        if s.get("source") != "agent":
            continue
        m = s.get("metrics")
        if m:
            if pending_usage:
                charge(pending_usage)
            turn += 1
            peak = max(peak, m.get("prompt_tokens", 0))
            pending_usage = m
        extra = s.get("extra") or {}
        thinking_tokens = len(s.get("message") or "") // 4 if extra.get("event_type") == "thinking" else None
        calls = s.get("tool_calls") or []
        obs = (s.get("observation") or {}).get("content", "")
        if not calls:
            steps_out.append({"turn": turn, "role": extra.get("event_type", "agent"), "tool_name": None, "tool_args": None,
                              "tool_ok": None, "tool_result_chars": 0,
                              "prompt_tokens": m.get("prompt_tokens") if m else None,
                              "completion_tokens": m.get("completion_tokens") if m else None,
                              "thinking_tokens": thinking_tokens, "elapsed_s": extra.get("elapsed_s")})
            continue
        for c in calls:
            name, args = c["function_name"], c.get("arguments") or {}
            tool_counts[name] += 1
            ok = tool_ok(obs)
            path = rel(args.get("filepath", "")) if "filepath" in args else ""
            if name == "edit_file":
                edit_calls += 1
                edit_fails += not ok
            if name == "read_file" and ok and path:
                files_read.setdefault(path, turn)
                start = int(args.get("start_line") or 1)
                reads.append((path, start, min(int(args.get("end_line") or 10**9), start + 149)))
            elif name in EDIT_TOOLS and ok and path and not path.startswith("tmp/"):
                files_edited.setdefault(path, turn)
                phase = "edit"
            elif name == "run_command":
                cmd = args.get("command", "")
                for p in PATH_IN_CMD.findall(cmd):
                    if not p.startswith("/tmp") and re.search(r"\b(cat|sed|head|tail|grep|less|nl)\b", cmd):
                        files_read.setdefault(rel(p), turn)
                if REPRO_CMD.search(cmd):
                    repro_runs += 1
                    repro_after_edit += phase != "explore"
                pytest_runs += "pytest" in cmd
                if phase != "explore" and VERIFY_CMD.search(cmd):
                    phase = "verify"
            elif name == "submit_patch":
                phase, submitted = "submit", True
            if pending_usage:
                charge(pending_usage)
                pending_usage = None
            steps_out.append({"turn": turn, "role": "tool_call", "tool_name": name,
                              "tool_args": {k: str(v)[:300] for k, v in args.items()}, "tool_ok": ok,
                              "tool_result_chars": len(obs), "prompt_tokens": None, "completion_tokens": None,
                              "thinking_tokens": None, "elapsed_s": (c.get("extra") or {}).get("elapsed_s")})

    if pending_usage:
        charge(pending_usage)

    gold = gold_files(task["patch"])
    error = result.get("error") or ""
    hit_turns = [files_read[f] for f in gold if f in files_read]
    if submitted:
        stop = "submit"
    elif re.search(r"time|timed out", error, re.I):
        stop = "timeout"
    elif re.search(r"tool call", error, re.I):
        stop = "max_tool_calls"
    elif re.search(r"turn|iteration", error, re.I):
        stop = "max_turns"
    elif "without calling submit_patch" in error:
        stop = "ended"  # model stopped (or ran out of nudges) without submitting
    else:
        stop = "error" if error else "ended"
    has_patch = int(result.get("agent_patch_size") or 0) > 0
    # Function-level hit: a gold hunk's lines were read, or the agent edited the same enclosing function.
    gold_hunks = hunks(task["patch"])
    agent_hunks = hunks(result.get("_agent_patch") or "")
    func_read = any(f == h["file"] and a <= h["end"] and b >= h["start"] for h in gold_hunks for f, a, b in reads)
    gold_funcs = {(h["file"], h["func"]) for h in gold_hunks if h["func"]}
    func_edit = any((h["file"], h["func"]) in gold_funcs for h in agent_hunks)
    record = {
        "task_id": result["instance_id"], "repo": result["repo"], "resolved": bool(result["resolved"]),
        "patch_applies": (not error.startswith("Failed to apply")) if has_patch else None,
        "has_patch": has_patch, "submitted": submitted, "stop_reason": stop, "error": error or None,
        "gold_files": gold, "files_read": files_read, "files_edited": files_edited,
        "localization_hit": bool(hit_turns), "first_hit_turn": min(hit_turns) if hit_turns else None,
        "edit_hit": any(f in files_edited for f in gold),
        "function_hit": func_read or func_edit, "function_hit_read": func_read, "function_hit_edit": func_edit,
        "n_gold_files": len(gold), "n_files_edited": len(files_edited),
        "has_traceback": bool(TRACEBACK.search(task.get("problem_statement") or "")),
        "edit_file_calls": edit_calls, "edit_file_failures": edit_fails,
        "repro_runs": repro_runs, "repro_run": repro_runs > 0, "repro_after_edit": repro_after_edit > 0,
        "pytest_run": pytest_runs > 0,
        "graph_calls": sum(tool_counts[t] for t in GRAPH_TOOLS),
        "neighbor_calls": tool_counts["get_code_neighbors"] + tool_counts["get_code_subgraph"],
        "tokens_by_phase": tokens, "peak_context_tokens": peak,
        "turns": turn, "tool_calls": sum(tool_counts.values()), "tool_usage": dict(tool_counts),
        "tool_errors": sum(1 for s in steps_out if s["tool_ok"] is False),
        "truncated_tool_calls": truncated,
        "prompt_tokens": trace.get("final_metrics", {}).get("total_prompt_tokens", 0),
        "completion_tokens": trace.get("final_metrics", {}).get("total_completion_tokens", 0),
        "wall_s": float(result.get("duration_seconds") or 0),
        "failure_tag": None,
    }
    return record, steps_out


def mean_median(values: list[float]) -> dict:
    return {"mean": round(statistics.fmean(values), 2), "median": statistics.median(values)} if values else {}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--model", required=True)
    ap.add_argument("--backend", required=True, choices=["ollama", "vllm-kaggle", "leaderboard"])
    ap.add_argument("--subset", required=True)
    ap.add_argument("--seed", type=int, default=0, help="repeat index (sampling is at T > 0)")
    ap.add_argument("--parent-run-id")
    ap.add_argument("--config-diff", default="")
    ap.add_argument("--tasks", default="data/competition/tasks.jsonl")
    args = ap.parse_args()

    run = Path(args.run_dir)
    raw = run / "raw"
    tasks = {t["instance_id"]: t for t in map(json.loads, open(args.tasks, encoding="utf-8"))}
    (run / "tasks").mkdir(exist_ok=True)
    (run / "traces").mkdir(exist_ok=True)

    records = []
    for line in open(raw / "task_results.jsonl", encoding="utf-8"):
        result = json.loads(line)
        tid = result["instance_id"]
        trace_path = raw / "traces" / f"trace_{tid}.json"
        trace = json.loads(trace_path.read_text(encoding="utf-8")) if trace_path.exists() else {}
        patch_path = raw / "patches" / f"{tid}.patch"
        result["_agent_patch"] = patch_path.read_text(encoding="utf-8", errors="replace") if patch_path.exists() else ""
        record, steps = process_task(trace, result, tasks[tid])
        records.append(record)
        (run / "tasks" / f"{tid}.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        with open(run / "traces" / f"{tid}.jsonl", "w", encoding="utf-8") as f:
            f.writelines(json.dumps(s) + "\n" for s in steps)

    n = len(records)
    with_patch = [r for r in records if r["has_patch"]]
    tool_usage = Counter()
    for r in records:
        tool_usage.update(r["tool_usage"])
    total_calls = sum(r["tool_calls"] for r in records)

    def rate(key):
        return round(sum(bool(r[key]) for r in records) / n, 4) if n else 0

    summary = {
        "run_id": run.name, "date": date.today().isoformat(), "model": args.model, "backend": args.backend,
        "subset": args.subset, "seed": args.seed, "parent_run_id": args.parent_run_id, "config_diff": args.config_diff,
        "n_tasks": n, "n_resolved": sum(r["resolved"] for r in records),
        "resolve_rate": round(sum(r["resolved"] for r in records) / n, 4) if n else 0,
        "patch_applies_rate": round(sum(bool(r["patch_applies"]) for r in with_patch) / len(with_patch), 4) if with_patch else None,
        "submitted_rate": round(sum(r["submitted"] for r in records) / n, 4) if n else 0,
        "timeout_rate": round(sum(r["stop_reason"] == "timeout" for r in records) / n, 4) if n else 0,
        "tool_error_rate": round(sum(r["tool_errors"] for r in records) / total_calls, 4) if total_calls else 0,
        "truncated_tool_call_rate": round(sum(r["truncated_tool_calls"] for r in records) / total_calls, 4) if total_calls else 0,
        "localization_hit_rate": round(sum(r["localization_hit"] for r in records) / n, 4) if n else 0,
        "stop_reasons": dict(Counter(r["stop_reason"] for r in records)),
        **{k: mean_median([r[k] for r in records]) for k in ("turns", "tool_calls", "prompt_tokens", "completion_tokens", "wall_s")},
        "peak_context_tokens": mean_median([r["peak_context_tokens"] for r in records]),
        "tool_usage": dict(tool_usage),
        # Week 2 diagnostics (E6 edit reliability, E7 repro, E8/E4 graph use, paper metrics)
        "edit_file_failure_rate": round(sum(r["edit_file_failures"] for r in records)
                                        / max(1, sum(r["edit_file_calls"] for r in records)), 4),
        "repro_run_rate": rate("repro_run"), "repro_after_edit_rate": rate("repro_after_edit"),
        "pytest_run_rate": rate("pytest_run"), "function_hit_rate": rate("function_hit"),
        "graph_call_rate": round(sum(r["graph_calls"] for r in records) / total_calls, 4) if total_calls else 0,
        "neighbor_calls_per_task": round(sum(r["neighbor_calls"] for r in records) / n, 2) if n else 0,
        "fewer_files_than_gold": sum(r["n_files_edited"] < r["n_gold_files"] for r in records if not r["resolved"]),
        "resolved_by_traceback": {k: f'{sum(r["resolved"] for r in records if r["has_traceback"] == v)}'
                                     f'/{sum(r["has_traceback"] == v for r in records)}'
                                  for k, v in (("has_traceback", True), ("no_traceback", False))},
        "resolved_ids": sorted(r["task_id"] for r in records if r["resolved"]),
    }
    (run / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("run_id", "n_tasks", "n_resolved", "resolve_rate", "localization_hit_rate")}))


if __name__ == "__main__":
    main()
