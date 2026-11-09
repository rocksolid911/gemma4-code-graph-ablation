"""Per-task timeline from ATIF traces: when the agent finds the gold file, first edits, tests and submits.

usage: python -I edit_timing.py <repo_root> <out.jsonl> <run> [<run> ...]
"""
import json, re, sys
from pathlib import Path

FREE = {"submit_patch", "get_status"}
HUNK = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+\d+(?:,\d+)? @@")
SHELL_WRITE = re.compile(r"\bsed\s+(?:-\S+\s+)*-i|\.write\(|write_text\(|\bgit\s+apply\b|\bpatch\s+-p|fileinput|\btee\b")
REDIRECT = re.compile(r"(?:^|[^0-9&>])>>?\s*([\w./-]+)")
PY_PATH = re.compile(r"[\w./-]+\.(?:py|pyi|md|toml|cfg|txt|rst|json|yaml|yml)\b")
VERIFY = re.compile(r"\bpytest\b|\bpython3?\s+(?:-\S+\s+)*(?:-c\b|/tmp/\S+|\S+\.py|-\s*<<)|\bunittest\b")
SCRATCH = re.compile(r"^(repro|reproduce|debug|verify|check|scratch|tmp|temp|fix|patch)[\w-]*\.py$|(^|/)test_[\w-]*\.py$|(^|/)conftest\.py$|^tests?/|/tests?/")


def rel(p):
    p = p.strip()
    if p.startswith("/workspace/"):
        p = p[len("/workspace/"):]
    return p[2:] if p.startswith("./") else p


def is_source(p):
    p = rel(p)
    return bool(p) and not p.startswith("/") and not SCRATCH.search(p)


def shell_targets(cmd):
    """Repo source files a shell command writes, or [] if it writes none."""
    targets = set()
    for m in REDIRECT.finditer(cmd):
        if is_source(m[1]) and "." in m[1].rsplit("/", 1)[-1]:
            targets.add(rel(m[1]))
    if SHELL_WRITE.search(cmd):
        targets |= {rel(p) for p in PY_PATH.findall(cmd) if is_source(p)}
    return sorted(targets)


def hunks(patch):
    out, path = [], None
    for line in patch.splitlines():
        if line.startswith("+++ b/"):
            path = line[6:].strip()
        elif (m := HUNK.match(line)) and path:
            start, length = int(m[1]), int(m[2] or 1)
            out.append((path, start, start + max(length, 1) - 1))
    return out


def test_status(obs):
    """pass / fail / unknown for a test or script run, from its tool result."""
    try:
        j = json.loads(obs)
    except (json.JSONDecodeError, TypeError):
        j = {}
    status = j.get("status") if isinstance(j, dict) else None
    text = obs or ""
    failed = re.search(r"\b(\d+) (?:failed|error)", text)
    passed = re.search(r"\b(\d+) passed", text)
    if failed and int(failed[1]) > 0:
        return "fail"
    if passed:
        return "pass"
    if status == "error" or re.search(r"Traceback \(most recent call last\)|AssertionError", text):
        return "fail"
    return "pass" if status == "ok" else "unknown"


def analyse(trace, result, gold_patch, agent_patch):
    gold_h = hunks(gold_patch)
    gold = {h[0] for h in gold_h}
    calls = []
    for s in trace.get("steps", []):
        if s.get("source") != "agent":
            continue
        obs = (s.get("observation") or {}).get("content", "") or ""
        for c in s.get("tool_calls") or []:
            calls.append((c["function_name"], c.get("arguments") or {}, (c.get("extra") or {}).get("elapsed_s"), obs))
    last_t = max((((s.get("extra") or {}).get("elapsed_s") or 0) for s in trace.get("steps", [])), default=0)

    n = 0  # budget calls so far
    r = dict(gold_read=None, gold_seen=None, first_edit_try=None, first_edit_ok=None, first_gold_edit=None,
             t_gold_read=None, t_first_edit_try=None, t_first_edit_ok=None, t_first_submit=None,
             edit_tries=0, edit_ok=0, last_edit_call=None, verify_runs=0, verify_after_edit=0, last_verify=None,
             submit_calls=0, submit_call_at=None, read_calls=0, shell_edits=0, reads_before_edit=0)
    verify_since_edit = None
    for name, args, t, obs in calls:
        if name == "submit_patch":
            r["submit_calls"] += 1
            if r["t_first_submit"] is None:
                r["t_first_submit"], r["submit_call_at"] = t, n
            continue
        if name in FREE:
            continue
        n += 1
        ok = True
        try:
            ok = json.loads(obs).get("status") == "ok"
        except (json.JSONDecodeError, AttributeError):
            ok = "error" not in obs.lower()[:200]
        targets, is_edit = [], False
        if name in ("edit_file", "write_file"):
            p = rel(args.get("filepath", ""))
            if is_source(p):
                targets, is_edit = [p], True
        elif name == "run_command":
            cmd = args.get("command", "")
            targets = shell_targets(cmd)
            is_edit = bool(targets)
            r["shell_edits"] += is_edit
            if not is_edit and VERIFY.search(cmd):
                r["verify_runs"] += 1
                st = test_status(obs)
                r["last_verify"] = st
                if r["first_edit_ok"] is not None:
                    r["verify_after_edit"] += 1
                    verify_since_edit = st
            viewed = name == "run_command" and re.search(r"\b(cat|sed|head|tail|nl|awk)\b", cmd)
            if viewed and not is_edit and r["gold_read"] is None and any(g in cmd for g in gold):
                r["gold_read"], r["t_gold_read"] = n, t
        if name == "read_file":
            r["read_calls"] += 1
            if r["first_edit_try"] is None:
                r["reads_before_edit"] += 1
            if ok and rel(args.get("filepath", "")) in gold and r["gold_read"] is None:
                r["gold_read"], r["t_gold_read"] = n, t
        if r["gold_seen"] is None and (any(g in obs for g in gold) or any(g in json.dumps(args) for g in gold)):
            r["gold_seen"] = n
        if is_edit:
            r["edit_tries"] += 1
            if r["first_edit_try"] is None:
                r["first_edit_try"], r["t_first_edit_try"] = n, t
            if ok:
                r["edit_ok"] += 1
                r["last_edit_call"] = n
                verify_since_edit = None
                if r["first_edit_ok"] is None:
                    r["first_edit_ok"], r["t_first_edit_ok"] = n, t
                if r["first_gold_edit"] is None and set(targets) & gold:
                    r["first_gold_edit"] = n

    # Where the submitted patch landed, relative to the gold change.
    a_h = hunks(agent_patch)
    a_files = {h[0] for h in a_h}
    if not agent_patch.strip():
        bucket = "no_patch"
    elif not a_files & gold:
        bucket = "other_files"
    elif any(af == gf and a0 <= g1 + 20 and a1 >= g0 - 20 for af, a0, a1 in a_h for gf, g0, g1 in gold_h):
        bucket = "near_gold"
    else:
        bucket = "gold_file_far"
    missed = sorted(gold - a_files)
    r.update(task=result["instance_id"], resolved=bool(result["resolved"]), budget_calls=n, harness_calls=result.get("tool_calls"),
             error=result.get("error"), duration=result.get("duration_seconds"), agent_end=last_t, bucket=bucket,
             n_gold=len(gold), gold_files_missed=len(missed), verify_since_last_edit=verify_since_edit,
             tested_after_last_edit=verify_since_edit is not None, patch_chars=len(agent_patch))
    return r


def main():
    root, out, runs = Path(sys.argv[1]), sys.argv[2], sys.argv[3:]
    tasks = {}
    for line in open(root / "data/competition/tasks.jsonl", encoding="utf-8"):
        j = json.loads(line)
        tasks[j.get("instance_id") or j.get("task_id")] = j
    with open(out, "w", encoding="utf-8") as f:
        for run in runs:
            raw = root / "experiments" / run / "raw"
            for line in open(raw / "task_results.jsonl", encoding="utf-8"):
                res = json.loads(line)
                tid = res["instance_id"]
                trace = json.load(open(raw / "traces" / f"trace_{tid}.json", encoding="utf-8"))
                pf = raw / "patches" / f"{tid}.patch"
                agent = pf.read_text(encoding="utf-8", errors="replace") if pf.exists() else ""
                rec = analyse(trace, res, tasks[tid]["patch"], agent)
                rec["run"] = run
                f.write(json.dumps(rec) + "\n")


main()
