"""Tables for docs/research/20261102-edit-timing.md. usage: python -I edit_timing_report.py <timeline.jsonl> (run from the repo root)."""
import json, sys, re, collections, statistics as st
from pathlib import Path

root = Path(".")
R = [json.loads(l) for l in open(sys.argv[1])]
tasks = {}
for l in open("data/competition/tasks.jsonl", encoding="utf-8"):
    j = json.loads(l)
    tasks[j["instance_id"]] = j


def grp(r):
    if "e4b" in r["run"]:
        return "E4B local"
    return "31B 40-call" if "merge40" in r["run"] else "31B 28-call"


def med(x):
    x = [v for v in x if v is not None]
    return round(st.median(x), 1) if x else None


PASSED = re.compile(r"\b(\d+) passed")
FAILED = re.compile(r"\b(\d+) (?:failed|errors?)\b")


def status(obs):
    f = FAILED.search(obs or "")
    p = PASSED.search(obs or "")
    if f and int(f[1]) > 0:
        return "fail"
    if p:
        return "pass"
    return "other"


CAP = {"31B 28-call": (28, 480), "31B 40-call": (40, 480), "E4B local": (60, 1200)}
for r in R:
    r["g"] = grp(r)
    r["end"] = r["t_first_submit"] if r["t_first_submit"] is not None else r["agent_end"]
    tid = r["task"]
    raw = root / "experiments" / r["run"] / "raw"
    tp = tasks[tid]["test_patch"]
    tfiles = sorted({l[6:].strip() for l in tp.splitlines() if l.startswith("+++ b/")})
    t = json.load(open(raw / "traces" / f"trace_{tid}.json", encoding="utf-8"))
    n = 0
    any_pytest = target = None
    pytest_runs = 0
    for s in t["steps"]:
        if s.get("source") != "agent":
            continue
        obs = (s.get("observation") or {}).get("content", "") or ""
        for c in s.get("tool_calls") or []:
            name = c["function_name"]
            if name in ("submit_patch", "get_status"):
                continue
            n += 1
            a = c.get("arguments") or {}
            if r["last_edit_call"] is not None and n == r["last_edit_call"]:
                any_pytest = target = None
            if name == "run_command" and re.search(r"\bpytest\b", a.get("command", "")):
                pytest_runs += 1
                cmd = a["command"]
                any_pytest = status(obs)
                if any(f in cmd or Path(f).name in cmd for f in tfiles):
                    target = status(obs)
    r["pytest_runs"] = pytest_runs
    r["pytest_after_last_edit"] = any_pytest
    r["target_after_last_edit"] = target
    p = raw / "test_outputs" / f"{tid}.log"
    txt = p.read_text(encoding="utf-8", errors="replace") if p.exists() else ""
    names = re.findall(r"^(?:FAILED|ERROR) (\S+)", txt, flags=re.M)
    d = {k.rstrip("s"): int(v) for v, k in re.findall(r"(\d+) (failed|passed|errors?)\b", txt[-400:])}
    r["hidden_failed"] = d.get("failed", 0) + d.get("error", 0)
    r["hidden_passed"] = d.get("passed", 0)
    r["hidden_collection_error"] = bool(re.search(r"Interrupted: |errors? during collection|ModuleNotFoundError", txt[-3000:]))
    tp_new = set(re.findall(r"^\+\s*(?:async )?def (test_\w+)", tp, flags=re.M))
    tp_new |= set(re.findall(r"^\+class (Test\w+)", tp, flags=re.M))
    tp_new |= set(re.findall(r"^@@.*?(?:def|class) (\w+)", tp, flags=re.M))
    fn = [re.split(r"::", x)[-1].split("[")[0] for x in names]
    r["hidden_fail_in_patch_names"] = sum(x in tp_new for x in fn)
    r["hidden_fail_other"] = len(fn) - r["hidden_fail_in_patch_names"]

json.dump(R, open(sys.argv[1] + ".full.json", "w"))

for g in ("31B 28-call", "31B 40-call", "E4B local"):
    A = [r for r in R if r["g"] == g]
    if not A:
        continue
    cap, tcap = CAP[g]
    print(f"\n=== {g}: {len(A)} runs, {sum(r['resolved'] for r in A)} resolved")
    for lab, S in (("resolved", [r for r in A if r["resolved"]]), ("failed", [r for r in A if not r["resolved"]])):
        locate = [(r["t_first_edit_try"] if r["t_first_edit_try"] is not None else r["end"]) for r in S]
        tot = sum(r["end"] for r in S)
        after = sum(r["agent_end"] - r["end"] for r in S)
        print(f"{lab:9} n={len(S):3} gold read@{med([r['gold_read'] for r in S])} first edit@{med([r['first_edit_try'] for r in S])}"
              f" calls {med([r['harness_calls'] for r in S])} | t edit {med(locate)}s end {med([r['end'] for r in S])}s"
              f" locate {round(sum(locate) / tot * 100)}% aftersubmit {round(after / (tot + after) * 100)}%"
              f" | edits {med([r['edit_ok'] for r in S])} pytest {med([r['pytest_runs'] for r in S])}")
    F = [r for r in A if not r["resolved"]]
    P = [r for r in F if r["patch_chars"]]
    print(" buckets", collections.Counter(r["bucket"] for r in F).most_common())
    nf = [r for r in F if r["bucket"] == "near_gold"]
    print(" near_gold missed a gold file", sum(r["gold_files_missed"] > 0 for r in nf), "of", len(nf))
    vol = [r for r in P if not r["error"]]
    print(f" failed with patch {len(P)}; stopped by themselves {len(vol)}; median calls left {med([cap - r['harness_calls'] for r in vol])} of {cap};"
          f" median s left {med([tcap - r['end'] for r in vol])};"
          f" with >=10 calls and >=120 s left: {sum(cap - r['harness_calls'] >= 10 and tcap - r['end'] >= 120 for r in vol)}")
    print(" last pytest after last edit (failed w/ patch):", collections.Counter(r["pytest_after_last_edit"] for r in P).most_common())
    print(" target test file run after last edit (failed w/ patch):", collections.Counter(r["target_after_last_edit"] for r in P).most_common())
    Y = [r for r in A if r["resolved"]]
    print(" last pytest after last edit (resolved):", collections.Counter(r["pytest_after_last_edit"] for r in Y).most_common())
    print(" target file after last edit (resolved):", collections.Counter(r["target_after_last_edit"] for r in Y).most_common())
    print(" hidden: failing tests per failed run, median", med([r["hidden_failed"] for r in P]), "collection errors",
          sum(r["hidden_collection_error"] for r in P), "| 1-2 failing:", sum(1 <= r["hidden_failed"] <= 2 for r in P))
    print(" hidden failures in tests named by the test patch vs others:", sum(r["hidden_fail_in_patch_names"] for r in P),
          sum(r["hidden_fail_other"] for r in P), "| runs with any failing test outside those names", sum(r["hidden_fail_other"] > 0 for r in P))
    bins = [(1, 5), (6, 10), (11, 15), (16, 20), (21, 28), (29, 99)]
    print(" first edit vs resolved:", [(f"{a}-{b}", f"{sum(r['resolved'] for r in A if r['first_edit_try'] and a <= r['first_edit_try'] <= b)}/{sum(1 for r in A if r['first_edit_try'] and a <= r['first_edit_try'] <= b)}") for a, b in bins])
    print(" runs with >3 submit calls", sum(r["submit_calls"] > 3 for r in A), "seconds after first submit", round(sum(r["agent_end"] - r["end"] for r in A)))

for lab, A in (("31B all", [r for r in R if "e4b" not in r["run"]]), ("E4B all", [r for r in R if "e4b" in r["run"]])):
    by = collections.defaultdict(list)
    for r in A:
        by[r["task"]].append(r)
    e = l = t0 = 0
    rows = []
    for t, S in sorted(by.items()):
        y = [r for r in S if r["resolved"]]
        n = [r for r in S if not r["resolved"]]
        if y and n:
            fy = [r["first_edit_try"] for r in y if r["first_edit_try"]]
            fn = [r["first_edit_try"] for r in n if r["first_edit_try"]]
            if fy and fn:
                a, b = st.mean(fy), st.mean(fn)
                e += a < b
                l += a > b
                t0 += a == b
                rows.append((t, f"{len(y)}/{len(S)}", round(a, 1), round(b, 1), round(st.mean(r["harness_calls"] for r in y), 1), round(st.mean(r["harness_calls"] for r in n), 1)))
    print("\n", lab, "flip tasks: solved runs edited earlier on", e, ", later on", l, ", tie", t0)
    for x in rows:
        print("  ", x)
    print("  never solved", [t for t, S in sorted(by.items()) if not any(r["resolved"] for r in S)])
