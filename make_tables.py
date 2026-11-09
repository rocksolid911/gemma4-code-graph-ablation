"""Every run-derived number in paper/draft.md comes from this script. It only reads experiments/ (and writes nothing).

    python paper/make_tables.py > paper/tables_output.txt     (run from the repo root; standard library only)

Section K calls harness/scripts/edit_timing*.py on raw traces, which also needs data/competition/tasks.jsonl (Kaggle
competition data, not in the repo). Without it, section K is skipped.
"""
import collections
import glob
import itertools
import json
import math
import os
import random
import statistics as st
import subprocess
import sys
import tempfile

E = "experiments"
GRAPH = ("search_similar_code", "get_code_neighbors", "get_code_subgraph")
SEED = 20261104

E4 = {
    "off": ["20261024-e4b-e4off-s0", "20261024-e4b-e4off-s1", "20261024-e4b-e4off-s2"],
    "on": ["20261023-e4b-e6e7v2-s0", "20261024-e4b-e4on-s1", "20261024-e4b-e4on-s2"],
    "skill": ["20261025-e4b-e4skill-s0", "20261024-e4b-e4skill-s1", "20261024-e4b-e4skill-s2"],
    "placebo": ["20261024-e4b-e4placebo-s0", "20261024-e4b-e4placebo-s1", "20261024-e4b-e4placebo-s2"],
}
O1 = ["20261024-e4b-oracleO1-s0", "20261104-e4b-oracleO1-s1"]
SINGLES = {
    "skill never loaded": ["20261024-e4b-e4skillunused-s0"],
    "E8 required neighbour check": ["20261023-e4b-e8-s0"],
    "O2 graph neighbours hint": ["20261024-e4b-oracleO2-s0"],
    "O3 failing tests hint": ["20261024-e4b-oracleO3-s0"],
    "O1 first run only": ["20261024-e4b-oracleO1-s0"],
}
KAGGLE_E4 = {
    "31B off": ["20261024-31b-kaggle-e4off-s0"],
    "31B on": ["20261024-31b-kaggle-kagglefix-s0"],
    "31B on + skill": ["20261024-31b-kaggle-e4skill-s0"],
    "12B on (same files as 31B on)": ["20261025-12b-kaggle-kagglefix-s0"],
    "E4B on (same files as 31B on)": ["20261025-e4b-kaggle-kagglefix-s0"],
}
LATER_31B = {
    "8 min / 28 calls, tools on, 13 scoreable": ["20261025-31b-kaggle-budget8m28-s0", "20261025-31b-kaggle-budget8m28-s1"],
    "baseline20, 8 min / 28, tools on": ["20261028-31b-kaggle-baseline20-s0"],
    "Budget-Fit, 8 min / 28, told not to use": ["20261028-31b-kaggle-budgetfit-s0"],
    "merge, 8 min / 28, tools removed": ["20261101-31b-kaggle-merge-s0"],
    "merge, 8 min / 40, tools removed": ["20261101-31b-kaggle-merge40-s0", "20261101-31b-kaggle-merge40-s1"],
}
LATER_E4B = {
    "kagglefix 60 calls": ["20261022-e4b-kagglefix-s0"],
    "kagglefix 28 calls": ["20261026-e4b-budget28-s0", "20261026-e4b-budget28-s1"],
    "merge, tools removed, 15 min": ["20261101-e4b-merge-s0", "20261101-e4b-merge-s1"],
    "E4 off + scorer compaction": ["20261103-e4b-e4off-compact-s0"],
    "merge + scorer compaction": ["20261103-e4b-merge-compact-s0"],
}
TIMING_RUNS = [
    "20261028-31b-kaggle-baseline20-s0", "20261028-31b-kaggle-budgetfit-s0", "20261101-31b-kaggle-merge-s0",
    "20261101-31b-kaggle-merge40-s0", "20261101-31b-kaggle-merge40-s1", "20261024-e4b-e4off-s0",
    "20261024-e4b-e4off-s1", "20261024-e4b-e4off-s2", "20261101-e4b-merge-s0", "20261101-e4b-merge-s1",
]
# Read from the Kaggle submissions page; recorded in each run's notes.md. Not derived from files here.
LEADERBOARD = [
    ("57553747", "31B", "kagglefix, 5 min / 60 calls", "on", "0.08", "20261022-e4b-kagglefix-s0/notes.md"),
    ("57639721", "31B", "8 min / 28 calls", "on", "0.13", "20261025-31b-kaggle-budget8m28-s1/notes.md"),
    ("57907569", "31B", "merge, 8 min / 40 calls", "removed", "0.20 (12 of 58)", "20261101-31b-kaggle-merge40-s0/notes.md"),
    ("58013493", "31B", "same files as 57907569, resubmitted", "removed", "0.12 (7 of 58)", "20261101-31b-kaggle-merge40-s0/notes.md"),
    ("57867361", "E4B", "E4 off, 5 min / 60 calls", "removed", "0.03 (2 of 58)", "20261101-e4b-lb-e4off/notes.md"),
    ("57974203", "E4B", "merge, 5 min / 60 calls", "removed", "0.05 (3 of 58)", "20261103-e4b-lb-merge/notes.md"),
]


def done(run):
    return os.path.isfile(f"{E}/{run}/summary.json")


def summary(run):
    return json.load(open(f"{E}/{run}/summary.json", encoding="utf-8"))


def tasks(run):
    return [json.load(open(f, encoding="utf-8")) for f in sorted(glob.glob(f"{E}/{run}/tasks/*.json"))]


def steps(run):
    for f in glob.glob(f"{E}/{run}/traces/*.jsonl"):
        for line in open(f, encoding="utf-8"):
            if line.strip():
                row = json.loads(line)
                if row.get("role") == "tool_call":
                    yield row


def phi(xs, ys):
    a = sum(1 for x, y in zip(xs, ys) if x and y)
    b = sum(1 for x, y in zip(xs, ys) if x and not y)
    c = sum(1 for x, y in zip(xs, ys) if not x and y)
    d = sum(1 for x, y in zip(xs, ys) if not x and not y)
    den = math.sqrt((a + b) * (c + d) * (a + c) * (b + d))
    return round((a * d - b * c) / den, 2) if den else None


def describe(label, runs):
    runs = [r for r in runs if done(r)]
    if not runs:
        print(f"{label}: no finished runs")
        return
    scores = [summary(r)["n_resolved"] for r in runs]
    ts = [t for r in runs for t in tasks(r)]
    n = len(ts)
    calls = sum(t.get("tool_calls", 0) for t in ts)
    gcalls = sum(t.get("graph_calls", 0) for t in ts)
    peak = [t["peak_context_tokens"] for t in ts if t.get("peak_context_tokens")]
    edit_calls = sum(t.get("edit_file_calls", 0) for t in ts)
    edit_fail = sum(t.get("edit_file_failures", 0) for t in ts)
    usage = collections.Counter()
    for r in runs:
        usage.update(summary(r).get("tool_usage") or {})
    sd = round(st.stdev(scores), 2) if len(scores) > 1 else None
    print(f"{label}: runs={len(runs)} scores={scores} mean={round(st.mean(scores), 2)} sd={sd} range={min(scores)}-{max(scores)}")
    print(f"    task_runs={n} graph_calls={gcalls} of {calls} ({100 * gcalls / calls:.1f}%) per_task={gcalls / n:.2f}"
          f" file_hit={st.mean(bool(t.get('localization_hit')) for t in ts):.2f}"
          f" function_hit={st.mean(bool(t.get('function_hit')) for t in ts):.2f}"
          f" phi(function_hit,resolved)={phi([t.get('function_hit') for t in ts], [t.get('resolved') for t in ts])}"
          f" task_runs_with_graph_call={sum(1 for t in ts if t.get('graph_calls', 0) > 0)}")
    print(f"    peak_ctx_median={st.median(peak) if peak else None} over_26624={sum(p > 26624 for p in peak)}"
          f" wall_mean={st.mean(t['wall_s'] for t in ts):.0f}s calls_median={st.median(t.get('tool_calls', 0) for t in ts)}"
          f" edit_failures={edit_fail} of {edit_calls} ({100 * edit_fail / edit_calls if edit_calls else 0:.0f}%)"
          f" stops={dict(collections.Counter(t.get('stop_reason') for t in ts))}")
    print(f"    tool_usage={dict(usage)}")


def per_task(runs, key):
    table = collections.defaultdict(list)
    for r in runs:
        for t in tasks(r):
            table[t["task_id"]].append(float(t.get(key) or 0))
    return {k: st.mean(v) for k, v in table.items()}


def paired(a, b, key, total=False):
    pa, pb = per_task(a, key), per_task(b, key)
    ids = sorted(pa)
    diffs = [pa[t] - pb[t] for t in ids]
    agg = sum if total else st.mean
    rng = random.Random(SEED)
    boots = sorted(agg([rng.choice(diffs) for _ in ids]) for _ in range(20000))
    return round(agg(diffs), 2), round(boots[500], 2), round(boots[19500], 2)


def sign_flip(a, b):
    pa, pb = per_task(a, "resolved"), per_task(b, "resolved")
    diffs = [pa[t] - pb[t] for t in sorted(pa) if abs(pa[t] - pb[t]) > 1e-9]
    obs = abs(sum(diffs))
    if not diffs:
        return 1.0, 0
    hits = sum(abs(sum(s * d for s, d in zip(signs, diffs))) >= obs - 1e-9 for signs in itertools.product((1, -1), repeat=len(diffs)))
    return round(hits / 2 ** len(diffs), 3), len(diffs)


def gpu_readings(label, path, start, end):
    """Sensor log written by harness/scripts/gpu_log.py, copied into the run folder. Busy = GPU utilisation of 50% or more."""
    rows = []
    for line in open(path, encoding="utf-8", errors="replace"):
        cols = [c.strip() for c in line.split(",")]
        if line.startswith("#") or len(cols) < 10 or not start <= cols[0] <= end:
            continue
        try:
            rows.append((float(cols[1]), int(cols[4].split()[0]), cols[8] == "Active", cols[9] == "Active", float(cols[2].split()[0]),
                         int(cols[7], 16), cols[0], int(cols[3].split()[0])))
        except (ValueError, IndexError):
            continue
    busy = [r for r in rows if r[1] >= 50]
    temps = sorted(r[0] for r in busy)
    # NVML throttle-reason bits: 0x4 software power cap, 0x8 hardware slowdown, 0x20 software thermal, 0x40 hardware thermal.
    limited = sum(1 for r in busy if r[5] & 0x6c)
    heat = sum(1 for r in busy if r[5] & 0x60)
    cap_only = sum(1 for r in busy if r[5] & 0x4 and not r[5] & 0x60)
    print(f"{label}: busy_readings={len(busy)} of {len(rows)} gpu_c_median={temps[len(temps) // 2]} max={temps[-1]}"
          f" limited_by_heat_or_power_cap={limited} ({100 * limited / len(busy):.1f}%) heat_flags={heat} ({100 * heat / len(busy):.1f}%)"
          f" power_cap_no_heat_flag={cap_only} ({100 * cap_only / len(busy):.1f}%)"
          f" software_thermal={100 * sum(r[3] for r in busy) / len(busy):.1f}% hardware_thermal_readings={sum(r[2] for r in busy)}")
    pure = [r for r in busy if r[5] == 0x20]
    if pure:
        print(f"    heat-limited readings={len(pure)} power_w_median={st.median(r[4] for r in pure)} clock_mhz_median={st.median(r[7] for r in pure)} gpu_c_median={st.median(r[0] for r in pure)}")
    capped = [r for r in rows if "15:24:40" <= r[6] <= "15:50:04" and r[5] & 0x4]
    if capped:
        rest = [r for r in busy if not "15:24:40" <= r[6] <= "15:50:04"]
        print(f"    power-cap stretch 15:24:40-15:50:04: readings={len(capped)} power_w_median={st.median(r[4] for r in capped)}"
              f" range={min(r[4] for r in capped)}-{max(r[4] for r in capped)} clock_mhz_median={st.median(r[7] for r in capped)}"
              f" gpu_c_median={st.median(r[0] for r in capped)}; busy outside it with a heat flag={sum(1 for r in rest if r[5] & 0x60)} of {len(rest)}")


def main():
    arms = dict(E4)
    finished_o1 = [r for r in O1 if done(r)]
    arms["O1"] = finished_o1
    print("== A. E4B ablation (Ollama, 15 min, 60 calls, no compaction, 20 scoreable)")
    for k, v in arms.items():
        describe(k, v)
    print("\n== B. Single-run E4B arms, same conditions")
    for k, v in SINGLES.items():
        describe(k, v)

    print("\n== C. Paired by task: a minus b, 95% task bootstrap (20,000 resamples, not corrected for multiple comparisons)")
    pairs = [("on", "off"), ("on", "placebo"), ("placebo", "off"), ("skill", "on"), ("skill", "off"), ("O1", "on"), ("O1", "off")]
    for a, b in pairs:
        print(f"-- {a} - {b}: resolved per run {paired(arms[a], arms[b], 'resolved', total=True)}; "
              f"sign-flip (p, tasks that differ) {sign_flip(arms[a], arms[b])}")
        for key in ("peak_context_tokens", "wall_s", "tool_calls", "function_hit", "localization_hit"):
            print(f"     {key}: {paired(arms[a], arms[b], key)}")

    print("\n== D. Task by arm: runs resolved")
    allids = sorted(per_task(E4["off"], "resolved"))
    print("task".ljust(16), *[k.rjust(8) for k in arms])
    for t in allids:
        print(t.ljust(16), *[str(sum(t in summary(r)["resolved_ids"] for r in arms[k])).rjust(8) for k in arms])
    for k, v in arms.items():
        passes = [sum(t in summary(r)["resolved_ids"] for r in v) for t in allids]
        print(f"{k}: Pass@{len(v)}={sum(p > 0 for p in passes)} All@{len(v)}={sum(p == len(v) for p in passes)}")
    e4_runs = [r for v in E4.values() for r in v]
    solved_any = {t for r in e4_runs for t in summary(r)["resolved_ids"]}
    print(f"tasks solved in at least one of the {len(e4_runs)} E4 runs: {len(solved_any)}; never: {len(allids) - len(solved_any)}")
    b31 = [r for r in ("20261028-31b-kaggle-baseline20-s0", "20261028-31b-kaggle-budgetfit-s0", "20261101-31b-kaggle-merge-s0",
                       "20261101-31b-kaggle-merge40-s0", "20261101-31b-kaggle-merge40-s1")]
    s31 = {t for r in b31 for t in summary(r)["resolved_ids"]}
    print(f"tasks solved in at least one of the 5 all-20 31B runs: {len(s31)}; of the E4B-never tasks, 31B solved {sorted(s31 - solved_any)}")

    print("\n== E. Size of tool results, characters")
    for k, v in arms.items():
        chars = collections.defaultdict(list)
        ok = collections.Counter()
        for r in v:
            for row in steps(r):
                chars[row["tool_name"]].append(row.get("tool_result_chars") or 0)
                ok[row["tool_name"]] += bool(row.get("tool_ok"))
        n = len(v) * 20
        total = sum(sum(c) for c in chars.values())
        gr = sum(sum(chars[g]) for g in GRAPH)
        print(f"{k}: total/task={total / n:.0f} graph/task={gr / n:.0f} graph_share={100 * gr / total:.1f}%"
              f" read_file calls={len(chars['read_file'])} chars/task={sum(chars['read_file']) / n:.0f}")
        for g in GRAPH:
            if chars[g]:
                print(f"    {g}: calls={len(chars[g])} ok={ok[g]} median_chars={st.median(chars[g])}")

    print("\n== F. Kaggle vLLM, compaction on, 5 min / 60 calls, 13 of 20 scoreable, before the src/ notebook fix")
    for k, v in KAGGLE_E4.items():
        describe(k, v)
    print("\n== G. Later 31B runs, Kaggle vLLM, before the src/ notebook fix (20 scoreable unless stated)")
    for k, v in LATER_31B.items():
        describe(k, v)
    print("\n== H. Other E4B runs (Ollama, 15-min limit)")
    for k, v in LATER_E4B.items():
        describe(k, v)
    print("\n== I. Leaderboard (public, 58 tasks; one draw each)")
    for row in LEADERBOARD:
        print(" | ".join(row))

    print("\n== J. Edit failures and repeat spread")
    for r in TIMING_RUNS:
        s = summary(r)
        print(f"{r}: resolved={s['n_resolved']} edit_failure={s.get('edit_file_failure_rate')} calls_median={s['tool_calls']['median']} timeouts={s.get('timeout_rate')}")

    print("\n== M. Function hit against tasks resolved, across E4B laptop runs (Pearson r; Figure 2)")
    every = list(arms.values()) + [v for k, v in SINGLES.items() if k.split()[0] in ("O2", "O3", "E8")]
    for name, sets in (("all", every), ("without O1", [v for v in every if v is not arms["O1"]])):
        xy = [(summary(r)["function_hit_rate"], summary(r)["n_resolved"]) for v in sets for r in v]
        mx, my = st.mean(x for x, _ in xy), st.mean(y for _, y in xy)
        num = sum((x - mx) * (y - my) for x, y in xy)
        den = math.sqrt(sum((x - mx) ** 2 for x, _ in xy) * sum((y - my) ** 2 for _, y in xy))
        print(f"{name}: runs={len(xy)} r={num / den:.2f}")

    print("\n== L. GPU readings during O1 repeat 1 (part 1, then part 2 on a new BIOS and driver)")
    o1 = f"{E}/20261104-e4b-oracleO1-s1"
    gpu_readings("part 1, 15 tasks", f"{o1}/gpu-log-20261104.csv", "15:22:05", "17:03:02")
    gpu_readings("part 2, 5 tasks", f"{o1}/gpu-log-20261104-last5.csv", "23:10:27", "23:34:56")
    gpu_readings("old log, last 23 min of load", f"{o1}/gpu-log-20261104.csv", "16:47:00", "17:10:00")
    first = {r["instance_id"]: r["duration_seconds"] for r in map(json.loads, open(f"{E}/20261024-e4b-oracleO1-s0/raw/task_results.jsonl", encoding="utf-8"))}
    for name, part in (("part 1", "raw-first15"), ("part 2", "raw-last5")):
        rows = [json.loads(line) for line in open(f"{o1}/{part}/task_results.jsonl", encoding="utf-8")]
        print(f"{name}: tasks={len(rows)} resolved={sum(r['resolved'] for r in rows)} minutes={sum(r['duration_seconds'] for r in rows) / 60:.1f}"
              f" same_tasks_in_first_O1_run={sum(first[r['instance_id']] for r in rows) / 60:.1f}"
              f" timed_out={[r['instance_id'] for r in rows if 'timeout' in (r.get('error') or '')]}")

    print("\n== K. Timing on 200 task runs (harness/scripts/edit_timing*.py on raw/traces)")
    if not os.path.isfile("data/competition/tasks.jsonl"):
        print("skipped: data/competition/tasks.jsonl not present")
        return
    with tempfile.TemporaryDirectory() as tmp:
        tl = os.path.join(tmp, "timeline.jsonl")
        subprocess.run([sys.executable, "-I", "harness/scripts/edit_timing.py", ".", tl, *TIMING_RUNS], check=True)
        out = subprocess.run([sys.executable, "-I", "harness/scripts/edit_timing_report.py", tl], check=True, capture_output=True, text=True).stdout
        keep = ("===", "resolved  n=", "failed    n=", " buckets", " near_gold", " failed with patch")
        print("\n".join(line for line in out.splitlines() if line.startswith(keep)))
        rows = [json.loads(line) for line in open(tl, encoding="utf-8")]
    m40 = [r for r in rows if "merge40" in r["run"] and not r["resolved"]]
    hit = [r for r in m40 if (r.get("budget_calls") or 0) >= 40 or (r.get("harness_calls") or 0) >= 40 or r.get("error")]
    print(f"31B 40-call failed runs={len(m40)} hit a call or time limit={len(hit)} stopped with budget left={len(m40) - len(hit)}")
    e4 = [r for r in rows if "e4b" in r["run"]]
    calls = sorted(r["harness_calls"] for r in e4)
    q = st.quantiles(calls, n=4, method="inclusive")
    print(f"E4B task runs={len(e4)} calls quartiles={q} of 60; exactly one successful edit={sum(r['edit_ok'] == 1 for r in e4)}")
    long_runs = [r for r in e4 if r["harness_calls"] >= 20]
    print(f"E4B runs with 20+ calls={len(long_runs)} solved={sum(r['resolved'] for r in long_runs)}")


if __name__ == "__main__":
    main()
