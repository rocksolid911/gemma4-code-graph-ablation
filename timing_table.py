"""Table 4 and the timing counts in section 3.6 of the paper, from results/timing_200_runs.csv. Standard library only.

    python timing_table.py     (run from the release folder)

`bucket` is near_gold when a failed run's patch is within 20 lines of the gold change. Call numbers count tool calls.
"""
import collections
import csv
import statistics as st


def num(value):
    return float(value) if value not in ("", "None") else None


rows = list(csv.DictReader(open("results/timing_200_runs.csv", encoding="utf-8")))
for r in rows:
    r["solved"] = r["resolved"] == "True"
    r["errored"] = r["had_error"] == "True"
    r["group"] = "E4B, 60 calls" if "e4b" in r["run"] else "31B, 40 calls" if "merge40" in r["run"] else "31B, 28 calls"
    r["calls"] = num(r["harness_calls"])


def med(items, key):
    values = [num(i[key]) for i in items if num(i[key]) is not None]
    return round(st.median(values), 1) if values else None


for group in ("31B, 28 calls", "31B, 40 calls", "E4B, 60 calls"):
    runs = [r for r in rows if r["group"] == group]
    solved, failed = [r for r in runs if r["solved"]], [r for r in runs if not r["solved"]]
    print(f"{group}: task runs {len(runs)}, solved {len(solved)}")
    for key, label in (("gold_read", "first read of a gold file"), ("first_edit_try", "first edit"), ("harness_calls", "calls used")):
        print(f"  {label}, solved / failed: {med(solved, key)} / {med(failed, key)}")
    buckets = collections.Counter(r["bucket"] for r in failed)
    print(f"  failed runs {len(failed)}: {dict(buckets)}")

m40 = [r for r in rows if "merge40" in r["run"] and not r["solved"]]
hit = [r for r in m40 if (num(r["budget_calls"]) or 0) >= 40 or (r["calls"] or 0) >= 40 or r["errored"]]
print(f"31B 40-call failed runs={len(m40)} hit a call or time limit={len(hit)} stopped with budget left={len(m40) - len(hit)}")
e4 = [r for r in rows if r["group"] == "E4B, 60 calls"]
print(f"E4B calls quartiles={st.quantiles(sorted(r['calls'] for r in e4), n=4, method='inclusive')} of 60;"
      f" exactly one successful edit={sum(num(r['edit_ok']) == 1 for r in e4)}")
long_runs = [r for r in e4 if r["calls"] >= 20]
print(f"E4B runs with 20+ calls={len(long_runs)} solved={sum(r['solved'] for r in long_runs)}")
