"""Build the frozen dev20 subset from the 129 public tasks.

Run once; the output (dev20.json, pool109.json) is committed and never regenerated.

    python harness/subsets/build_dev20.py data/competition/tasks.jsonl

Method:
- Drop tasks whose gold patch is known to fail in the sandbox environment (ENV_UNSTABLE), so
  dev20 measures the agent, not environment noise. These stay in the training pool.
- Size bins: tertiles of lines changed (added + removed) in the gold patch, over the stable tasks.
- Per-repo quotas (fastapi 10, rich 7, requests 2, httpx 1), split across bins in proportion to
  that repo's bin counts (largest remainder), then sampled with a fixed seed.
"""

import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

SEED = 20261021
QUOTAS = {"fastapi/fastapi": 10, "Textualize/rich": 7, "psf/requests": 2, "encode/httpx": 1}

# Gold patch fails under the public sandbox image (connect timeouts with network_mode=none,
# pygments-version ANSI output, starlette-version fallbacks). Source: experiments/splits.json in
# github.com/happyc0der/gemma-swe-agent (2026-09-24 gold sweep, 109/129). Re-check with our own
# gold run once the snapshots are local.
ENV_UNSTABLE = {
    "fastapi_14077", "fastapi_14356", "fastapi_14851", "fastapi_15745", "fastapi_15763",
    "fastapi_15785", "requests_6589", "requests_6592", "requests_6629", "requests_6757",
    "requests_7328", "requests_7433", "requests_7502", "requests_7505", "rich_3064", "rich_3130",
    "rich_3296", "rich_3469", "rich_3472", "rich_3782", "rich_4070", "rich_4079",
}


def patch_stats(patch: str) -> tuple[int, int]:
    files, lines = set(), 0
    for line in patch.splitlines():
        if line.startswith("+++ "):
            files.add(line[4:])
        elif line.startswith(("+", "-")) and not line.startswith(("+++", "---")):
            lines += 1
    return len(files), lines


def allocate(quota: int, counts: dict[str, int]) -> dict[str, int]:
    """Split quota across bins in proportion to counts (largest remainder, capped by count)."""
    total = sum(counts.values())
    raw = {b: quota * c / total for b, c in counts.items()}
    alloc = {b: int(r) for b, r in raw.items()}
    for b in sorted(raw, key=lambda b: raw[b] - alloc[b], reverse=True):
        if sum(alloc.values()) == quota:
            break
        if alloc[b] < counts[b]:
            alloc[b] += 1
    return alloc


def main(tasks_path: str) -> None:
    tasks = [json.loads(line) for line in open(tasks_path, encoding="utf-8")]
    meta = {}
    for t in tasks:
        n_files, n_lines = patch_stats(t["patch"])
        meta[t["instance_id"]] = {"repo": t["repo"], "files": n_files, "lines": n_lines}

    stable = sorted(i for i in meta if i not in ENV_UNSTABLE)
    sizes = sorted(meta[i]["lines"] for i in stable)
    t1, t2 = sizes[len(sizes) // 3], sizes[2 * len(sizes) // 3]
    for m in meta.values():
        m["bin"] = "small" if m["lines"] < t1 else "medium" if m["lines"] < t2 else "large"

    rng = random.Random(SEED)
    picked = []
    for repo, quota in QUOTAS.items():
        by_bin = defaultdict(list)
        for i in stable:
            if meta[i]["repo"] == repo:
                by_bin[meta[i]["bin"]].append(i)
        alloc = allocate(quota, {b: len(v) for b, v in by_bin.items()})
        for b in sorted(by_bin):
            picked += rng.sample(by_bin[b], alloc.get(b, 0))
    picked.sort()
    assert len(picked) == sum(QUOTAS.values())

    out = Path(__file__).parent
    dev20 = {
        "name": "dev20",
        "frozen": "2026-10-21",
        "seed": SEED,
        "method": "stratified by repo quota and gold-patch size tertile; env-unstable tasks excluded",
        "size_bins_lines": {"small": f"< {t1}", "medium": f"{t1}-{t2 - 1}", "large": f">= {t2}"},
        "task_ids": picked,
        "tasks": {i: meta[i] for i in picked},
    }
    pool = sorted(i for i in meta if i not in picked)
    (out / "dev20.json").write_text(json.dumps(dev20, indent=2) + "\n", encoding="utf-8")
    (out / "pool109.json").write_text(json.dumps({
        "name": "pool109",
        "note": "Training/trajectory pool: every public task not in dev20. env_unstable ones are flagged.",
        "task_ids": pool,
        "env_unstable": sorted(ENV_UNSTABLE),
    }, indent=2) + "\n", encoding="utf-8")

    print("dev20 by repo:", dict(Counter(meta[i]["repo"] for i in picked)))
    print("dev20 by bin: ", dict(Counter(meta[i]["bin"] for i in picked)))
    print("bins over stable:", dict(Counter(meta[i]["bin"] for i in stable)), "cuts", t1, t2)


if __name__ == "__main__":
    main(sys.argv[1])
