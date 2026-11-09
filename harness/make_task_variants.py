"""Build local-only copies of the dev20 tasks for the paper's control and ceiling arms.

    python -I harness/make_task_variants.py            # writes data/variants/{placebo,O1,O2,O3}/

Each variant is a directory with its own tasks.jsonl (dev20 tasks only) plus graphs/ and embeddings/ next to it,
because swegemma looks for the graph data in <tasks dir>/graphs and <tasks dir>/embeddings. Run with
`run_eval.py --tasks data/variants/<name>/tasks.jsonl`.

- placebo: tasks unchanged, graphs/ and embeddings/ empty. The graph tools stay declared but find no data
  (ARISE-Coarse style control).
- O1: hints_text = gold file(s) + enclosing function(s) of each gold hunk (ORACLE-SWE "edit location").
- O2: hints_text = code linked by calls to those functions in the organizers' graph (its edges are stored
  both ways, so callers and callees can't be separated).
- O3: hints_text = the new test functions from test_patch (not in the snapshot): file::name and source.

hints_text is empty for all public tasks; the harness prints it under "## Hints:". These files never go into
submission/ and dev20 itself is not modified.
"""

import json
import re
import shutil
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
COMP = ROOT / "data" / "competition"
OUT = ROOT / "data" / "variants"
HUNK = re.compile(r"^@@ -\d+(?:,\d+)? \+\d+(?:,\d+)? @@ ?(.*)$")
MAX_NEIGHBORS = 12
MAX_TEST_CHARS = 3000


def gold_hunks(patch: str) -> list[dict]:
    """Per hunk: file, git's hunk-header function name, and the old-side source lines (context + removed)."""
    out, path = [], None
    for line in patch.splitlines():
        if line.startswith("+++ b/"):
            path = line[6:].strip()
        elif (m := HUNK.match(line)) and path:
            func = re.search(r"\b(?:def|class)\s+(\w+)", m[1])
            out.append({"file": path, "header_func": func[1] if func else None, "lines": []})
        elif out and line[:1] in (" ", "-") and not line.startswith("---"):
            out[-1]["lines"].append(line[1:].strip())
    return out


def module_of(path: str) -> str:
    return path.removesuffix(".py").removeprefix("src/").replace("/", ".").removesuffix(".__init__")


def locate(task: dict, graph: dict | None) -> dict[str, list[str]]:
    """File -> qualified symbols (e.g. `Console.__init__`) enclosing the gold hunks.

    The graph node whose source text contains most of a hunk's old-side lines wins (smallest node on ties),
    which recovers the class of a method; without a graph match, git's hunk-header name is used.
    """
    out = {}
    for h in gold_hunks(task["patch"]):
        module = module_of(h["file"])
        lines = [x for x in h["lines"] if len(x) >= 8]
        best = None
        if graph:
            for n in graph["nodes"]:
                if not n["id"].startswith(module + "."):
                    continue
                score = sum(x in n.get("text", "") for x in lines)
                if score and (best is None or (score, -len(n["text"])) > best[0]):
                    best = ((score, -len(n["text"])), n["id"][len(module) + 1:])
        name = best[1] if best else h["header_func"]
        names = out.setdefault(h["file"], [])
        if name and name not in names:
            names.append(name)
    return out


def o1_hint(task: dict, graph: dict) -> str:
    lines = ["The fix belongs in these locations:"]
    for path, funcs in locate(task, graph).items():
        lines.append(f"- `{path}`" + (": " + ", ".join(f"`{f}`" for f in funcs) if funcs else " (module level)"))
    return "\n".join(lines)


def o2_hint(task: dict, graph: dict) -> str:
    # The released graphs store each call edge in both directions, so callers and callees can't be told apart.
    linked = {}
    for e in graph["edges"]:
        if e.get("type") == "calls":
            linked.setdefault(e["source"], set()).add(e["target"])
    lines = ["Code linked by calls to the code involved in this issue (from the repository's code graph):"]
    for path, funcs in locate(task, graph).items():
        module = module_of(path)
        for func in funcs:
            node = f"{module}.{func}"
            if node not in linked:
                continue
            near = sorted(linked[node] - {node}, key=lambda x: (not x.startswith(module), x))[:MAX_NEIGHBORS]
            lines.append(f"- `{node}` is linked to: " + ", ".join(f"`{c}`" for c in near))
    return "\n".join(lines) if len(lines) > 1 else ""


def o3_hint(task: dict) -> str:
    """Test functions the test_patch adds that aren't in the snapshot yet: file, names, and their source."""
    snapshot = tarfile.open(COMP / "snapshots" / f"{task['instance_id']}.tgz")
    names_in_tar = set(snapshot.getnames())
    added, path = {}, None
    for line in task["test_patch"].splitlines():
        if line.startswith("+++ b/"):
            path = line[6:].strip()
        elif line.startswith("+") and not line.startswith("+++") and path:
            added.setdefault(path, []).append(line[1:])
    listing, sources = [], []
    for path, new_lines in added.items():
        old = ""
        if f"./{path}" in names_in_tar:
            old = snapshot.extractfile(f"./{path}").read().decode("utf-8", "replace")
        existing = set(re.findall(r"def\s+(test\w*)", old))
        # Split the added lines into def-blocks and keep the ones that define a new test.
        blocks, cur = [], []
        for line in new_lines:
            if re.match(r"\s*(?:async\s+)?def\s+test\w*", line) and cur:
                blocks.append(cur)
                cur = []
            cur.append(line)
        blocks.append(cur)
        for block in blocks:
            m = re.match(r"\s*(?:async\s+)?def\s+(test\w*)", block[0])
            if m and m[1] not in existing:
                listing.append(f"`{path}::{m[1]}`")
                sources.append("\n".join(block).rstrip())
    if not listing:
        return o3_modified(task)
    source = "\n\n".join(sources)
    if len(source) > MAX_TEST_CHARS:
        source = source[:MAX_TEST_CHARS] + "\n# ... (truncated)"
    return ("The fix must make these new tests pass (they are not in the repository):\n- " + "\n- ".join(listing)
            + "\n\nTest source:\n```python\n" + source + "\n```")


def o3_modified(task: dict) -> str:
    """Fallback when test_patch only changes existing tests: the changed test functions and the diff itself."""
    names, path = [], None
    for line in task["test_patch"].splitlines():
        if line.startswith("+++ b/"):
            path = line[6:].strip()
        elif (m := HUNK.match(line)) and path:
            func = re.search(r"\bdef\s+(test\w*)", m[1])
            names.append(f"`{path}::{func[1]}`" if func else f"`{path}`")
    diff = task["test_patch"]
    if len(diff) > MAX_TEST_CHARS:
        diff = diff[:MAX_TEST_CHARS] + "\n# ... (truncated)"
    return ("The fix must make these changed tests pass (the change is not in the repository):\n- "
            + "\n- ".join(dict.fromkeys(names)) + "\n\nTest change:\n```diff\n" + diff + "\n```")


def data_name(task: dict) -> str:
    return f"{task['instance_id'].rsplit('_', 1)[0]}_{task['base_commit']}"


def main() -> None:
    ids = json.loads((ROOT / "harness" / "subsets" / "dev20.json").read_text(encoding="utf-8"))["task_ids"]
    tasks = [t for t in map(json.loads, open(COMP / "tasks.jsonl", encoding="utf-8")) if t["instance_id"] in ids]
    assert len(tasks) == len(ids), "dev20 task missing from tasks.jsonl"
    for variant in ("placebo", "O1", "O2", "O3"):
        d = OUT / variant
        shutil.rmtree(d, ignore_errors=True)
        (d / "graphs").mkdir(parents=True)
        (d / "embeddings").mkdir()
        empty = []
        with open(d / "tasks.jsonl", "w", encoding="utf-8") as f:
            for t in tasks:
                t = dict(t)
                name = data_name(t)
                if variant != "placebo":
                    shutil.copy(COMP / "graphs" / f"{name}.json", d / "graphs")
                    shutil.copy(COMP / "embeddings" / f"{name}.npz", d / "embeddings")
                graph = json.loads((COMP / "graphs" / f"{name}.json").read_text(encoding="utf-8"))
                if variant == "O1":
                    t["hints_text"] = o1_hint(t, graph)
                elif variant == "O2":
                    t["hints_text"] = o2_hint(t, graph)
                elif variant == "O3":
                    t["hints_text"] = o3_hint(t)
                if variant != "placebo" and not t["hints_text"]:
                    empty.append(t["instance_id"])
                f.write(json.dumps(t) + "\n")
        print(f"{variant}: {len(tasks)} tasks" + (f", no hint for {empty}" if empty else ""))


if __name__ == "__main__":
    main()
