"""Run the official `swegemma eval` with our local sandbox fix-ups and the local Ollama model.

    .venv-harness/Scripts/python harness/run_eval.py --subset harness/subsets/dev20.json \
        --results-dir experiments/20261022-baseline-e4b/raw [extra swegemma eval flags, e.g. --task-ids X]

Budgets come from <submission-dir>/eval_config.yaml unless passed as flags. The scorer's inference.py loads
that file itself, but the `swegemma eval` CLI doesn't, so without this the local run falls back to swegemma's
defaults (60 min, no tool-call or turn cap, 300 s per command).

Before the run, every Ollama model in --models-yaml is loaded once and must sit fully in VRAM; a partial CPU
offload would make the time budget mean something else, so the run stops instead (skip with --allow-cpu).

--scorer-compaction turns on the context compaction the scorer uses (HARNESS_README section 7.2). The `swegemma eval`
CLI leaves it off, so without this flag a local agent never loses context the way a scored one does.
"""

import argparse
import functools
import json
import sys
import urllib.request
from pathlib import Path

import local_env
import yaml
from swegemma import cli

# eval_config.yaml `evaluation:` key -> swegemma eval flag (HARNESS_README section 7.1)
BUDGET_FLAGS = {
    "max_time_minutes": "--max-time-minutes",
    "max_tool_calls": "--max-tool-calls",
    "max_turns": "--max-turns",
    "timeout_seconds": "--timeout-seconds",
}


def budget_flags(submission_dir: str, extra: list[str]) -> list[str]:
    path = Path(submission_dir) / "eval_config.yaml"
    if not path.exists():
        return []
    evaluation = (yaml.safe_load(path.read_text(encoding="utf-8")) or {}).get("evaluation") or {}
    flags = []
    for key, flag in BUDGET_FLAGS.items():
        if evaluation.get(key) is not None and not any(a == flag or a.startswith(flag + "=") for a in extra):
            flags += [flag, str(evaluation[key])]
    return flags


def _ollama_json(api_base: str, path: str, body: dict | None = None) -> dict:
    url = api_base.rstrip("/").removesuffix("/v1") + path
    data = json.dumps(body).encode() if body is not None else None
    with urllib.request.urlopen(urllib.request.Request(url, data=data), timeout=300) as r:
        return json.load(r)


def check_ollama_gpu(models_yaml: str) -> list[str]:
    """Load each Ollama model behind --models-yaml and report any that isn't 100% in VRAM."""
    entries = (yaml.safe_load(Path(models_yaml).read_text(encoding="utf-8")) or {}).get("models") or {}
    targets = {(e["api_base"], e["path"].split("/", 1)[1]) for e in entries.values()
               if "11434" in e.get("api_base", "")}
    problems = []
    for api_base, model in sorted(targets):
        _ollama_json(api_base, "/api/generate", {"model": model, "prompt": "hi", "stream": False,
                                                 "options": {"num_predict": 1}})
        loaded = {m["name"].removesuffix(":latest"): m for m in _ollama_json(api_base, "/api/ps")["models"]}
        m = loaded.get(model.removesuffix(":latest"))
        if m is None or m["size_vram"] < m["size"]:
            vram = 0 if m is None else m["size_vram"] / m["size"]
            problems.append(f"{model}: {vram:.0%} in VRAM (rest on CPU); free GPU memory or restart Ollama")
        else:
            print(f"ollama {model}: 100% GPU")
    return problems


def enable_scorer_compaction() -> None:
    """Make the CLI's EvalConfig use the scorer's EventsCompactionConfig, as kaggle/make_eval_notebook.py does."""
    from google.adk.apps._configs import EventsCompactionConfig

    cli.EvalConfig = functools.partial(
        cli.EvalConfig,
        events_compaction_config=EventsCompactionConfig(
            compaction_interval=15, overlap_size=2, token_threshold=14336, event_retention_size=5
        ),
    )


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--subset", help="JSON file with a task_ids list")
    p.add_argument("--results-dir", required=True)
    p.add_argument("--tasks", default="data/competition/tasks.jsonl")
    p.add_argument("--snapshots-dir", default="data/competition/snapshots")
    p.add_argument("--submission-dir", default="submission")
    p.add_argument("--models-yaml", default="harness/models.ollama.yaml")
    p.add_argument("--official-env", action="store_true", help="skip harness/local_env.py fix-ups")
    p.add_argument("--allow-cpu", action="store_true", help="skip the Ollama all-in-VRAM check")
    p.add_argument("--scorer-compaction", action="store_true", help="use the scorer's context compaction")
    args, extra = p.parse_known_args()

    if not args.allow_cpu and (problems := check_ollama_gpu(args.models_yaml)):
        print("\n".join(problems), file=sys.stderr)
        return 2

    if args.scorer_compaction:
        enable_scorer_compaction()

    if not args.official_env:
        local_env.patch()
        if "--image" not in extra:
            extra += ["--image", "swebench-sandbox:local"]

    argv = ["eval", "--tasks", args.tasks, "--snapshots-dir", args.snapshots_dir,
            "--submission-dir", args.submission_dir, "--results-dir", args.results_dir,
            "--models-yaml", args.models_yaml, "--display", "quiet",
            *budget_flags(args.submission_dir, extra), *extra]
    if args.subset:
        argv += ["--task-ids", *json.load(open(args.subset, encoding="utf-8"))["task_ids"]]
    return cli.main(argv)


if __name__ == "__main__":
    sys.exit(main())
