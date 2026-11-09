"""Gold/null sanity check on the official swegemma harness.

gold: the agent phase is replaced by the task's own gold patch, so every stable task should resolve.
null: no patch (swegemma's --skip-agent-patch), so no task should resolve.
Anything else means the local sandbox differs from the scorer for that task.

    .venv-harness/Scripts/python harness/gold_check.py gold --subset harness/subsets/dev20.json \
        --results-dir experiments/20261021-gold-dev20 [extra swegemma eval flags]
"""

import argparse
import json
import sys

from adk_eval_core.tracing import SessionTrace
from swegemma import cli, evaluate


async def _gold_patch(self, task, *args, **kwargs):
    return task.patch, None, SessionTrace()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["gold", "null"])
    p.add_argument("--subset", help="JSON file with a task_ids list")
    p.add_argument("--results-dir", required=True)
    p.add_argument("--tasks", default="data/competition/tasks.jsonl")
    p.add_argument("--snapshots-dir", default="data/competition/snapshots")
    p.add_argument("--submission-dir", default="submission")
    p.add_argument("--official-env", action="store_true", help="skip harness/local_env.py fix-ups")
    args, extra = p.parse_known_args()

    if not args.official_env:
        import local_env
        local_env.patch()
        if "--image" not in extra:
            extra += ["--image", "swebench-sandbox:local"]

    argv = ["eval", "--tasks", args.tasks, "--snapshots-dir", args.snapshots_dir,
            "--submission-dir", args.submission_dir, "--results-dir", args.results_dir,
            "--display", "quiet", *extra]
    if args.subset:
        argv += ["--task-ids", *json.load(open(args.subset, encoding="utf-8"))["task_ids"]]
    if args.mode == "gold":
        evaluate.Evaluator._run_agent_sandbox = _gold_patch
    else:
        argv.append("--skip-agent-patch")
    return cli.main(argv)


if __name__ == "__main__":
    sys.exit(main())
