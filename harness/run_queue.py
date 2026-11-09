"""Run a queue of dev20 experiments back to back (overnight GPU use), postprocessing each one.

    .venv-harness/Scripts/python harness/run_queue.py queue.json

queue.json is a list of jobs, run in order (re-read before each job, so jobs can be appended live):
    {"run": "experiments/20261024-e4b-e4on-s1",  # created; must contain submission/ already
     "seed": 1, "parent": "20261023-e4b-e6e7-s0", "diff": "...",
     "tasks": "data/variants/placebo/tasks.jsonl",   # optional, default data/competition/tasks.jsonl
     "max_time_minutes": 15}                        # optional, default 15 (the local comparison cap)

A job whose run dir already has summary.json is skipped, so a crashed queue can be restarted as is.
Before each job, Docker must answer `docker info` (Docker Desktop is started and waited for if not);
run_eval.py itself stops when the Ollama model isn't fully in VRAM. A failed job is logged and the
queue moves on.
"""

import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

PY = sys.executable
DOCKER_DESKTOP = r"C:\Program Files\Docker\Docker\Docker Desktop.exe"


def log(msg: str) -> None:
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}", flush=True)


def docker_ok() -> bool:
    return subprocess.run(["docker", "info"], capture_output=True).returncode == 0


def ensure_docker() -> bool:
    if docker_ok():
        return True
    log("docker not answering; starting Docker Desktop")
    subprocess.Popen([DOCKER_DESKTOP])
    for _ in range(36):
        time.sleep(10)
        if docker_ok():
            return True
    return False


def ensure_ollama() -> bool:
    # After a sleep/reboot the Ollama server may be down; any `ollama` CLI call starts the Windows app again.
    for _ in range(12):
        if subprocess.run(["ollama", "ps"], capture_output=True).returncode == 0:
            return True
        time.sleep(10)
    return False


def run_job(job: dict) -> None:
    run = Path(job["run"])
    if (run / "summary.json").exists():
        log(f"skip {run.name}: already has summary.json")
        return
    if not (run / "submission").is_dir():
        log(f"skip {run.name}: no submission/ snapshot")
        return
    if not ensure_docker():
        log(f"FAIL {run.name}: docker did not come up")
        return
    if not ensure_ollama():
        log(f"FAIL {run.name}: ollama did not come up")
        return
    tasks = job.get("tasks", "data/competition/tasks.jsonl")
    cmd = [PY, "harness/run_eval.py", "--subset", "harness/subsets/dev20.json", "--tasks", tasks,
           "--results-dir", str(run / "raw"), "--submission-dir", str(run / "submission"),
           "--max-time-minutes", str(job.get("max_time_minutes", 15))]
    log(f"start {run.name} ({tasks})")
    with open(run / "console.log", "a", encoding="utf-8") as console:
        console.write(f"started {datetime.now().isoformat()} cap={job.get('max_time_minutes', 15)}min tasks={tasks}\n")
        console.flush()
        rc = subprocess.run(cmd, stdout=console, stderr=subprocess.STDOUT).returncode
        console.write(f"exit {rc} {datetime.now().isoformat()}\n")
    if not (run / "raw" / "task_results.jsonl").exists():
        log(f"FAIL {run.name}: exit {rc}, no task_results.jsonl")
        return
    pp = [PY, "-I", "harness/postprocess.py", str(run), "--model", "e4b-qat-ollama", "--backend", "ollama",
          "--subset", "dev20", "--seed", str(job.get("seed", 0)), "--config-diff", job.get("diff", "")]
    if job.get("parent"):
        pp += ["--parent-run-id", job["parent"]]
    out = subprocess.run(pp, capture_output=True, text=True)
    log(f"done {run.name}: exit {rc}; {out.stdout.strip() or out.stderr.strip()[-300:]}")


def main() -> None:
    # The queue file is re-read before each job, so jobs can be appended while the queue runs.
    queue, attempted = Path(sys.argv[1]), set()
    while True:
        jobs = [j for j in json.loads(queue.read_text(encoding="utf-8")) if j["run"] not in attempted]
        if not jobs:
            break
        attempted.add(jobs[0]["run"])
        run_job(jobs[0])
    log("queue finished")


if __name__ == "__main__":
    main()
