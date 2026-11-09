"""Build a private Kaggle GPU notebook that runs one submission on dev20 the way the scorer does.

    python -I kaggle/make_eval_notebook.py --submission-dir experiments/20261022-e4b-kagglefix-s0/submission \
        --slug gemma4-dev20-kagglefix
    .venv/Scripts/kaggle kernels push -p kaggle/build/gemma4-dev20-kagglefix     # runs it (uses GPU quota)
    .venv/Scripts/kaggle kernels output YOUR_KAGGLE_USERNAME/gemma4-dev20-kagglefix -p <run>/raw-kaggle

The notebook installs the official wheelhouse, serves the declared base model with vLLM using the scorer's
settings (TP=4, max_model_len 32768, gemma4 tool/reasoning parsers, 0.90 GPU memory), and calls swegemma's
`Evaluator.run()` on the frozen dev20 ids with the subprocess sandbox and the submission's own eval_config
budgets, as the organizers' "Getting Started" notebook does. It writes the usual results dir
(task_results.jsonl, patches/, traces/) to /kaggle/working/results, so harness/postprocess.py reads it as is.
The submission files are embedded in the notebook; nothing else from this repo is uploaded.
"""

import argparse
import base64
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
USER = "YOUR_KAGGLE_USERNAME"
# Same image as the organizers' "Getting Started" notebook; the wheelhouse's vLLM/torch wheels are built for it.
DOCKER_IMAGE = "gcr.io/kaggle-private-byod/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461"
HOST_PROVIDED = {"pygments", "rich"}  # test-dep wheels the Kaggle image already has; left out of the notebook source

SETUP = r'''import glob, importlib, json, os, shutil, subprocess, sys, time
from pathlib import Path

for k, v in {"TRANSFORMERS_NO_TF": "1", "TRANSFORMERS_NO_FLAX": "1", "TRANSFORMERS_OFFLINE": "1",
             "VLLM_NO_USAGE_STATS": "1", "VLLM_WORKER_MULTIPROC_METHOD": "spawn", "VLLM_ENGINE_READY_TIMEOUT_S": "1200",
             "VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS": "1", "NCCL_DEBUG": "WARN", "LITELLM_LOCAL_MODEL_COST_MAP": "True",
             "LITELLM_TELEMETRY": "False", "LITELLM_LOG": "WARNING", "OPENAI_API_KEY": "EMPTY", "LOCAL_API_KEY": "EMPTY",
             "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True", "OTEL_SDK_DISABLED": "true"}.items():
    os.environ.setdefault(k, v)

WHEELHOUSE = Path("/kaggle/input/datasets/metric/gemma-4-developer-agent-wheelhouse")
if not WHEELHOUSE.exists():  # older mount layout
    WHEELHOUSE = Path("/kaggle/input/gemma-4-developer-agent-wheelhouse")
for pattern in ("/usr/local/lib/python*/dist-packages/*cutlass*.pth", "/usr/local/lib/python*/site-packages/*cutlass*.pth"):
    for pth in glob.glob(pattern):
        os.unlink(pth)
# Kaggle dataset uploads strip the '+' from '+cu128' wheel names; restore it so pip accepts them.
tmp = Path("/tmp/wheelhouse"); tmp.mkdir(parents=True, exist_ok=True)
for w in WHEELHOUSE.glob("*.whl"):
    if "cutlass" in w.name.lower():
        continue
    name = w.name.replace("cu128", "+cu128") if ("cu128" in w.name and "+" not in w.name) else w.name
    if not (tmp / name).exists():
        shutil.copyfile(w, tmp / name)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "--force-reinstall",
                *sorted(map(str, tmp.glob("*.whl")))], check=True)
importlib.invalidate_caches()
T0 = time.time()
print("wheelhouse installed")
'''

# Test-only deps that the public wheel cache lacks (see experiments/20261021-envcheck-dev20 and
# harness/docker/Dockerfile.local). Without them 7 of the 13 dev20 fastapi tasks can't even collect tests.
# The subprocess sandbox skips install_test_dependencies, and its task venvs see the host's site-packages, so the
# common ones go into the host; inline-snapshot is fastapi-only (its pytest plugin breaks rich), added per task via
# a .pth, like harness/sandbox_fixup.py does locally.
TEST_DEPS = r'''import base64, glob, importlib.util
from swegemma.harness import agent_runner, container_setup, sample_verification, verification
from swegemma import evaluate as _evaluate

DEPS = Path("/tmp/test_deps")
for rel, b64 in json.loads(__WHEELS__).items():
    p = DEPS / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(base64.b64decode(b64))
for whl in sorted((DEPS / "common").glob("*.whl")):  # ujson: one wheel per CPython, so let pip pick
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "--no-index", str(whl)], check=False,
                   capture_output=True)
EXTRAS = Path("/kaggle/working/extras")
for repo in [d.name for d in DEPS.iterdir() if d.name != "common"]:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "--no-index", "--target",
                    str(EXTRAS / repo), *map(str, (DEPS / repo).glob("*.whl"))], check=True)
importlib.invalidate_caches()

_original_install = container_setup.install_test_dependencies


def _install_test_dependencies(docker, container_id, repo="", **kwargs):
    _original_install(docker, container_id, repo, **kwargs)
    extras = EXTRAS / repo.split("/")[-1]
    paths = getattr(docker, "sandboxes", {}).get(container_id)
    if not paths:
        return
    for sp in glob.glob(str(paths["venv"] / "lib" / "python*" / "site-packages")):
        # This sandbox skips `pip install -e /workspace`, so a src/ repo (requests, httpx) would import the host's copy
        # and never see the agent's patch. Sorts before swegemma's _host_env.pth; ignored when there's no src/.
        Path(sp, "00_workspace_src.pth").write_text(f"{paths['workspace'] / 'src'}\n")
        if extras.is_dir():
            Path(sp, "zz_test_extras.pth").write_text(f"{extras}\n")


for mod in (container_setup, _evaluate, agent_runner, verification, sample_verification):
    if hasattr(mod, "install_test_dependencies"):
        setattr(mod, "install_test_dependencies", _install_test_dependencies)
for name in ("dirty_equals", "typing_inspection", "multipart", "ujson"):
    print(name, "ok" if importlib.util.find_spec(name) else "MISSING")
print("extras:", sorted(p.name for p in EXTRAS.iterdir()))
# pygments and rich aren't embedded (1 MB source limit); fail now, not after 1.5 h of tasks, if the image lacks them.
check = subprocess.run([sys.executable, "-c", f"import sys; sys.path.append({str(EXTRAS / 'fastapi')!r}); "
                        "import inline_snapshot, rich, pygments; print(inline_snapshot.__file__, rich.__file__)"],
                       capture_output=True, text=True)
print(check.stdout, check.stderr[-2000:])
assert check.returncode == 0, "inline_snapshot can't import with the host's rich/pygments"
'''

SUBMISSION = r'''DATA_DIR = Path("/kaggle/input/competitions/gemma-4-developer-agent")
if not DATA_DIR.exists():
    DATA_DIR = Path("/kaggle/input/gemma-4-developer-agent")
WORK = Path("/kaggle/working")
AGENT_DIR = WORK / "submission"
shutil.rmtree(AGENT_DIR, ignore_errors=True)
FILES = json.loads(__FILES__)
for rel, text in FILES.items():
    p = AGENT_DIR / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
TASK_IDS = __TASK_IDS__
LABEL = __LABEL__
print(f"{LABEL}: {len(FILES)} submission files, {len(TASK_IDS)} tasks")
'''

SERVER = r'''import litellm, torch
from adk_submission import VllmConfig, VllmServer, discover_adapters
from adk_submission.discovery import discover_declared_models
from swegemma.config import ALLOWED_ADAPTER_EXTENSIONS

litellm.drop_params = True
models_declared = discover_declared_models(AGENT_DIR)
assert len(models_declared) == 1, models_declared
MODEL = next(iter(models_declared))
MODEL_PATH = Path(f"/kaggle/input/models/google/gemma-4/other/{MODEL}/2")
assert MODEL_PATH.exists(), MODEL_PATH
EXTRA = {"gemma-4-12b-it-qat-w4a16-ct": ("--hf-overrides", '{"architectures": ["Gemma4ForCausalLM"], "model_type": "gemma4"}')}
adapters = discover_adapters(str(AGENT_DIR), adapter_extensions=ALLOWED_ADAPTER_EXTENSIONS)
adapter_map = getattr(adapters, "adapters", None) or {}
ranks = [r for a in adapter_map.values() if (r := getattr(a, "rank", None))]
gpus = torch.cuda.device_count()
server = VllmServer(VllmConfig(
    model=str(MODEL_PATH), port=8000, host="127.0.0.1", tool_call_parser="gemma4", reasoning_parser="gemma4",
    max_model_len=32768, dtype="bfloat16" if torch.cuda.is_bf16_supported() else "auto",
    gpu_memory_utilization=0.90, enable_auto_tool_choice=True, enable_lora=bool(adapter_map),
    max_loras=max(1, len(adapter_map)), max_lora_rank=max(ranks) if ranks else 128,
    scheduler_reserve_full_isl=False, tensor_parallel_size=4 if gpus >= 4 else (2 if gpus >= 2 else 1),
    startup_timeout=60 * 20, extra_args=EXTRA.get(MODEL, ()),
), adapter_manifest=adapters)
server.start()
models = server.create_model_registry(aliases=[MODEL], model_prefix="openai/", api_key="EMPTY")
print(f"vLLM up: {MODEL}, {gpus} GPUs, {time.time() - T0:.0f}s since setup")
'''

EVAL = r'''import asyncio, yaml
from google.adk.agents.context_cache_config import ContextCacheConfig
from google.adk.apps._configs import EventsCompactionConfig
from swegemma.config import EvalConfig, build_submission_limits
from swegemma.evaluate import Evaluator

ev = (yaml.safe_load((AGENT_DIR / "eval_config.yaml").read_text()) or {}).get("evaluation", {}) \
    if (AGENT_DIR / "eval_config.yaml").exists() else {}
limits, gen = build_submission_limits()
cfg = EvalConfig(
    tasks_path=DATA_DIR / "tasks.jsonl", snapshots_dir=DATA_DIR / "snapshots", results_dir=WORK / "results",
    submission_dir=AGENT_DIR, models=models, sandbox="subprocess",
    timeout_seconds=int(ev.get("timeout_seconds", 300)),
    verification_timeout_seconds=int(ev.get("verification_timeout_seconds", 300)),
    max_time_minutes=float(ev.get("max_time_minutes", 60.0)), max_tool_calls=int(ev.get("max_tool_calls", 100)),
    max_turns=ev.get("max_turns"), task_ids=TASK_IDS, limits=limits, generation_constraints=gen,
    adapter_manifest=adapters,
    context_cache_config=ContextCacheConfig(min_tokens=2048, ttl_seconds=1800, cache_intervals=10),
    events_compaction_config=EventsCompactionConfig(compaction_interval=15, overlap_size=2, token_threshold=14336,
                                                    event_retention_size=5),
    graph_dir=DATA_DIR / "graphs", embeddings_dir=DATA_DIR / "embeddings", wheels_dir=DATA_DIR / "wheels",
    verbose=False, display_mode="quiet",
)
print("budgets:", {k: ev.get(k) for k in ("max_time_minutes", "max_tool_calls", "max_turns", "timeout_seconds")})


def run(coro):
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(1) as pool:
        return pool.submit(asyncio.run, coro).result()


result = run(Evaluator(cfg).run())
print(f"{LABEL}: {result.resolved}/{result.total} resolved ({result.resolution_rate:.2%}), "
      f"{time.time() - T0:.0f}s total")
'''

REPORT = r'''import collections
rows = [json.loads(l) for l in open(WORK / "results" / "task_results.jsonl")]
for r in rows:
    print(r["instance_id"], "RESOLVED" if r["resolved"] else "-", f'{float(r.get("duration_seconds") or 0):.0f}s',
          (r.get("error") or "")[:100])
print(collections.Counter((r.get("error") or "ok")[:40] for r in rows).most_common())
server.stop()
'''

# No model and no GPU: the agent phase returns the task's gold patch, or nothing with skip_agent_patch. Every usable
# task should resolve on gold and fail on null, in the same sandbox and with the same test deps as a real run.
GOLD = r'''import asyncio, collections, concurrent.futures
from adk_eval_core.tracing import SessionTrace
from adk_submission.registry import ModelRegistry
from swegemma import evaluate as _evaluate_mod
from swegemma.config import EvalConfig, build_submission_limits
from swegemma.evaluate import Evaluator


async def _gold_patch(self, task, *args, **kwargs):
    return task.patch, None, SessionTrace()


_evaluate_mod.Evaluator._run_agent_sandbox = _gold_patch
limits, gen = build_submission_limits()
report = {}
for mode in ("gold", "null"):
    cfg = EvalConfig(
        tasks_path=DATA_DIR / "tasks.jsonl", snapshots_dir=DATA_DIR / "snapshots", results_dir=WORK / f"results_{mode}",
        submission_dir=AGENT_DIR, models=ModelRegistry(), sandbox="subprocess", timeout_seconds=300,
        verification_timeout_seconds=300, max_time_minutes=5.0, max_tool_calls=1, task_ids=TASK_IDS, limits=limits,
        generation_constraints=gen, skip_agent_patch=mode == "null", graph_dir=DATA_DIR / "graphs",
        embeddings_dir=DATA_DIR / "embeddings", wheels_dir=DATA_DIR / "wheels", verbose=False, display_mode="quiet",
    )
    with concurrent.futures.ThreadPoolExecutor(1) as pool:
        result = pool.submit(asyncio.run, Evaluator(cfg).run()).result()
    print(f"{mode}: {result.resolved}/{result.total} resolved, {time.time() - T0:.0f}s since setup")
    for line in open(WORK / f"results_{mode}" / "task_results.jsonl"):
        r = json.loads(line)
        report.setdefault(r["instance_id"], {})[mode] = {"resolved": r["resolved"], "exit": r.get("test_exit_code"),
                                                         "seconds": r.get("duration_seconds"), "error": r.get("error")}
(WORK / "gold_report.json").write_text(json.dumps(report, indent=1))
for task, modes in sorted(report.items()):
    usable = modes.get("gold", {}).get("resolved") and not modes.get("null", {}).get("resolved")
    print("OK " if usable else "BAD", task, modes)
print(collections.Counter(bool(m.get("gold", {}).get("resolved")) and not m.get("null", {}).get("resolved")
                          for m in report.values()))
'''


def code_cell(src: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": src.splitlines(keepends=True)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--submission-dir", required=True)
    ap.add_argument("--slug", required=True, help="kernel slug, e.g. gemma4-dev20-kagglefix")
    ap.add_argument("--subset", default="harness/subsets/dev20.json")
    ap.add_argument("--gold-check", action="store_true", help="CPU notebook, no GPU quota: gold and null patches only")
    args = ap.parse_args()

    sub = ROOT / args.submission_dir
    files = {p.relative_to(sub).as_posix(): p.read_text(encoding="utf-8")
             for p in sorted(sub.rglob("*")) if p.is_file()}
    assert "agent.yaml" in files and all(not f.endswith(".safetensors") for f in files), "text-only submissions"
    ids = json.loads((ROOT / args.subset).read_text(encoding="utf-8"))["task_ids"]
    deps = ROOT / "kaggle" / "test_deps"
    # Kaggle rejects kernel sources over 1 MB. pygments and rich (1.5 MB of wheels) are already in the Kaggle image,
    # and the task venvs inherit its site-packages; the notebook checks that inline_snapshot imports before vLLM starts.
    wheels = {p.relative_to(deps).as_posix(): base64.b64encode(p.read_bytes()).decode()
              for p in sorted(deps.rglob("*.whl")) if p.name.split("-")[0].lower() not in HOST_PROVIDED}
    test_deps_cell = TEST_DEPS.replace("__WHEELS__", repr(json.dumps(wheels)))
    submission_cell = (SUBMISSION.replace("__FILES__", repr(json.dumps(files)))
                       .replace("__TASK_IDS__", repr(ids)).replace("__LABEL__", repr(args.slug)))
    cells = (GOLD,) if args.gold_check else (SERVER, EVAL, REPORT)
    nb = {"cells": [{"cell_type": "markdown", "metadata": {},
                     "source": [f"# {args.slug}\n", f"{Path(args.subset).stem} eval of `{args.submission_dir}`, built by "
                                "`kaggle/make_eval_notebook.py`."]},
                    *map(code_cell, (SETUP, test_deps_cell, submission_cell, *cells))],
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
          "nbformat": 4, "nbformat_minor": 5}
    out = ROOT / "kaggle" / "build" / args.slug
    out.mkdir(parents=True, exist_ok=True)
    source = json.dumps(nb, indent=1)
    assert len(source.encode("utf-8")) < 1_000_000, f"notebook is {len(source):,} bytes; Kaggle's limit is 1 MB"
    (out / f"{args.slug}.ipynb").write_text(source, encoding="utf-8")
    meta = {
        "id": f"{USER}/{args.slug}", "title": args.slug, "code_file": f"{args.slug}.ipynb",
        "language": "python", "kernel_type": "notebook", "is_private": True,
        "enable_gpu": True, "enable_tpu": False, "enable_internet": False, "machine_shape": "NvidiaL4",
        "docker_image": DOCKER_IMAGE,
        "dataset_sources": ["metric/gemma-4-developer-agent-wheelhouse"],
        "competition_sources": ["gemma-4-developer-agent"], "kernel_sources": [],
        "model_sources": [f"google/gemma-4/Other/{m}/2" for m in
                          ("gemma-4-31b-it-qat-w4a16-ct", "gemma-4-12b-it-qat-w4a16-ct",
                           "gemma-4-e4b-it-qat-w4a16-ct", "gemma-4-e2b-it-qat-w4a16-ct")],
    }
    if args.gold_check:
        meta.update(enable_gpu=False, model_sources=[])
        del meta["machine_shape"]
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"wrote {out} ({len(files)} submission files, {len(ids)} tasks)")


if __name__ == "__main__":
    main()
