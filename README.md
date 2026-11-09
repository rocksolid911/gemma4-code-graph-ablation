# Finding the Code Isn't the Problem: code and results

Code, configs and numbers behind the Kaggle Paper Track writeup *Finding the Code Isn't the Problem: a placebo-controlled test of code-graph tools for Gemma 4 coding agents under a 32k context*, for Google – The Gemma 4 Developer Agent Competition.

**Licence:** Apache 2.0 (see `LICENSE`).

## Check the paper's numbers

No GPU or competition data needed. Python 3.11 or later, standard library only. Run these from this folder:

```bash
python make_tables.py        # the paper's tables and counts; section K prints "skipped", timing_table.py covers it
python timing_table.py       # Table 4 and the counts in section 3.6
python make_figures.py       # Figures 1 and 2; needs Edge or Chrome to write the PNGs
```

`tables_output.txt` and `timing_output.txt` hold the expected output.

## What's here

| Path | Contents |
|---|---|
| `experiments/<run>/summary.json` | One run's totals |
| `experiments/<run>/tasks/` | Per-task numbers: resolved or not, hits, calls, tokens, time |
| `experiments/<run>/traces/` | Per-call numbers: tool name, success, result size, tokens, time |
| `experiments/<run>/submission/` | The exact agent config, where it can be shared (see below) |
| `experiments/20261104-e4b-oracleO1-s1/` | Also the two parts of this run and its GPU sensor logs |
| `results/timing_200_runs.csv` | Numbers behind Table 4 |
| `harness/` | Our glue around the organizers' evaluator, which is used unchanged, and the frozen `dev20` task list |
| `kaggle/make_eval_notebook.py` | Builds the notebook used for the Kaggle runs |

## What's left out, and why

The competition's data stays with the competition. This folder has no task text, gold patches, test patches, repository snapshots, or graph and embedding files. It also has no model output: no prompts as sent, raw traces, tool arguments, agent patches, test logs or run logs. Gold file names are out too, and the per-task records keep only counts and flags.

Task IDs and `harness/subsets/dev20.json` (each task's repository, number of gold files and lines) are included, because the results can't be read without them.

**Configs not included.** The Budget-Fit and merge prompts reuse text from the organizers' sample submission and from public notebooks, so those snapshots aren't here. Their results are. Sources:

- Budget-Fit Single Agent, by Wilmer E. Henao: Kaggle `verracodeguacas/gemma-4-agent-budget-fit-single-agent`
- Anytime Patch, by leoprovorov
- "LB 0.24" v6, by honghanhhh, on flexonafft v2

## Rerun the experiments

This needs access to the competition, its data and its wheelhouse (`metric/gemma-4-developer-agent-wheelhouse`), plus Docker and Ollama. Put competition files in `data/competition/`, and the wheelhouse in `data/wheelhouse/`. In Git Bash on Windows, from this folder:

```bash
uv venv --python 3.12 .venv-harness
uv pip install --python .venv-harness --find-links data/wheelhouse -r harness/requirements.txt
ollama pull gemma4:e4b-it-qat && ollama create gemma4-e4b-32k -f harness/ollama/Modelfile.e4b-32k
mkdir -p data/build/sandbox && cp data/competition/docker/{Dockerfile.public,imp.py,telnetlib.py} data/build/sandbox/ && cp -r data/competition/wheels data/build/sandbox/
docker build -f data/build/sandbox/Dockerfile.public -t swebench-sandbox:latest data/build/sandbox
docker build -f harness/docker/Dockerfile.local -t swebench-sandbox:local harness/docker
python harness/make_task_variants.py
```

Then, for one local E4B run on the same conditions as the paper:

```bash
.venv-harness/Scripts/python harness/run_eval.py --subset harness/subsets/dev20.json --tasks data/variants/O1/tasks.jsonl --results-dir my-run/raw --submission-dir experiments/20261024-e4b-oracleO1-s0/submission --max-time-minutes 15
.venv-harness/Scripts/python -I harness/postprocess.py my-run --model e4b-qat-ollama --backend ollama --subset dev20 --seed 0
```

Leave out `--max-time-minutes` and the 5 minutes in `eval_config.yaml` applies instead. `harness/run_eval.py` adds the scorer's compaction only with `--scorer-compaction`, and the paper's main arms ran without it. `kaggle/test_deps_needed.txt` lists the wheels the notebook builder expects.

## Conditions to keep in mind

- Laptop runs use Ollama's E4B QAT build on an 8 GB GPU: 15 minutes, 60 calls, no compaction. Ollama ignores the output cap and thinking budget, so these runs stand in for the scorer and don't match it.
- Kaggle runs use vLLM and the scorer's compaction. Runs before 2026-10-28 could score only 13 of 20 tasks, and all of them predate the 2026-11-02 fix for `requests_7309` and `httpx_3672`.
- `seed` on E4B is a repeat index at temperature 0.2, not a sampling seed.
- O1's second run is in two parts, split by a laptop power-off, and the GPU was limited by heat or a power cap for nearly all of it.
- In `make_tables.py`, the laptop's 15-minute runs and the Kaggle runs are kept in separate sections. Don't compare their scores directly.

## Credit

The evaluator (`swegemma`), tools, tasks, graph data and embeddings are the competition organizers'.
