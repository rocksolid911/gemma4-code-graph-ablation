"""Build submission.zip from submission/ and check it against the competition rules.

    .venv-harness/Scripts/python harness/build_submission.py [--submission-dir submission] [--out submission.zip]

Checks the zip's own entries (no absolute paths, `..`, or symlinks), then unpacks it and runs our rules
(docs/context.md: declarative file types, .py only in skills/*/scripts/, < 3 GiB unpacked) plus the organizers'
validators from the wheelhouse: adk_submission.validate_directory with swegemma's competition limits, and the
single-allowed-base-model check that scripts/inference.py does via discover_declared_models.
Finally it compiles the agent the way the harness does (compile_submission with swegemma's tool names bound to
stubs and the CLI model registry), which catches errors validate_directory misses, such as a bad skill name.
"""

import argparse
import stat
import sys
import tempfile
import zipfile
from pathlib import Path, PurePosixPath

from adk_submission import compile_submission
from adk_submission.discovery import discover_declared_models, validate_directory
from swegemma.cli import _build_cli_model_registry
from swegemma.config import ALLOWED_SUBMISSION_EXTENSIONS, MAX_SUBMISSION_SIZE_BYTES, build_submission_limits

# HARNESS_README section 3 (ALLOWED_MODEL_NAMES lives in scripts/inference.py, which isn't in the wheelhouse)
ALLOWED_MODEL_NAMES = {
    "gemma-4-31b-it-qat-w4a16-ct",
    "gemma-4-12b-it-qat-w4a16-ct",
    "gemma-4-e4b-it-qat-w4a16-ct",
    "gemma-4-e2b-it-qat-w4a16-ct",
}


def check_tree(root: Path) -> list[str]:
    errors, total = [], 0
    for p in sorted(root.rglob("*")):
        rel = p.relative_to(root).as_posix()
        if p.is_symlink():
            errors.append(f"symlink: {rel}")
            continue
        if ".." in p.relative_to(root).parts:
            errors.append(f"'..' in path: {rel}")
        if not p.is_file():
            continue
        total += p.stat().st_size
        if p.suffix.lower() not in ALLOWED_SUBMISSION_EXTENSIONS:
            errors.append(f"disallowed file type: {rel}")
        parts = PurePosixPath(rel).parts
        if p.suffix.lower() == ".py" and not (len(parts) >= 4 and parts[0] == "skills" and parts[2] == "scripts"):
            errors.append(f".py outside skills/*/scripts/: {rel}")
    if total >= MAX_SUBMISSION_SIZE_BYTES:
        errors.append(f"unpacked size {total:,} B >= 3 GiB")
    print(f"  {sum(1 for p in root.rglob('*') if p.is_file())} files, {total:,} bytes unpacked")

    try:
        validate_directory(root, build_submission_limits()[0])
    except Exception as e:
        errors.append(f"adk_submission.validate_directory: {e}")
    try:
        models = discover_declared_models(root)
        print(f"  declared models: {sorted(models)}")
        if len(models) != 1 or not models <= ALLOWED_MODEL_NAMES:
            errors.append(f"need exactly one allowed base model, got {sorted(models)}")
    except Exception as e:
        errors.append(f"discover_declared_models: {e}")
    return errors


def stub_tool(name: str):
    def tool(arg: str = "") -> dict:
        """Stub: compiling only needs the name."""
        return {}
    tool.__name__ = name
    return tool


def check_compiles(root: Path) -> list[str]:
    names = ["run_command", "read_file", "write_file", "edit_file", "submit_patch", "get_status",
             "get_code_neighbors", "search_similar_code", "get_code_subgraph"]  # swegemma create_tools()
    try:
        agent = compile_submission(submission_dir=root, tool_registry={n: stub_tool(n) for n in names},
                                   model_registry=_build_cli_model_registry(root))
        print(f"  compiles: {type(agent).__name__} '{agent.name}'")
        return []
    except Exception as e:
        return [f"compile_submission: {e}"]


def check_zip(zip_path: Path) -> list[str]:
    errors = []
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        for info in zf.infolist():
            path = PurePosixPath(info.filename)
            if path.is_absolute() or ".." in path.parts or "\\" in info.filename:
                errors.append(f"bad zip entry name: {info.filename}")
            if stat.S_ISLNK(info.external_attr >> 16):
                errors.append(f"symlink zip entry: {info.filename}")
        if "agent.yaml" not in names:
            errors.append("agent.yaml is not at the zip root")
    return errors


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--submission-dir", default="submission")
    p.add_argument("--out", default="submission.zip")
    args = p.parse_args()
    src, out = Path(args.submission_dir), Path(args.out)

    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in sorted(src.rglob("*")):
            if f.is_file() and not f.is_symlink():
                zf.write(f, f.relative_to(src).as_posix())
    print(f"wrote {out} ({out.stat().st_size:,} bytes)")

    errors = check_zip(out)
    with tempfile.TemporaryDirectory() as tmp:
        zipfile.ZipFile(out).extractall(tmp)
        errors += check_tree(Path(tmp))
        errors += check_compiles(Path(tmp))

    for e in errors:
        print(f"FAIL {e}")
    print("OK" if not errors else f"{len(errors)} problem(s)")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
