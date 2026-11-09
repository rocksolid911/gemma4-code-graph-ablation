"""Local-only fix-ups of the official sandbox setup, applied on top of swegemma without editing it.

The public setup unpacks the NEWEST cached version of every wheel into site-packages (e.g. starlette 1.6,
which breaks older fastapi commits) and leaves cached copies of the repo's own package ahead of /workspace
on sys.path (requests, httpx). After swegemma's own install step we run harness/sandbox_fixup.py inside the
container to restore the repo's pinned versions and per-repo test extras. This is a known deviation from
the scorer (whose prebaked image we can't see); it only changes installed packages, never the tools,
prompts or budgets the agent sees. Use with the swebench-sandbox:local image (harness/docker/Dockerfile.local).

It also fixes a Windows-host bug in swegemma's file tools (see _resolve_workspace_path below); the Linux scorer
doesn't need it.
"""

import posixpath
from pathlib import Path, PurePosixPath

from swegemma import evaluate
from swegemma.harness import agent_runner, container_setup, verification
from swegemma.tools import workspace

FIXUP = Path(__file__).with_name("sandbox_fixup.py")
DEBUG_LOG = None  # set to a path to append each container's fix-up output (debugging only)

_original = container_setup.install_test_dependencies


def _install_test_dependencies(docker, container_id, repo="", **kwargs):
    _original(docker, container_id, repo, **kwargs)
    docker.copy_to(container_id, FIXUP, "/tmp/_sandbox_fixup.py")
    res = docker.exec(container_id, f"python3 /tmp/_sandbox_fixup.py {repo} 2>&1; rm -f /tmp/_sandbox_fixup.py")
    if DEBUG_LOG:
        with open(DEBUG_LOG, "a", encoding="utf-8") as f:
            f.write(f"=== {repo} {container_id[:12]}\n{res.stdout}\n")


def _resolve_workspace_path(filepath, workspace_root=None, *, allow_root=True):
    """POSIX twin of swegemma.tools.workspace._resolve_workspace_path.

    The original builds container paths with pathlib.Path and .resolve()s them on the HOST, which on Windows
    turns `/workspace/pkg/module.py` into `C:\\workspace\\pkg\\module.py`, so every read_file/edit_file/write_file
    404s. On the Linux scorer .resolve() is only normalisation, which posixpath.normpath reproduces.
    """
    rel = filepath
    if rel == "/workspace":
        rel = ""
    elif rel.startswith("/workspace/"):
        rel = rel[len("/workspace/"):]
    if rel.startswith("/"):
        raise ValueError(f"Path traversal detected: '{filepath}' escapes workspace root.")
    resolved = posixpath.normpath(posixpath.join("/workspace", rel))
    if resolved != "/workspace" and not resolved.startswith("/workspace/"):
        raise ValueError(f"Path traversal detected: '{filepath}' escapes workspace root.")
    if not allow_root and resolved == "/workspace":
        raise ValueError(f"Invalid file path '{filepath}': refers to workspace root directory, not a file.")
    return PurePosixPath(resolved)


def patch() -> None:
    for mod in (container_setup, evaluate, agent_runner, verification):
        if hasattr(mod, "install_test_dependencies"):
            setattr(mod, "install_test_dependencies", _install_test_dependencies)
    workspace._resolve_workspace_path = _resolve_workspace_path
