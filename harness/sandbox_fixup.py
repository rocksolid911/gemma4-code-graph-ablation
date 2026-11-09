"""Runs INSIDE the sandbox container after swegemma's setup (copied in by harness/local_env.py). Offline.

1. Pins: ask pip which versions the repo's own requirements resolve to from /wheels, and swap in any that
   differ from what swegemma unpacked (it unpacks the newest of everything, without RECORD files, so pip
   can't uninstall them itself).
2. Shadowing: delete site-packages copies of the packages the repo itself provides, so `import <pkg>`
   loads /workspace code instead of a cached wheel.
3. Per-repo extras: add /opt/extras/<repo>/ (baked into swebench-sandbox:local) to sys.path via a .pth.
"""

import json
import re
import shutil
import subprocess
import sys
import sysconfig
from importlib import metadata
from pathlib import Path

WS = Path("/workspace")
SP = Path(sysconfig.get_paths()["purelib"])
PIP = [sys.executable, "-m", "pip", "-q", "--disable-pip-version-check", "--no-input"]
OFFLINE = ["--no-index", "--find-links=/wheels", "--no-build-isolation"]


def norm(name: str) -> str:
    return re.sub(r"[-_.]+", "_", name).lower()


def repo_packages() -> set[str]:
    pkgs = set()
    for base in (WS / "src", WS):
        if base.is_dir():
            pkgs |= {p.name for p in base.iterdir() if (p / "__init__.py").is_file() and p.name not in ("tests", "docs", "scripts")}
    return pkgs


def remove_from_site_packages(dist_name: str, top_levels: set[str]) -> None:
    for p in SP.iterdir():
        stem = p.name.split("-")[0]
        if p.name in top_levels or (p.suffix == ".dist-info" and norm(stem) == norm(dist_name)):
            shutil.rmtree(p, ignore_errors=True) if p.is_dir() else p.unlink(missing_ok=True)


def fix_pins(own: set[str]) -> None:
    report = Path("/tmp/_fixup_report.json")
    # Without --ignore-installed: packages that are only in the image (typing-inspection, ...) still count, and
    # the report lists exactly what pip would add, upgrade or downgrade.
    res = subprocess.run([*PIP, "install", *OFFLINE, "--dry-run", "--report", str(report), "-e", str(WS)],
                         check=False, capture_output=True, text=True)
    if not report.exists():
        print(f"fixup: pip could not resolve the repo's requirements offline: {res.stderr.strip()[-400:]}")
        return
    wanted = {}
    for item in json.loads(report.read_text())["install"]:
        md = item["metadata"]
        if item.get("download_info", {}).get("dir_info") is None:  # skip the workspace package itself
            wanted[md["name"]] = md["version"]
    for name, version in wanted.items():
        try:
            have = metadata.version(name)
        except metadata.PackageNotFoundError:
            have = None
        if have == version or norm(name) in {norm(o) for o in own}:
            continue
        top = {norm(name), name.replace("-", "_"), name.lower()}
        remove_from_site_packages(name, top)
        subprocess.run([*PIP, "install", *OFFLINE, "--no-deps", f"{name}=={version}"], check=False, capture_output=True)
        print(f"fixup: {name} {have} -> {version}")


def main(repo: str) -> None:
    own = repo_packages()
    for pkg in own:  # stale cached copies of the repo's own packages
        remove_from_site_packages(pkg, {pkg})
    fix_pins(own)
    extras = Path("/opt/extras") / repo.replace("/", "__")
    if extras.is_dir():
        (SP / "zz_local_extras.pth").write_text(f"{extras}\n")
        print(f"fixup: extras {extras}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "")
