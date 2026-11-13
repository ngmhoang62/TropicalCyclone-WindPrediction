"""Check tracked files for accidental secrets, raw observations and heavyweight artifacts."""

import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
names = (
    subprocess.check_output(["git", "ls-files", "--cached", "-z"], cwd=ROOT)
    .decode("utf-8")
    .split("\0")
)
errors = []
for name in filter(None, names):
    path = ROOT / name
    if any(name.startswith(prefix) for prefix in ["data/raw/", "artifacts/", "reports/runs/"]):
        if path.name not in {"README.md", ".gitkeep"}:
            errors.append(f"Private artifact staged: {name}")
    if path.suffix in {".pkl", ".joblib", ".pt", ".keras", ".env"}:
        errors.append(f"Unsafe artifact staged: {name}")
    if not path.exists() or path.stat().st_size > 10_000_000:
        errors.append(f"Missing or oversized file: {name}")
        continue
    if path.suffix in {".py", ".md", ".txt", ".json", ".toml", ".yml", ".ipynb"}:
        text = path.read_text(encoding="utf-8")
        if re.search(r"(?:gh[pousr]_[A-Za-z0-9]{30,}|sk-[A-Za-z0-9]{32,}|AKIA[A-Z0-9]{16})", text):
            errors.append(f"Possible credential: {name}")
        if (
            re.search(r"(?:C:\\Users\\|/content/drive/)", text)
            and "legacy" not in name
            and name != "scripts/check_release.py"
        ):
            errors.append(f"Private machine path: {name}")
    if path.suffix == ".ipynb":
        notebook = json.loads(path.read_text(encoding="utf-8"))
        if any(o.get("output_type") == "error"
               for c in notebook["cells"] for o in c.get("outputs", [])):
            errors.append(f"Failed notebook execution: {name}")
if errors:
    raise SystemExit("\n".join(errors))
print(f"PASS: {len(list(filter(None, names)))} tracked files inspected")
