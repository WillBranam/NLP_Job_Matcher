"""Execute every cell in the project environment and save authentic outputs."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import zipfile

from _bootstrap import ROOT
import nbformat
from nbclient import NotebookClient
from nbconvert import HTMLExporter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=ROOT / "notebooks/01_colab_smoke.ipynb")
    parser.add_argument("--output", type=Path, default=ROOT / "notebooks/01_midterm_mac_executed.ipynb")
    args = parser.parse_args()
    cache = ROOT / ".cache/notebook"
    kernel = cache / "data/kernels/job-matcher"
    kernel.mkdir(parents=True, exist_ok=True)
    (kernel / "kernel.json").write_text(json.dumps({
        "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
        "display_name": "NLP Job Matcher (project Python)", "language": "python"}) + "\n")
    for key, value in {"JUPYTER_PATH": cache / "data", "JUPYTER_RUNTIME_DIR": cache / "runtime",
                       "JUPYTER_CONFIG_DIR": cache / "config", "IPYTHONDIR": cache / "ipython"}.items():
        value.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(value)
    notebook = nbformat.read(args.input, as_version=4)
    notebook.metadata["execution_record"] = {
        "started_at": datetime.now(timezone.utc).isoformat(), "allow_errors": False}

    def progress(cell, cell_index):
        if cell.cell_type == "code":
            print(f"Running cell {cell_index + 1}: {cell.source.splitlines()[0]}", flush=True)

    client = NotebookClient(notebook, kernel_name="job-matcher", timeout=1800,
                            allow_errors=False, store_widget_state=False,
                            resources={"metadata": {"path": str(ROOT)}}, on_cell_start=progress)
    try:
        client.execute()
    except Exception:
        failed = args.output.with_name(args.output.stem + "_failed.ipynb")
        nbformat.write(notebook, failed)
        print(f"Execution stopped. Real partial outputs saved to {failed}", flush=True)
        raise
    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    assert all(cell.execution_count is not None for cell in code_cells), "Unexecuted code cell"
    assert not any(output.output_type == "error" for cell in code_cells for output in cell.outputs)
    assert all(cell.outputs for cell in code_cells), "A code cell has no saved output"
    notebook.metadata["execution_record"].update({
        "completed_at": datetime.now(timezone.utc).isoformat(), "code_cells_executed": len(code_cells),
        "error_outputs": 0, "all_code_cells_have_outputs": True})
    notebook.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3 (project .venv)", "language": "python"}
    nbformat.validate(notebook)
    nbformat.write(notebook, args.output)
    html, _ = HTMLExporter(template_name="lab").from_notebook_node(notebook)
    html_path = args.output.with_suffix(".html")
    html_path.write_text(html)
    evidence = ROOT / "outputs/notebook_run"
    evidence.mkdir(parents=True, exist_ok=True)
    (evidence / "execution_verification.json").write_text(
        json.dumps(notebook.metadata["execution_record"], indent=2) + "\n")
    # This final package is built AFTER saving the executed notebook, so it includes
    # the final cell's actual output rather than an earlier notebook snapshot.
    archive_path = ROOT / "notebooks/midterm_submission.zip"
    paths = [args.output, html_path, ROOT / "README.md", ROOT / "AGENT.md", ROOT / "FinalReport.md",
             ROOT / "requirements.txt", ROOT / "requirements-lock.txt", ROOT / "requirements-notebook.txt",
             ROOT / "reports/MidtermScreenshotGuide.txt"]
    for folder in ["scripts", "src/job_matcher", "configs", "tests", "examples", "outputs/notebook_run",
                   "models/adapters/notebook_smoke"]:
        paths.extend(path for path in (ROOT / folder).rglob("*")
                     if path.is_file() and "__pycache__" not in path.parts and path.name != ".gitkeep")
    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in paths:
            if path.exists():
                archive.write(path, path.relative_to(ROOT))
    print(f"SUCCESS: {len(code_cells)} code cells executed; no error outputs.", flush=True)
    print(f"Notebook: {args.output}\nHTML: {html_path}\nSubmission package: {archive_path}", flush=True)


if __name__ == "__main__":
    main()
