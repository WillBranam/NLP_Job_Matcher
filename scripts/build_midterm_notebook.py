"""Build the Mac/Colab notebook template; execution is a separate explicit step."""
import textwrap

from _bootstrap import ROOT
import nbformat as nbf


def main():
    cells = []

    def md(source):
        cells.append(nbf.v4.new_markdown_cell(textwrap.dedent(source).strip()))

    def code(source):
        cells.append(nbf.v4.new_code_cell(textwrap.dedent(source).strip()))

    md("""
    # NLP Job Matcher — Mac MPS / CUDA / CPU midterm demonstration

    This notebook demonstrates downloaded training/testing data, base-model inference,
    a two-step LoRA training run, and adapter reload. **No accuracy improvement is claimed.**
    Every stage prints its real output; the executed copy retains those outputs.

    **Mac:** open with the project's .venv Python kernel. Device "auto" chooses MPS
    on Apple Silicon when available. You can explicitly choose mps, cuda, or cpu.
    MPS/CPU use float32, batch size 1 and a 1,024-token smoke limit; model stages run
    in separate processes to release memory between stages. CUDA is also supported.

    **Project location:** run inside the repository, or set PROJECT_ROOT in code
    cell 1 to your local checkout. On Colab, upload the complete source project or
    clone a revision containing these scripts before running. Local execution does
    not establish that Colab/CUDA has been tested.

    The IT pilot uses source AI labels and a senior job, so it is only a pipeline
    check. The demonstration resume is synthetic. Review the limitations at the end.
    """)
    md("""
    ## 1. Locate the project and capture subprocess output
    Output from every command is streamed here and saved to a log. Re-running uses
    separate notebook-run outputs and does not replace the original smoke adapter.
    """)
    code(r'''
    from pathlib import Path
    from datetime import datetime, timezone
    import csv, hashlib, html, importlib.metadata, json, os, platform, re, subprocess, sys, time, zipfile
    from IPython.display import HTML, display

    PROJECT_ROOT = None  # Example: "/Users/yourname/path/to/NLP_Job_Matcher"
    candidates = ([Path(PROJECT_ROOT).expanduser()] if PROJECT_ROOT else [])
    candidates += [Path.cwd(), *Path.cwd().parents, Path("/content/NLP_Job_Matcher")]
    ROOT = next((p.resolve() for p in candidates if (p / "scripts/train_lora.py").is_file()), None)
    if ROOT is None:
        raise FileNotFoundError("Set PROJECT_ROOT to the source repository, not just the downloaded notebook.")
    os.chdir(ROOT)
    RUN_DIR = ROOT / "outputs/notebook_run"
    (RUN_DIR / "logs").mkdir(parents=True, exist_ok=True)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["PYTHONUNBUFFERED"] = "1"
    os.environ.setdefault("OMP_NUM_THREADS", "6")
    sys.path.insert(0, str(ROOT / "src"))
    PYTHON = sys.executable
    RUN_STARTED = datetime.now(timezone.utc).isoformat()

    def run_command(arguments, log_name):
        command = [str(arg) for arg in arguments]
        print("Running:", " ".join(command), flush=True)
        lines = []
        with (RUN_DIR / "logs" / log_name).open("w") as log:
            with subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT, text=True, bufsize=1) as process:
                for line in process.stdout:
                    print(line, end="", flush=True)
                    log.write(line)
                    lines.append(line)
                return_code = process.wait()
        if return_code:
            raise subprocess.CalledProcessError(return_code, command, output="".join(lines))
        print("Completed successfully (exit code 0).", flush=True)
        return "".join(lines)

    def show_table(headers, rows):
        head = "".join(f"<th>{html.escape(str(v))}</th>" for v in headers)
        body = "".join("<tr>" + "".join(f"<td>{html.escape(str(v))}</td>" for v in row) + "</tr>" for row in rows)
        display(HTML(f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"))

    print("Project:", ROOT)
    print("Python:", PYTHON)
    print("Run started (UTC):", RUN_STARTED)
    print("Operating system:", platform.system(), platform.machine())
    ''')
    md("""
    ## 2. Check dependencies
    Existing matching versions are reused. A fresh environment installs only missing
    or mismatched training dependencies. No repeated multi-gigabyte PyTorch reinstall
    is needed on this Mac. Use Python 3.11 for the locally verified environment.
    """)
    code(r'''
    expected = {}
    for line in (ROOT / "requirements.txt").read_text().splitlines():
        if "==" in line and not line.lstrip().startswith("#"):
            name, version = line.strip().split("==", 1)
            expected[name] = version
    mismatches = []
    for name, version in expected.items():
        try:
            installed = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            installed = None
        if installed != version:
            mismatches.append((name, installed, version))
    if mismatches:
        print("Installing pinned versions for:", mismatches)
        run_command([PYTHON, "-m", "pip", "install", "--cache-dir", ROOT / ".cache/pip",
                     "-r", ROOT / "requirements.txt"], "dependencies.log")
    else:
        print("All pinned training dependencies already installed; reusing them.")
    show_table(["Package", "Installed version"],
               [(name, importlib.metadata.version(name)) for name in expected])
    print("Python version:", platform.python_version())
    run_command([PYTHON, "-m", "pip", "check"], "pip_check.log")
    ''')
    md("""
    ## 3. Select the device — screenshot A (optional)
    Set DEVICE_REQUEST to auto, mps, cuda, or cpu. MPS is Apple's GPU backend.
    An explicitly requested unavailable GPU raises an error instead of silently
    claiming GPU execution. The selected device is passed to both inference and training.
    """)
    code(r'''
    import torch, psutil, yaml
    from job_matcher.modeling import select_device

    DEVICE_REQUEST = "auto"  # Set "mps" to require Apple GPU; "cpu" for a fallback.
    DEVICE = select_device(DEVICE_REQUEST)
    torch.set_num_threads(min(6, os.cpu_count() or 1))
    show_table(["Setting", "Value"], [
        ("Requested device", DEVICE_REQUEST), ("Selected device", DEVICE),
        ("Apple MPS available", torch.backends.mps.is_available()),
        ("CUDA available", torch.cuda.is_available()),
        ("System RAM (GiB)", round(psutil.virtual_memory().total / 1024**3, 1)),
        ("Model precision", "float16" if DEVICE == "cuda" else "float32"),
        ("Smoke batch size", 1), ("Smoke maximum sequence length", 1024),
    ])
    if DEVICE == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
    print("Device check passed:", DEVICE)
    ''')
    md("""
    ## 4. Download or verify the model and datasets — screenshot B
    Assets already on disk are reused; missing assets are downloaded. The output
    labels which happened. SHA-256 hashes are checked against the original saved
    manifest when available. Reuse is not presented as a fresh network download.
    """)
    code(r'''
    project = json.loads((ROOT / "configs/project.json").read_text())
    original_manifest = ROOT / "reports/evidence/all_manifest.json"
    expected_hashes = {}
    if original_manifest.exists():
        expected_hashes = {a["path"]: a["sha256"] for a in json.loads(original_manifest.read_text())["assets"]}
    existed_before = {path for path in expected_hashes if (ROOT / path).exists()}
    run_command([PYTHON, ROOT / "scripts/download_assets.py"], "download_and_verify.log")
    manifest = json.loads((ROOT / "outputs/downloads/all_manifest.json").read_text())
    assert not manifest["errors"], manifest["errors"]
    asset_rows = []
    for asset in manifest["assets"]:
        path = ROOT / asset["path"]
        assert path.stat().st_size == asset["bytes"]
        if asset["path"] in expected_hashes:
            assert asset["sha256"] == expected_hashes[asset["path"]], f"Hash mismatch: {path}"
        asset_rows.append((asset["path"], f'{asset["bytes"] / 1024**2:,.2f}',
                           "reused existing file" if asset["path"] in existed_before else "available after download",
                           "PASS" if asset["path"] in expected_hashes else "hash recorded"))
    show_table(["Asset", "MiB", "Availability", "SHA-256"], asset_rows)
    (RUN_DIR / "download_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("SUCCESS: model and both source datasets are available; download manifest has no errors.")
    ''')
    md("""
    ## 5. Load, inspect, and prepare datasets — screenshot C
    The generated pilot is separate from final reviewed data. This step regenerates
    machine-produced review queues; keep human annotations in separate files.
    """)
    code(r'''
    run_command([PYTHON, ROOT / "scripts/prepare_data.py"], "prepare_data.log")
    data_summary = json.loads((ROOT / "outputs/data/dataset_summary.json").read_text())
    (RUN_DIR / "dataset_summary.json").write_text(json.dumps(data_summary, indent=2) + "\n")
    show_table(["Dataset / subset", "Actual count"], [
        ("Resume-Job Fit source pairs", data_summary["source_rows"]),
        ("Distinct source job descriptions", data_summary["unique_job_descriptions"]),
        ("Explicit IT pilot pairs", data_summary["pilot_rows"]),
        ("LinkedIn source postings", data_summary["linkedin"]["rows"]),
        ("Unreviewed entry-level tech candidates", data_summary["linkedin"]["unreviewed_entry_tech_candidates"]),
    ])
    print("LIMITATION: IT pilot pairs all use a senior infrastructure-manager job, not an entry-level benchmark.")
    ''')
    md("""
    ## 6. Show loaded records and separate splits — screenshot C continued
    The preview shows schema and source record metadata without printing full source
    resumes. Every normalized exact resume belongs to only one pilot split. Jobs are
    not disjoint: the same senior IT job is used in all pilot splits.
    """)
    code(r'''
    source_path = ROOT / project["resume_dataset"]["local_dir"] / "job_resume_fit.csv"
    with source_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        print("Loaded CSV columns:", reader.fieldnames)
        preview = []
        for row in reader:
            if row["category"] == "INFORMATION-TECHNOLOGY":
                preview.append((row["ID"], row["category"], row["job_text"].splitlines()[0],
                                len(row["resume_text"]), row["ai_match_score"]))
                if len(preview) == 3:
                    break
    show_table(["Source ID", "Category", "Job title", "Resume characters", "Source AI score"], preview)
    splits, split_rows = {}, []
    for name in ("train", "validation", "test"):
        path = ROOT / "data/processed/pilot" / f"{name}.jsonl"
        splits[name] = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        assert len(splits[name]) == data_summary["splits"][name]
        split_rows.append((name, len(splits[name]), len({row["resume_id"] for row in splits[name]}),
                           str(path.relative_to(ROOT))))
    groups = {name: {row["resume_id"] for row in rows} for name, rows in splits.items()}
    assert not (groups["train"] & groups["validation"] or groups["train"] & groups["test"]
                or groups["validation"] & groups["test"])
    show_table(["Split", "Pairs", "Unique resumes", "Saved file"], split_rows)
    print("PASS: train, validation, and test files load; no normalized exact resume overlaps.")
    print("Seed:", project["seed"], "| exact normalized pair deduplication; near-duplicate review remains.")
    ''')
    md("""
    ## 7. Show the LoRA training configuration
    This is a short execution check, not a complete fine-tuning experiment. Frozen
    model weights remain shared with the base model; rank-8 adapters train on q/v
    projections. Prompt and padding labels are masked; assistant targets are supervised.
    """)
    code(r'''
    smoke = yaml.safe_load((ROOT / "configs/smoke.yaml").read_text())
    smoke["output_dir"] = "models/adapters/notebook_smoke"
    smoke["evidence_dir"] = "outputs/notebook_run/training"
    smoke_path = RUN_DIR / "smoke.yaml"
    smoke_path.write_text(yaml.safe_dump(smoke, sort_keys=False))
    print(yaml.safe_dump(smoke, sort_keys=False))
    print("Model:", project["model"]["repo_id"])
    print("Pinned model revision:", project["model"]["revision"])
    print("Training device:", DEVICE)
    print("Training implementation: scripts/train_lora.py (included in the submission package)")
    ''')
    md("""
    ## 8. Run the base model — screenshot D
    The same explicitly synthetic resume/job pair is used before and after adaptation.
    Raw model text is preserved. Invalid JSON/schema is recorded as a model-output
    limitation; it does not mean that the inference process failed to execute.
    """)
    code(r'''
    example = json.loads((ROOT / "examples/pair.json").read_text())
    print("DEMONSTRATION INPUT (synthetic, not an evaluation example):")
    print(json.dumps(example, indent=2))
    base_path = RUN_DIR / "base_prediction.json"
    run_command([PYTHON, ROOT / "scripts/infer.py", "--device", DEVICE, "--output", base_path], "base_inference.log")
    base = json.loads(base_path.read_text())
    assert base["device"] == DEVICE and base["adapter"] is None and base["generated_tokens"] > 0
    print("SUCCESS: frozen base model generated a response.")
    print("Strict output schema valid:", base["valid_schema"])
    ''')
    md("""
    ## 9. Execute two LoRA optimizer steps and save the adapter — screenshot E
    Training runs in a new process to avoid retaining the previous model in unified
    memory. The compact table after the logs is convenient for the written report.
    """)
    code(r'''
    run_command([PYTHON, ROOT / "scripts/train_lora.py", "--config", smoke_path, "--device", DEVICE], "lora_training.log")
    training_path = RUN_DIR / "training/summary.json"
    training_summary = json.loads(training_path.read_text())
    steps = [row for row in training_summary["log_history"] if "loss" in row]
    assert len(steps) == smoke["max_steps"] and training_summary["device"] == DEVICE
    adapter_dir = ROOT / smoke["output_dir"]
    assert (adapter_dir / "adapter_model.safetensors").stat().st_size > 0
    assert (adapter_dir / "adapter_config.json").is_file()
    show_table(["Completed step", "Loss", "Gradient norm", "Learning rate"],
               [(row["step"], row["loss"], round(row["grad_norm"], 5), row["learning_rate"]) for row in steps])
    show_table(["Training evidence", "Value"], [
        ("Device", training_summary["device"]),
        ("Prepared smoke subset size", training_summary["samples"]),
        ("Completed optimizer steps", len(steps)),
        ("Trainable parameters", f'{training_summary["trainable_parameters"]:,}'),
        ("Total parameters including adapter", f'{training_summary["total_parameters"]:,}'),
        ("Training runtime (seconds)", training_summary["metrics"]["train_runtime"]),
        ("Truncated examples in prepared subset", training_summary["truncated_samples"]),
        ("Saved adapter bytes", (adapter_dir / "adapter_model.safetensors").stat().st_size),
    ])
    print("SUCCESS: LoRA training completed and adapter files were saved.")
    print("These losses are execution evidence, not proof of improved matching quality.")
    ''')
    md("""
    ## 10. Reload the saved adapter in a fresh process — screenshot F
    A new base model is loaded from disk, then the new adapter is loaded and used on
    the same synthetic pair. This verifies that the saved adapter can be used later.
    """)
    code(r'''
    adapted_path = RUN_DIR / "adapter_prediction.json"
    run_command([PYTHON, ROOT / "scripts/infer.py", "--device", DEVICE, "--adapter", adapter_dir,
                 "--output", adapted_path], "adapter_reload_inference.log")
    adapted = json.loads(adapted_path.read_text())
    assert Path(adapted["adapter"]).resolve() == adapter_dir.resolve()
    assert adapted["device"] == DEVICE and adapted["generated_tokens"] > 0
    assert adapted["pair_file"] == base["pair_file"]
    assert adapted["model"]["revision"] == base["model"]["revision"]
    training_summary.update({
        "adapter_reload_verified": True,
        "adapter_reload_evidence": str(adapted_path.relative_to(ROOT)),
        "adapter_reload_verified_at": datetime.now(timezone.utc).isoformat(),
    })
    training_path.write_text(json.dumps(training_summary, indent=2) + "\n")
    show_table(["Run", "Device", "Generated tokens", "Strict schema valid"], [
        ("Base", base["device"], base["generated_tokens"], base["valid_schema"]),
        ("Reloaded LoRA adapter", adapted["device"], adapted["generated_tokens"], adapted["valid_schema"]),
    ])
    print("SUCCESS: saved adapter reloaded in a fresh process and generated a response.")
    print("No accuracy or speed comparison is inferred from these two example runs.")
    ''')
    md("""
    ## 11. Check split separation and training labels
    These tests check that normalized exact resumes do not cross pilot splits,
    truncation retains both input fields, and prompt/padding tokens are excluded
    from training loss while the assistant end token remains supervised.
    """)
    code(r'''
    test_output = run_command([PYTHON, "-m", "unittest", "discover", "-s", ROOT / "tests", "-v"], "tests.log")
    match = re.search(r"Ran (\d+) tests?", test_output)
    assert match and int(match.group(1)) >= 3 and "\nOK" in test_output
    tests_passed = int(match.group(1))
    print(f"SUCCESS: {tests_passed} pipeline tests passed.")
    ''')
    md("""
    ## 12. Save the evidence and summarize the milestone
    The evidence archive contains source code, configuration, logs, package versions,
    and the saved adapter. It excludes the multi-gigabyte base model and raw datasets.
    The executed notebook and HTML are saved by the runner after this cell finishes.

    **Scope:** the IT pilot's single senior job and source AI labels are unsuitable
    for claims about entry-level recommendation accuracy. The split checks cover
    normalized exact duplicates, not all near duplicates. Any invalid model JSON
    remains visible. Final-quality labels, ranking, and performance evaluation remain
    future work. Training is task-specific matching; employment is the application domain.
    """)
    code(r'''
    packages = subprocess.run([PYTHON, "-m", "pip", "freeze"], check=True, capture_output=True, text=True).stdout
    (RUN_DIR / "packages.txt").write_text(packages)
    verification = {
        "run_started_at": RUN_STARTED, "completed_at": datetime.now(timezone.utc).isoformat(),
        "platform": platform.system(), "machine": platform.machine(), "device": DEVICE,
        "source_resume_pairs": data_summary["source_rows"],
        "source_job_postings": data_summary["linkedin"]["rows"], "pilot_splits": data_summary["splits"],
        "base_inference_executed": True, "lora_optimizer_steps": len(steps),
        "adapter_saved": True, "adapter_reload_verified": training_summary["adapter_reload_verified"],
        "tests_passed": tests_passed, "base_valid_schema": base["valid_schema"],
        "adapter_valid_schema": adapted["valid_schema"],
        "performance_evaluation_run": False, "data_scope": "senior-job proxy pilot, not an entry-level benchmark",
    }
    (RUN_DIR / "milestone_verification.json").write_text(json.dumps(verification, indent=2) + "\n")
    evidence_zip = RUN_DIR / "midterm_evidence.zip"
    with zipfile.ZipFile(evidence_zip, "w", zipfile.ZIP_DEFLATED) as archive:
        for folder in [RUN_DIR, adapter_dir, ROOT / "configs", ROOT / "src/job_matcher", ROOT / "scripts"]:
            for path in folder.rglob("*"):
                if path.is_file() and path != evidence_zip and "__pycache__" not in path.parts:
                    archive.write(path, path.relative_to(ROOT))
    show_table(["Midterm execution requirement", "Observed status"], [
        ("Training/testing datasets downloaded and loadable", "PASS"),
        ("Separate pilot train/validation/test files", str(data_summary["splits"])),
        ("LLM inference runs", "PASS"), ("LoRA optimizer steps", len(steps)),
        ("Adapter save and fresh-process reload", "PASS"), ("Pipeline tests", f"{tests_passed} passed"),
        ("Strict response schema: base / adapter", f'{base["valid_schema"]} / {adapted["valid_schema"]}'),
        ("Matching accuracy evaluated", "No — not required for this midterm"),
    ])
    print("Evidence archive:", evidence_zip)
    print("Run verification:", RUN_DIR / "milestone_verification.json")
    print("SUCCESS: all pipeline stages above completed. Preserve the notebook with its actual outputs.")
    ''')
    md("""
    ## Brief report and screenshot guide
    Use sections **4–6** for downloaded/loaded datasets and splits, **8** for base
    inference, **9** for LoRA settings/losses and saved adapter, and **10** for reload
    and output. Section **3** shows MPS selection; section **12** summarizes execution.

    State that this is task-specific resume-job matching. Explain that the smoke run
    verifies implementation, not model quality. Disclose the senior-job pilot and
    strict-schema status shown above. Keyword overlap is a planned supporting metric:
    Python/SQL/Excel matched out of required Python/SQL/Excel/Tableau gives 3/4 = 75%.
    This percentage differs from the model's overall qualification score.
    """)
    notebook = nbf.v4.new_notebook(cells=cells)
    notebook.metadata.update({
        "kernelspec": {"name": "python3", "display_name": "Python 3 (project .venv)", "language": "python"},
        "language_info": {"name": "python"},
        "colab": {"name": "NLP Job Matcher — Mac MPS and CUDA", "provenance": []},
    })
    path = ROOT / "notebooks/01_colab_smoke.ipynb"
    nbf.write(notebook, path)
    print(f"Built {path}: {sum(c.cell_type == 'code' for c in cells)} code cells; no outputs fabricated.")


if __name__ == "__main__":
    main()
