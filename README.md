# NLP Job Matcher

Research project: compare Qwen2.5-1.5B-Instruct before and after LoRA fine-tuning for matching recent graduates' résumés to entry-level tech jobs and internships. Rank jobs, assign qualification-alignment scores from 0–100, and explain matched skills, missing requirements, and uncertain eligibility. See the original proposal in [AGENT.md](AGENT.md) and course requirements in [FinalReport.md](FinalReport.md).

Read [the fine-tuning plan](reports/FineTuningPlan.md), [the midterm checklist](reports/MidtermChecklist.md), and [the report draft](reports/MidtermReport.md).

[Instructor requirements and clarifications](reports/InstructorRequirements.md) distinguish the confirmed midterm execution requirements from later research work. The supplied feedback requires downloaded training/testing data and runnable LLM/LoRA implementations, without performance results; it does not explicitly require a newly annotated entry-level benchmark at this milestone.

Local setup verification completed: base inference, two LoRA training steps, adapter save/reload and inference, and three pipeline tests. Both inference responses failed strict JSON validation; their raw outputs and the training evidence are preserved in [reports/evidence](reports/evidence). No matching-quality evaluation has been run.

The [executed Mac notebook](notebooks/01_midterm_mac_executed.ipynb) contains actual outputs from all 12 code cells, run on Apple MPS. Open the [HTML copy](notebooks/01_midterm_mac_executed.html) to read outputs and take screenshots without starting a kernel. The [brief report and screenshot guide](reports/MidtermScreenshotGuide.txt) maps screenshots to numbered sections. This additional notebook run has separate evidence in outputs/notebook_run and a separate adapter in models/adapters/notebook_smoke.

The original notebook filename, notebooks/01_colab_smoke.ipynb, now contains the updated Mac/CUDA/CPU template. Choose auto, mps, cuda, or cpu in section 3. Existing pinned dependencies and downloads are reused; the notebook captures subprocess output instead of relying on terminal-only logs. No code cell needs CUDA specifically. Colab/CUDA execution remains unverified.

To execute and export a fresh copy from the repository root:

~~~bash
python -m pip install --cache-dir .cache/pip -r requirements-notebook.txt
python scripts/execute_notebook.py
~~~

The runner stops on errors, saves real partial outputs on failure, and verifies that every code cell has an execution count and saved output. On success it exports the executed notebook, HTML, and notebooks/midterm_submission.zip with source, configuration, logs, adapter, and notebook outputs. The ZIP excludes the base-model weights and raw datasets.

## Setup

Use Python 3.11. The repository already has a local `.venv` on the setup machine.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --cache-dir .cache/pip -r requirements-lock.txt
```

Run commands from the repository root. Scripts locate source code directly, so an editable install is optional. Raw data, model weights, adapters, local environments, and generated outputs are excluded from Git. Downloads require internet access; inference and training use local model files.

```bash
python scripts/download_assets.py
python scripts/prepare_data.py
python scripts/infer.py
python scripts/train_lora.py --config configs/smoke.yaml
python scripts/infer.py --adapter models/adapters/smoke --output outputs/predictions/smoke_adapter_example.json
python -m unittest discover -s tests -v
```

The downloader pins Hugging Face revisions, records byte sizes and SHA-256 hashes in `outputs/downloads/all_manifest.json`, and downloads Kaggle dataset version 13. Existing files are reused; compare them with the saved manifest if integrity is in doubt. Choose `--only model`, `--only resumes`, or `--only jobs` to retrieve one asset group. The adapter is created by training; the base and LoRA experiments share the same downloaded checkpoint. The optional API comparison has no local model download.

`--device auto` selects CUDA, Apple MPS, then CPU. Full experiments should use a Colab GPU; the smoke configuration is only two optimizer steps. No bitsandbytes or 4-bit backend is required. If you later add QLoRA, validate it separately on CUDA.

## Structure

```text
configs/                 Pinned sources and smoke/full training settings
data/raw/                Original source datasets and source documentation
data/interim/            Manual-review queues and heuristic job candidates
data/processed/pilot/    Provisional 96/12/12 splits for pipeline checks
data/processed/reviewed/ Future verified recent-graduate train/validation/test pairs
examples/                Clearly marked synthetic demonstration input
models/base/             Original Qwen weights and tokenizer
models/adapters/         Saved LoRA adapters
notebooks/               Mac/CUDA/CPU notebook, executed outputs, HTML and submission ZIP
outputs/downloads/       Download manifests and hashes
outputs/data/            Dataset inspection results
outputs/training/        Configuration, loss logs, adapter provenance
outputs/predictions/     Raw and parsed model responses, timings
outputs/metrics/         Future evaluation results
reports/                 Fine-tuning plan, checklist, report draft, figures
scripts/                 Download, prepare, infer, train entry points
src/job_matcher/         Shared prompting and model code
tests/                   Split separation and token-label checks
```

## Data scope

The downloaded Resume–Job Fit dataset has 2,385 rows and 23 distinct job descriptions. Its 120 explicit IT rows all use a senior IT Infrastructure Manager job requiring 8–10 years of experience. The saved pilot splits demonstrate the pipeline and **do not constitute an entry-level benchmark**. Their AI scores and source-derived skill lists are proxy targets; the explanation target is a disclosure template, not a human-written explanation. Full training refuses rows unless `scope` is `reviewed_recent_graduate`.

The LinkedIn source has 123,849 rows. The first filter yields 1,299 possible entry-level tech postings, saved to `data/interim/linkedin_entry_tech_candidates.jsonl`; these are awaiting manual review. The full source and side tables are retained. See `outputs/data/dataset_summary.json` for actual counts and limitations.

## Full training, after manual review

Create `data/processed/reviewed/train.jsonl`, `validation.jsonl`, and `test.jsonl`. Each row must include `resume`, `job`, `target`, and `scope: "reviewed_recent_graduate"`. Use the pilot JSONL as a structural example, replace proxy targets with reviewed labels, and retain source IDs and normalized text hashes. Do not relabel pilot data as reviewed without verifying it.

```bash
python scripts/train_lora.py --config configs/finetune.yaml
```

The full configuration is a starting point. Validation-based selection, batch ranking, human annotation, near-duplicate review, API comparison, and final evaluation are planned work, described in the plan. No API requests or paid compute are configured.
