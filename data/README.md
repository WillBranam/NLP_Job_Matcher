# Data locations and provenance

- `raw/job_resume_fit/`: unchanged CSV and source card from [mfaizan29/job_resume_fit](https://huggingface.co/datasets/mfaizan29/job_resume_fit), pinned in `configs/project.json`.
- `raw/linkedin_job_postings/`: version-13 archive and extracted tables from [arshkon/linkedin-job-postings](https://www.kaggle.com/datasets/arshkon/linkedin-job-postings/versions/13). Source license: CC BY-SA 4.0. Retain attribution for derivatives; the complete raw archive is not committed.
- `interim/resume_scope_review.jsonl`: IT-source résumé review queue; its job is marked out of scope because it requires 8–10 years of experience.
- `interim/linkedin_entry_tech_candidates.jsonl`: heuristic job candidate pool; all rows await eligibility review. An empty `required_skills` list is an annotation placeholder, not evidence that the job has no required skills.
- `processed/pilot/`: source-AI-label train/validation/test splits for execution checks only.
- `processed/reviewed/`: intended location for the eventual manually reviewed recent-graduate dataset. No reviewed pairs have been created by setup.

Run `python scripts/prepare_data.py` to recreate pilot splits, review queues, and `outputs/data/dataset_summary.json`. This command overwrites generated queues and splits: keep completed human annotations in a separate versioned file before rerunning it. Raw downloads and generated data are excluded from Git.

Prepared rows contain `resume`, `job`, `target` (score, matching_skills, missing_skills, eligibility, explanation), résumé/job hashes, source IDs, label provenance, and scope. Before full training, use `scope: "reviewed_recent_graduate"` only after checking every pair. Keep human annotation decisions and duplicate groups with the split manifest.
