# Midterm completion checklist

Use this alongside `FinalReport.md` and `InstructorRequirements.md`. The instructor's supplied feedback explicitly asks for an implemented LLM, downloaded training/testing datasets, and evidence that LoRA training runs. Performance results are not required. Final-quality labeling and system improvements below are future work, not extra midterm requirements.

## Data and setup

- [x] Read project goals and midterm requirements.
- [x] Download Qwen2.5-1.5B-Instruct weights, tokenizer, model card, and license.
- [x] Download Resume–Job Fit: 2,385 pairs.
- [x] Download and extract LinkedIn Job Postings version 13: 123,849 postings, plus side tables.
- [x] Save source revisions, file sizes, SHA-256 hashes, and package versions.
- [x] Inspect fields/categories and identify the senior-job mismatch in the IT training subset.
- [x] Save provisional résumé-disjoint pilot splits: 96 train / 12 validation / 12 test, seed 42.
- [x] Save a review queue of 1,299 possible entry-level tech postings.
- [x] Identify and document that final recent-graduate scope review and near-duplicate review are unfinished.

Evidence: `outputs/downloads/all_manifest.json`, `outputs/data/dataset_summary.json`, `data/interim/`, and `data/processed/pilot/`.

## Execution evidence

- [x] Run the base model and retain its raw output, schema validity, device, and timing.
- [x] Implement LoRA setup, input formatting/tokenization, prompt/padding loss masking, and training configuration.
- [x] Run two training steps successfully and retain loss logs.
- [x] Save the adapter and tokenizer.
- [x] Reload the adapter in a fresh process and retain inference output.
- [x] Verify split separation and that the training loss excludes prompt/padding tokens.

Commands are in `README.md`. Both inference runs generated text but failed strict JSON/schema validation (Markdown fences and capitalized `Eligible`). Record this limitation. Report-ready copies of the setup evidence are in `reports/evidence/`; raw datasets, weights, and working outputs remain excluded from Git.

## Report assembly and your review tasks

- [ ] Review and personalize `reports/MidtermReport.md`; add your name, course, submission date, and required formatting/length from the instructor.
- [x] State the research question and what the score means in the draft.
- [x] Explicitly identify the fine-tuning as task-specific resume-job matching within the tech-employment domain.
- [x] Explain keyword overlap with the Python/SQL/Excel/Tableau example and distinguish it from the model's overall score.
- [x] Cite the model/data sources; report actual counts and split method.
- [x] Clearly distinguish the out-of-scope pilot and synthetic demonstration from final evaluation data.
- [x] Include base-model and reloaded-adapter output excerpts from the same input example.
- [x] Include training logs showing completed steps/loss, trainable parameter count, and saved adapter files.
- [x] Explain what is implemented and what is unfinished: reviewed in-scope data, structured-output reliability, batch ranking, validation/checkpoint selection, evaluation, and optional API comparison.
- [ ] Confirm that claims match execution evidence. Do not claim improved accuracy from the smoke run.
- [ ] Check the current Canvas progress-report assignment and any deadline updates. The supplied overview lists October 30, 2026 and says no late submissions; submission/extension status is unknown.
- [ ] Confirm whether the revised proposal requested for October 19 at 5:00 pm was submitted. Do not infer its status from the approved project or repository text.
- [ ] Submit the reviewed report and any required supporting files in the assigned format.

## Subsequent project work, not an added midterm gate

- [ ] Review eligibility and recent-graduate scope and create suitable final training/evaluation pairs.
- [ ] Review near duplicates; current checks cover normalized exact matches only.
- [ ] Improve structured output, distinguish required/preferred skills, and handle uncertain eligibility.
- [ ] Implement ranking, validation-based selection, and final comparisons.

Keep the senior-job mismatch and provisional split limitations visible in the midterm report. The supplied feedback does not explicitly require a newly annotated entry-level benchmark before this milestone. If creating a new dataset becomes the chosen solution, follow the proposal handout's feasibility-memo requirement; existing-data cleaning should remain the first approach.
