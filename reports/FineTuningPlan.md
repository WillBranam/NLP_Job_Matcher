# Fine-tuning and evaluation plan

Prepared November 6, 2026, from `AGENT.md`, `FinalReport.md`, downloaded source data, and the setup evidence in `outputs/`.

Updated against the supplied instructor feedback and course PDFs: the midterm requires downloaded training/testing datasets and runnable LLM/LoRA implementations, with no performance results. The execution evidence already exists. Phases 2-4 below are proposed subsequent research work, not additional instructor-prescribed midterm requirements. See `InstructorRequirements.md` for source distinctions and dates.

## Research question and deliverable

Does task-specific LoRA tuning improve recent-graduate résumé–job matching compared with the **same untuned instruction checkpoint**? The system should score each résumé–job pair, explain supporting and missing qualifications, and sort the same candidate job pool for each résumé. Scores mean qualification alignment, not hiring probability. A general-purpose API model is an optional final comparison within the proposed $20 budget.

The training objective is the specific matching/recommendation task; tech employment is its application domain. Continued pretraining for general domain knowledge is not a second planned training objective. This addresses the instructor's question about domain adaptation versus task fine-tuning.

Only Qwen2.5-1.5B-Instruct needs to be downloaded. Its untuned weights serve as the base comparison; training creates a small adapter applied to those same weights. Do not compare a different pretrained model with the adapter version and attribute the difference solely to tuning.

## What the actual data permits

| Source | Observed contents | Intended use |
| --- | --- | --- |
| Resume–Job Fit | 2,385 rows, 23 categories and 23 distinct job descriptions; AI-generated scores and skills | Pipeline checks and cautiously scoped weak supervision |
| Explicit IT subset | 120 unique résumés paired with one senior infrastructure-manager posting | Pilot only; 96 train / 12 validation / 12 test |
| LinkedIn version 13 | 123,849 rows; 7 missing-text rows and 13,231 duplicate normalized title/description rows identified | Source of realistic entry-level candidate postings |
| Initial LinkedIn candidates | 1,299 after title/experience filters and deduplication | Manual-review queue, not automatically eligible jobs or labeled pairs |

The senior IT posting requires 8–10 years of experience. Strict recent-graduate job filtering would therefore leave no usable in-scope job in that category. There may be useful technical résumés in other categories, but broadening to all engineering/design rows without inspection would introduce unrelated roles. Manual scope review is the first substantive data task.

The pilot has résumé-disjoint splits but the same job appears in every split. It cannot establish generalization to unseen jobs. Skill-list-derived targets are also not independent proof of explanation quality. The current explanation target explicitly discloses proxy labels and requires manual eligibility verification.

## Phase 1 — Complete the midterm execution evidence

1. Keep downloaded source files unchanged and retain revision IDs, Kaggle version, hashes, and dataset cards. Record the training data's `license: other` metadata and clarify its usage terms before redistributing it.
2. Save inspection counts and the provisional split method: NFKC/case/whitespace normalization, exact pair deduplication, grouping by normalized résumé hash, seed 42, approximately 80/10/10 splits.
3. Run the frozen checkpoint on `examples/pair.json`. Save raw text, JSON validity, timing, device, and any truncation. This is a labeled synthetic demonstration, not a performance test.
4. Run `configs/smoke.yaml`: four source examples, two optimizer steps, sequence length 1,024, batch size 1, rank 8, alpha 16, dropout 0.05, `q_proj` and `v_proj` adapters, AdamW, learning rate 2e-4, seed 42.
5. Save the adapter and load it into a fresh base-model process. Run the same example again and retain its output. A loss value, changed response, or successful reload does not establish improvement.
6. Report execution honestly. Performance results are not required for this midterm, according to the supplied instructor feedback. Review/personalize the report and check current submission instructions; the listed October 30 progress-report date has passed as of setup.

## Phase 2 — Build the correct training and evaluation data

Suggested manageable target, subject to annotation time: **30 distinct recent-graduate résumés and 30 distinct reviewed jobs** across software development, data analysis, and IT support. This is a proposed collection target, not downloaded data.

Prefer cleaning/adapting existing datasets first. The proposal handout asks for an emailed feasibility memo and a demonstration of the completed dataset if creating a new dataset. Any original/synthetic dataset creation proposed below is therefore a planning option subject to that course process; no email or dataset-creation approval is assumed. The target count and proposed rubric are suggestions, not course requirements.

- Review the 1,299 job candidates. Keep internships and positions allowing roughly 0–2 years; read full descriptions for contradictory experience, degree, enrollment, authorization, and location requirements. Entry-level metadata alone is insufficient. Record why each job is accepted, rejected, or uncertain.
- Use consenting/de-identified résumés or clearly marked synthetic profiles. Separate synthetic and real data in reporting. Review remaining source categories for usable junior profiles; do not assume any are in scope.
- Create 20 training résumés, 5 validation résumés, and 5 test résumés. Split jobs into 10 training, 10 validation, and 10 test postings, with role-family coverage in each partition. Pair only within the corresponding partition: approximately 200/50/50 pairs. This keeps both résumés and job texts separate across partitions and gives each held-out résumé 10 postings to rank.
- Include strong, partial, and poor matches in each role family. Do not assign a high score merely because résumé and job share a category. Build negative examples with explicit reasons and reviewed labels, rather than retaining an original positive score after swapping jobs.
- Remove exact duplicates, then inspect near duplicates, shared résumé templates with the same underlying person, and reposted jobs before freezing the split. Keep transformations, IDs, seeds, counts, and exclusions in a manifest. Never tune prompts against final test examples.
- Review each label using a fixed rubric. AI-produced labels may assist training but must remain identified as weak labels; independently assess the held-out validation/test labels. A single-reviewer course study should disclose that limitation. If possible, obtain an independent second review on a small sample and report agreement.

### Proposed manual rubric

| Component | Points | Evidence to assess |
| --- | ---: | --- |
| Required technical skills | 40 | Explicit skills/projects, with aliases; missing required skills reduce points |
| Relevant experience/projects | 25 | Required level, internships, projects, and transferable evidence |
| Education | 15 | Required degree/discipline or accepted alternatives |
| Eligibility constraints | 10 | Explicit enrollment, authorization, location, or other stated requirements |
| Role/responsibility alignment | 10 | Evidence for the actual work rather than title similarity |

Use anchored examples to keep scoring consistent. Mark eligibility `eligible`, `ineligible`, or `unclear`, and distinguish absent evidence from a confirmed failure. Freeze a ranking policy before evaluation; a reasonable proposal is to rank confirmed ineligible postings below eligible/unclear postings, then sort by score within each group. Report unknown constraints rather than guessing. Protected characteristics must not enter scoring.

Annotate required skills separately from preferred skills, with a small reviewed alias map (e.g., JS → JavaScript, PostgreSQL → Postgres). Do not equate Java with JavaScript. Manually verify explanations against the original text, not the source dataset's AI skill lists.

## Phase 3 — Fine-tune on reviewed examples

Use `configs/finetune.yaml` as a starting point: two epochs, learning rate 1e-4, sequence length 2,048, batch size 1, gradient accumulation 8, rank 8, alpha 16, dropout 0.05, `q_proj` and `v_proj`. These settings are proposed, not optimized. Plain LoRA on a Colab CUDA GPU is the first choice; the local Mac can run smoke checks. Measure GPU memory and time before committing to a complete run.

The code uses the model's chat template, trains only on assistant completion tokens, masks prompt/padding labels with -100, and retains the complete target plus end-of-turn token. Long résumé/job inputs are truncated with both fields represented and a truncation count saved. For real experiments, inspect token distributions, choose a context length that preserves requirements, and compare base/adapter using the same input limits. Do not claim complete eligibility checking if the relevant section was truncated.

Before full experimentation, add validation loss/evaluation and checkpoint selection to the current starter trainer. Compare a small prespecified grid (e.g., learning rates 1e-4 and 2e-4 with one or two epochs), select on validation evidence, then evaluate the frozen winner once on test. The current trainer saves only the final adapter and does not implement validation-based selection.

Save package versions, model revision, data manifest, prompt, configuration, trainable parameter count, loss history, adapter, and compute time. Add 4-bit QLoRA only if memory requires it and validate it on CUDA separately; the current code does not depend on bitsandbytes.

## Phase 4 — Implement ranking and run final comparisons

The current inference script scores a single pair. Add batch scoring and ranking over a fixed candidate pool, preserving job IDs and raw/parsed outputs. Apply exactly the same prompt, input budget, generation settings, eligibility policy, and candidate pools to base and tuned models. Record malformed responses without silently repairing them differently across models.

| Measure | Method |
| --- | --- |
| Recommendation quality | Precision@5 for manually acceptable recommendations; NDCG@5 using the frozen relevance rubric |
| Required-skill overlap | Manually annotated required skills present in résumé / number of required skills, using aliases; report undefined if denominator is zero |
| Score agreement | MAE against human scores and rank correlation, with uncertainty appropriate to the small sample |
| Eligibility | Count confirmed-ineligible recommendations and cases handled as uncertain |
| Explanation quality | Manual checks of correct supporting evidence, correctly identified gaps, unsupported claims, and clarity |
| Reliability | JSON/schema success rate and truncation rate |
| Cost | Warm inference latency, separate load time, hardware/memory, training time, API tokens and dollars |

Keyword overlap measures a property of the résumé–job pair. It is identical across models on the same pair; compare average overlap in each model's **selected top jobs**, not as though the model changes the résumé's skills. Report zero-required-skill cases separately. Keep a transparent keyword baseline if time permits.

For the instructor's requested example: if a posting requires Python, SQL, Excel, and Tableau and the resume supports Python, SQL, and Excel, the required-skill overlap is `3/4 * 100 = 75%`. Normalize capitalization/common aliases, keep preferred skills separate, and do not interpret this percentage as the model's overall fit score. This evaluation workflow is planned rather than already implemented.

Compare paired results by held-out résumé, not by treating every pair as statistically independent. With only five test résumés, report per-résumé outcomes and wide uncertainty rather than strong significance claims. Add an API model only after the local comparisons work: freeze its exact identifier/date, use the same candidate pool, log tokens/costs, and stop before exceeding $20. No API model has been selected or called in this setup.

## Suggested order after the midterm

1. Review the job pool and obtain suitable recent-graduate profiles.
2. Label, deduplicate, and freeze both résumé and job partitions.
3. Improve human-written training explanations and add validation/checkpoint selection.
4. Run the base baseline and one complete LoRA experiment on a GPU.
5. Implement batch ranking; evaluate matching, eligibility, skills, explanations, and latency.
6. Run the optional API comparison if budget/time remains.
7. Write errors and limitations, including weak supervision, small samples, synthetic data if used, historical postings, score calibration, and bias risks.

## Sources

- Project requirements: repository `AGENT.md` and `FinalReport.md`.
- Instructor feedback supplied in chat and the proposal/overview/possible-projects PDFs; detailed provenance in `InstructorRequirements.md`.
- [Qwen model card](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct); pinned revision in `configs/project.json`.
- [Resume–Job Fit dataset card](https://huggingface.co/datasets/mfaizan29/job_resume_fit); pinned revision in `configs/project.json`.
- [LinkedIn Job Postings version 13](https://www.kaggle.com/datasets/arshkon/linkedin-job-postings/versions/13); source license CC BY-SA 4.0.
- [Hugging Face LoRA documentation](https://huggingface.co/docs/peft/en/package_reference/lora) and [Trainer documentation](https://huggingface.co/docs/transformers/en/main_classes/trainer). Runtime versions are pinned locally; these documentation URLs can evolve.
