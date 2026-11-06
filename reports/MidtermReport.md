# Midterm report draft: résumé–job matching with LoRA

**Author:** [add name]  
**Course:** CISC 488/688 [select your enrolled section]  
**Submission date:** [add date]  
**Setup date:** November 6, 2026

## Objective

This project investigates whether fine-tuning a small instruction language model improves résumé–job matching for recent graduates seeking entry-level technology jobs and internships. Given a résumé and candidate postings, the intended system will rank jobs, return qualification-alignment scores from 0–100, and explain matching skills and missing or uncertain requirements. The main comparison will use the same Qwen2.5-1.5B-Instruct checkpoint before and after LoRA adaptation. An API-model comparison is planned if time permits within an approximately $20 budget. Scores will not be interpreted as hiring probabilities.

In response to the instructor's clarification question, the fine-tuning is for the specific task of resume-job matching and recommendation. Recent-graduate technology employment is the application domain. I am not separately proposing continued pretraining on general employment text. LoRA is the training method used to adapt the model to structured matching examples.

The instructor's project-specific midterm feedback asks for downloaded training/testing datasets and evidence that the LLM and LoRA training implementations can run; it explicitly does not require performance results. The work reported below is an implementation milestone, with data-scope and output-format limitations disclosed.

## Downloaded data and inspection

The [Resume–Job Fit dataset](https://huggingface.co/datasets/mfaizan29/job_resume_fit) was downloaded at revision `4090a3f6c4aff1fae32429d7afa2feaafb399f5c`. It contains 2,385 résumé–job rows spanning 23 categories and 23 distinct job descriptions. Fields include résumé text, job text, required skills, extracted/matched skills, and AI-generated fit scores. These scores are weak supervision rather than human ground truth.

Inspection revealed a limitation for the proposed task: the 120 rows in the explicit INFORMATION-TECHNOLOGY category all use one IT Infrastructure Manager posting that requires 8–10 years of experience. The source therefore cannot simply be filtered to that category and treated as an entry-level training dataset. These examples were retained only as a labeled pipeline pilot. Recent-graduate scope and suitable entry-level pair labels remain to be verified.

The [LinkedIn Job Postings dataset, version 13](https://www.kaggle.com/datasets/arshkon/linkedin-job-postings/versions/13), was downloaded and extracted with its side tables. The postings file contains 123,849 rows. Inspection identified seven missing-title/description rows and 13,231 duplicate normalized title/description rows. A first-pass technology-title and junior/experience-level filter produced 1,299 candidate postings. These candidates still require manual checks of experience, education, enrollment, location, and work-authorization requirements. They are not labeled résumé–job evaluation pairs.

## Preprocessing and provisional splits

Text was normalized with Unicode NFKC, case folding, and whitespace normalization for hashing and exact duplicate detection. The IT pilot contains 120 unique résumé texts and no duplicate normalized résumé–job pairs. Résumés were grouped by normalized-text hash and split with seed 42 into 96 training, 12 validation, and 12 testing rows. No normalized exact résumé appears in more than one split. Near-duplicate review is unfinished. The same senior job description appears in all three splits; these pilot splits do not measure unseen-job generalization.

Raw sources were preserved. Revision IDs, downloaded byte counts, and SHA-256 hashes are saved in `outputs/downloads/all_manifest.json`; inspection results are in `outputs/data/dataset_summary.json`. A manual-review queue is saved separately from processed pilot files.

## Model and implementation

The downloaded model is [Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct), revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`. The base comparison and adapted comparison use these same weights. The code loads the tokenizer and weights locally and formats résumé/job inputs using the model's chat template. It requests JSON containing a score, matching skills, missing skills, an eligibility assessment, and an explanation. A parser records schema validity and retains the raw response. The demonstration input is a clearly labeled synthetic graduate profile and entry-level data-analyst posting.

The LoRA pipeline uses Hugging Face Transformers and PEFT. The smoke configuration selects four pilot examples, two optimizer steps, sequence length 1,024, batch size 1, learning rate 2e-4, rank 8, alpha 16, dropout 0.05, and adapters on `q_proj` and `v_proj`. Gradient checkpointing is enabled. Prompt and padding tokens are excluded from the loss; full assistant targets and the end-of-turn token are retained. Long input fields are truncated to preserve target space, with truncation counts saved.

The pilot targets retain source AI scores and skill lists. Missing skills are derived from required skills absent from the source's matched list. Eligibility is marked unclear, and the explanation target discloses the need for manual review. These targets demonstrate supervised training mechanics; they are not validated recent-graduate recommendation labels or quality explanation supervision.

## Execution evidence

Local verification completed on November 6, 2026 using this Mac's Apple MPS device and Python 3.11. The installed versions include PyTorch 2.8.0, Transformers 4.57.1, PEFT 0.17.1, and Accelerate 1.10.1. Dependency checks passed. The Colab walkthrough is provided but has not been executed.

| Check | Actual result | Saved evidence |
| --- | --- | --- |
| Base-model inference | Completed; 219 input tokens, 110 generated tokens; 11.21 s load and 17.67 s generation | `outputs/predictions/base_example.json` |
| LoRA training | Two optimizer steps on a four-example training subset; training runtime 12.93 s, total script time 28.70 s | `outputs/training/smoke/summary.json` |
| Trainable parameters | 1,089,536 out of 1,544,803,840, approximately 0.0705% | Same summary and `outputs/logs/smoke_training.log` |
| Adapter save | `adapter_model.safetensors` saved: approximately 4.2 MiB, plus configuration/tokenizer | `models/adapters/smoke/` |
| Fresh-process reload/inference | Completed; 106 generated tokens; 9.90 s load and 12.07 s generation | `outputs/predictions/smoke_adapter_example.json` |
| Split/token-label checks | All three tests passed | `reports/evidence/verification.json` |

Actual training log excerpt:

```text
step 1: loss=1.8364, grad_norm=0.8256629, learning_rate=0.0002
step 2: loss=1.9196, grad_norm=0.9725599, learning_rate=0.0001
train_loss=1.8779815, train_runtime=12.9307 seconds, epoch=0.5
```

All four examples in the prepared training subset required input truncation at the 1,024-token smoke limit. Only two optimizer steps were run. These losses establish that forward/backward passes and optimization executed; they do not establish convergence or improvement.

The base model's raw response assigned score 85 and listed Python, SQL, and Git as matches and Tableau as missing. Its explanation was: “The resume matches most of the required skills for the entry-level Data Analyst position, including Python, SQL, and Git.” The adapter run also assigned 85 and listed the same skills; its corresponding explanation began: “The resume matches most of the required skills for the position, including Python, SQL, and Git.” Both full responses are preserved in the prediction files.

Both responses wrapped the requested JSON in Markdown fences and used `Eligible` instead of the required lowercase enum. Consequently, both have `valid_schema: false`; `parsed_output` is null, and no silently repaired score is used by the pipeline. They also did not fully address unspecified location/authorization requirements. Tableau was preferred rather than required, so future outputs should distinguish those gaps. Successful generation/reload therefore verifies execution while exposing formatting and qualification-reasoning work still needed. The different timings are single local observations and are not a speed comparison.

Small, report-ready copies of manifests, dataset summaries, training logs, and synthetic-example prediction files are retained in `reports/evidence/`. A completed smoke run establishes execution, not improved matching quality.

## Remaining work and final evaluation

The principal subsequent data task is obtaining and labeling genuinely in-scope recent-graduate résumé–job pairs. I plan to prioritize cleaning and adapting existing data, manually review candidate jobs, define a fixed scoring rubric, and freeze partitions with both résumés and job texts held out. If a new dataset is necessary, I will follow the proposal handout's feasibility-memo and completed-dataset demonstration requirements. Full training, validation-based model selection, and batch ranking remain to be completed. The current starter trainer saves a final adapter without validation-based checkpoint selection. These are next-stage project tasks rather than additional performance requirements for this midterm.

The final comparison will use identical held-out résumés, candidate pools, prompts, generation settings, and input budgets. Planned measures include top-ranked recommendation quality, manually annotated required-skill overlap in selected jobs, score agreement with human judgments, experience/education/eligibility compliance, explanation grounding, malformed-output rates, latency, and cost. I will report sample-size limits and errors rather than assume that fine-tuning improves results.

To clarify the keyword-based measure requested by the instructor, I will manually identify required skills, normalize capitalization and common aliases, and compute the percentage supported by the resume. For example, a posting requiring Python, SQL, Excel, and Tableau has four required skills. A resume listing Python, SQL, and Excel supports three of them, producing `3 / 4 * 100 = 75%` skill coverage. Preferred skills are kept separate. This measure gives skills equal weight and does not assess proficiency, experience, or eligibility; it is distinct from the model's overall qualification score. Because it is fixed for each pair, I will compare it across the jobs each model recommends, alongside separate human checks. This metric has not yet been evaluated.

According to the instructor's supplied midterm feedback, performance results and proof of improvement are not required at this stage. The implementation runs have been demonstrated; final data suitability and recommendation quality remain unverified. Any unfinished scope filtering and unverified execution environment, including Colab, will remain explicitly identified in the submitted report.
