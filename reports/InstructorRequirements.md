# Instructor requirements and project clarifications

Updated November 6, 2026 from the instructor feedback supplied in chat and the three course PDFs supplied by the student. This file distinguishes course requirements from implementation choices and recommendations. It does not establish whether an assignment has been submitted or whether a deadline has been extended.

## Project-specific feedback: controlling midterm scope

Vijay Shanker's feedback, dated October 12 at 1:50 pm in the supplied text, approves the project based on the student's answers and asks for a revised proposal by Monday, October 19 at 5:00 pm. Submission of that revision has not been verified here.

The same feedback asks for:

1. Implementation of the LLM model.
2. Downloaded datasets for training and testing. The supplied feedback says "texting"; this is interpreted as testing in context.
3. Evidence that the training implementation using LoRA can run.

The instructor explicitly says results are not needed at the midterm. In context, the milestone is execution and data availability rather than a completed accuracy study or proof of improvement.

The repository has already demonstrated local model loading/inference, two LoRA optimizer steps, adapter save and fresh-process reload/inference. The datasets are downloaded, and provisional train/validation/test files are saved. Evidence is retained in `reports/evidence/`. This establishes execution; it does not establish final data suitability or acceptance by the instructor.

## Required versus recommended

| Item | Basis | Current status |
| --- | --- | --- |
| LLM implementation runs | Explicit instructor feedback | Demonstrated locally |
| Training and testing datasets downloaded | Explicit instructor feedback | Downloaded; pilot splits prepared |
| LoRA training implementation runs | Explicit instructor feedback | Two-step run completed |
| Progress report documents implementation/data availability | Project overview plus project-specific feedback | Draft prepared; student review/submission unverified |
| Save/reload adapter, dependency checks, token-label tests | Additional verification chosen for this repository | Completed |
| Final-quality entry-level labels and near-duplicate review | Research-quality work toward the final project; not explicitly required in the supplied midterm feedback | Unfinished; disclose pilot limitations |
| Strict JSON reliability, batch ranking, checkpoint selection | Planned system improvements; not explicitly required for this execution milestone | Unfinished |
| Accuracy comparisons, explanation evaluation, API experiment | Final-project experiments; no performance results required at midterm | Not run |

Do not treat suggested annotation counts, rubric weights, or a proposed data collection plan as instructor-prescribed midterm requirements.

## Answer to "domain adaptation or a specific task?"

The planned fine-tuning is supervised adaptation for a **specific task: resume-job matching and recommendation**. The application domain is recent-graduate and entry-level technology employment. Training inputs contain a resume and job description; outputs contain a qualification-alignment score and an evidence-grounded assessment. At inference, pair scores and eligibility information will support job ranking.

This is not a separate experiment in continued pretraining on general employment text. LoRA is the parameter-efficient training method, not the definition of the task. The central comparison holds the instruction checkpoint fixed and compares its untuned and task-tuned behavior. The optional API comparison is an additional experiment.

The course's `possible-projects (1).pdf`, page 5, describes Project 3B: a small LLM for job matching and recommendation with explanations and a LoRA/model comparison. It is a relevant example, not an instruction to implement every suggested experiment before the midterm.

## Answer to "keyword matching: explain with an example"

Keyword overlap is a supporting evaluation measure. Manually identify a posting's **required** skills, normalize capitalization and agreed aliases, and count the required skills also supported by the resume.

`keyword overlap = matched required skills / total required skills * 100`

Example:

- Job requires Python, SQL, Excel, and Tableau.
- Resume lists Python, SQL, and Excel, but provides no Tableau evidence.
- Three of four required skills match: `3 / 4 * 100 = 75%`.

This 75% is a skill-coverage measure, not the LLM's overall qualification score or a hiring probability. Preferred skills should be kept separate. All required skills receive equal weight in this simple measure, and it does not assess proficiency, experience, education, or eligibility. If no required skills are identified, report the measure as undefined rather than divide by zero.

For the same resume-job pair, keyword overlap is fixed across models. To compare recommendation systems, calculate it for the jobs each model selects and compare overlap across their top recommendations. Manually evaluate eligibility and explanation grounding separately. The metric is planned; the current code has not implemented this manual evaluation workflow.

## Dates and administrative requirements in the supplied PDFs

| Document or feedback | Stated milestone/date |
| --- | --- |
| Project overview | Initial proposal October 7, 2026 |
| Specific proposal handout | Initial proposal October 9, 2026, 11:59 pm, via Canvas |
| Project-specific approval feedback | Revised proposal Monday, October 19, 5:00 pm |
| Project overview | Progress report October 30, 2026; 20% of project grade |
| Project overview | Draft poster/presentation plus written report November 20, 2026; 10% |
| Project overview | Final poster/presentation and written report December 4, 2026; 40% |

The overview states no late submissions will be accepted. October 30 is before the setup date of November 6; verify the current Canvas assignment and any instructor updates rather than assume an extension or that the original date is still operative. The initial proposal dates differ across handouts; retain their provenance instead of silently treating them as one date.

The proposal handout strongly encourages existing datasets. If the student elects to create a new dataset, it asks for an emailed feasibility memo explaining that creation can be completed within a few days, followed by showing the completed dataset. This is a course requirement for that choice, not authorization for this agent to email anyone. No memo has been sent or proposal revision submitted by this setup.

## Source files

- Instructor comments supplied directly in this chat.
- `/Users/williambranam/Downloads/project_proposal (1).pdf`, page 1.
- `/Users/williambranam/Downloads/project_overview (1).pdf`, pages 1-2.
- `/Users/williambranam/Downloads/possible-projects (1).pdf`, especially page 5.
- Repository execution evidence in `reports/evidence/`.
