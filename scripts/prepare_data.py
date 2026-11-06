"""Inspect sources; create explicitly provisional, resume-disjoint pilot splits."""
import ast
import csv
from collections import Counter
import hashlib
import json
import math
import random
import re
import unicodedata

from _bootstrap import ROOT


def normalized(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def identity(text):
    return hashlib.sha256(normalized(text).encode()).hexdigest()


def save_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


def skill_list(value):
    result = ast.literal_eval(value)
    if not isinstance(result, list) or not all(isinstance(x, str) for x in result):
        raise ValueError("Invalid source skill list")
    return result


def main():
    config = json.loads((ROOT / "configs/project.json").read_text())
    source = ROOT / config["resume_dataset"]["local_dir"] / "job_resume_fit.csv"
    with source.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    pilot, seen, duplicates, invalid = [], set(), 0, 0
    review = []
    for row in rows:
        if row["category"] not in config["tech_categories"]:
            continue
        resume, job = row["resume_text"].strip(), row["job_text"].strip()
        try:
            score = float(row["ai_match_score"])
            if not resume or not job or not math.isfinite(score) or not 0 <= score <= 100:
                raise ValueError("Invalid text or score")
            matched = skill_list(row["ai_matched_skills"])
            required = skill_list(row["job_required_skills"])
        except (ValueError, SyntaxError):
            invalid += 1
            continue
        resume_id, job_id = identity(resume), identity(job)
        pair_id = (resume_id, job_id)
        if pair_id in seen:
            duplicates += 1
            continue
        seen.add(pair_id)
        # These are source-derived proxy labels, not independent human judgments.
        matched_set = {normalized(x) for x in matched}
        missing = [x for x in required if normalized(x) not in matched_set]
        target = {"score": score, "matching_skills": matched, "missing_skills": missing,
                  "eligibility": "unclear",
                  "explanation": "Proxy skill assessment from the source dataset; experience, education, and eligibility require manual verification."}
        pilot.append({"source_id": row["ID"], "resume_id": resume_id, "job_id": job_id,
                      "resume": resume, "job": job, "target": target,
                      "label_source": "source_ai_score_and_skill_lists",
                      "scope": "pipeline_pilot_not_recent_graduate_gold_data"})
        review.append({"source_id": row["ID"], "resume_id": resume_id, "job_id": job_id,
                       "resume_scope": "unreviewed", "job_scope": "out_of_scope_senior_role",
                       "notes": "Review resume manually; IT job requires 8–10 years."})
    groups = sorted({row["resume_id"] for row in pilot})
    random.Random(config["seed"]).shuffle(groups)
    n_train = int(len(groups) * config["split_fractions"]["train"])
    n_val = int(len(groups) * config["split_fractions"]["validation"])
    split_ids = {"train": set(groups[:n_train]), "validation": set(groups[n_train:n_train+n_val]),
                 "test": set(groups[n_train+n_val:])}
    counts = {}
    for split, ids in split_ids.items():
        subset = [row for row in pilot if row["resume_id"] in ids]
        save_jsonl(ROOT / "data/processed/pilot" / f"{split}.jsonl", subset)
        counts[split] = len(subset)
    save_jsonl(ROOT / "data/interim/resume_scope_review.jsonl", review)
    summary = {"source_rows": len(rows), "categories": dict(Counter(row["category"] for row in rows)),
               "unique_job_descriptions": len({identity(row["job_text"]) for row in rows}),
               "pilot_rows": len(pilot), "pilot_unique_resumes": len(groups),
               "pilot_unique_jobs": len({row["job_id"] for row in pilot}),
               "duplicate_pairs_removed": duplicates, "invalid_tech_rows_removed": invalid,
               "splits": counts, "seed": config["seed"], "resume_overlap_between_splits": False,
               "limitations": ["Pilot only; the IT posting is a senior infrastructure-management role.",
                               "AI labels and generated skill-list targets are not human gold labels.",
                               "Identical jobs cross splits; these splits do not test unseen-job generalization.",
                               "Exact normalized-text deduplication only; near-duplicate review remains.",
                               "No verified recent-graduate training or test set exists yet."]}
    jobs_dir = ROOT / config["job_dataset"]["local_dir"]
    postings = sorted(jobs_dir.rglob("postings.csv")) + sorted(jobs_dir.rglob("job_postings.csv"))
    if postings:
        total, empty, duplicate_jobs = 0, 0, 0
        candidates, seen_jobs = [], set()
        levels = Counter()
        tech_title = re.compile(r"\b(software|developer|data analyst|data scien(?:ce|tist)|cyber|security analyst|IT support|help ?desk|computer|web|QA|quality assurance|network|machine learning)\b", re.I)
        junior = re.compile(r"\b(intern(ship)?|entry[- ]level|junior|new grad(uate)?|graduate|early career)\b", re.I)
        senior = re.compile(r"\b(senior|sr\.?|lead|principal|director|manager|staff)\b", re.I)
        with postings[0].open(newline="", encoding="utf-8-sig") as stream:
            for row in csv.DictReader(stream):
                total += 1
                title, description = row.get("title", "").strip(), row.get("description", "").strip()
                level = row.get("formatted_experience_level", "").strip()
                levels[level or "unknown"] += 1
                if not title or not description:
                    empty += 1
                    continue
                key = identity(title + " " + description)
                if key in seen_jobs:
                    duplicate_jobs += 1
                    continue
                seen_jobs.add(key)
                if tech_title.search(title) and not senior.search(title) and (junior.search(title) or level.casefold() in {"entry level", "internship"}):
                    candidates.append({"job_id": row.get("job_id"), "title": title, "job": description,
                                       "experience_level": level, "source": config["job_dataset"]["source"],
                                       "scope_review": "unreviewed", "required_skills": [],
                                       "eligibility_review": "Check years, education, enrollment, location, and work authorization manually."})
        save_jsonl(ROOT / "data/interim/linkedin_entry_tech_candidates.jsonl", candidates)
        summary["linkedin"] = {"source_file": str(postings[0].relative_to(ROOT)), "rows": total,
                               "missing_text_rows": empty, "duplicate_title_description_rows": duplicate_jobs,
                               "experience_levels": dict(levels), "unreviewed_entry_tech_candidates": len(candidates),
                               "note": "Heuristic candidate pool, not confirmed eligibility or labeled test pairs."}
    else:
        summary["linkedin"] = {"status": "not_downloaded_or_not_extracted"}
    output = ROOT / "outputs/data"
    output.mkdir(parents=True, exist_ok=True)
    (output / "dataset_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
