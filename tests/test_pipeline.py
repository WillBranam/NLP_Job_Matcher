"""Check split separation and completion-only labels with the actual tokenizer."""
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from transformers import AutoTokenizer
from train_lora import Pairs
from job_matcher.modeling import prompt_tokens


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = AutoTokenizer.from_pretrained(
            ROOT / "models/base/Qwen2.5-1.5B-Instruct", local_files_only=True)

    def test_resume_groups_do_not_cross_splits(self):
        groups = []
        for split in ("train", "validation", "test"):
            rows = [json.loads(line) for line in (ROOT / f"data/processed/pilot/{split}.jsonl").read_text().splitlines()]
            self.assertTrue(rows)
            groups.append({row["resume_id"] for row in rows})
        self.assertFalse(groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])

    def test_prompt_truncation_retains_both_fields(self):
        ids, truncated = prompt_tokens(self.tokenizer, "Python graduate " * 1500, "SQL internship " * 1500, 512)
        text = self.tokenizer.decode(ids)
        self.assertTrue(truncated)
        self.assertLessEqual(len(ids), 512)
        self.assertIn("Python graduate", text)
        self.assertIn("SQL internship", text)

    def test_only_completion_has_loss_and_eos_is_supervised(self):
        pair = json.loads((ROOT / "examples/pair.json").read_text())
        pair["target"] = {"score": 75, "matching_skills": ["SQL"], "missing_skills": [],
                          "eligibility": "unclear", "explanation": "Example only"}
        item = Pairs([pair], self.tokenizer, 1024)[0]
        active = item["labels"] != -100
        self.assertTrue(active.any())
        self.assertTrue((item["labels"][item["attention_mask"] == 0] == -100).all())
        text = self.tokenizer.decode(item["labels"][active], skip_special_tokens=True)
        self.assertEqual(json.loads(text), pair["target"])
        self.assertEqual(item["labels"][active][-1].item(), self.tokenizer.eos_token_id)
        self.assertEqual(item["labels"][0].item(), -100)


if __name__ == "__main__":
    unittest.main()
