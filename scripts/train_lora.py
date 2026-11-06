"""Completion-only LoRA SFT, with explicit pilot/full-training separation."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

from _bootstrap import ROOT
import torch
from torch.utils.data import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import Trainer, TrainingArguments, set_seed
import yaml

from job_matcher.modeling import load_model, prompt_tokens


class Pairs(Dataset):
    def __init__(self, rows, tokenizer, max_length):
        self.items = []
        self.truncated = 0
        for row in rows:
            # Append the assistant end marker; never train on prompt or padding.
            completion = tokenizer.encode(json.dumps(row["target"], ensure_ascii=False) + tokenizer.eos_token,
                                          add_special_tokens=False)
            prompt, truncated = prompt_tokens(tokenizer, row["resume"], row["job"], max_length - len(completion))
            ids = prompt + completion
            padding = max_length - len(ids)
            self.items.append({"input_ids": torch.tensor(ids + [tokenizer.pad_token_id] * padding),
                               "attention_mask": torch.tensor([1] * len(ids) + [0] * padding),
                               "labels": torch.tensor([-100] * len(prompt) + completion + [-100] * padding)})
            self.truncated += int(truncated)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/smoke.yaml")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    project = json.loads((ROOT / "configs/project.json").read_text())
    set_seed(config["seed"])
    rows = [json.loads(line) for line in (ROOT / config["data"]).read_text().splitlines() if line.strip()]
    if not config["smoke"] and any(row.get("scope") != "reviewed_recent_graduate" for row in rows):
        raise ValueError("Full training requires manually reviewed recent-graduate pairs (scope=reviewed_recent_graduate).")
    if config["smoke"] and not 0 < config["max_steps"] <= 10:
        raise ValueError("Smoke runs must be limited to 1–10 steps")
    if config["max_samples"]:
        rows = rows[:config["max_samples"]]
    if not rows:
        raise ValueError("No training rows")
    started = time.perf_counter()
    model, tokenizer, device = load_model(ROOT / project["model"]["local_dir"], args.device)
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=config["lora_r"], lora_alpha=config["lora_alpha"],
        lora_dropout=config["lora_dropout"], target_modules=config["target_modules"], bias="none"))
    model.print_trainable_parameters()
    data = Pairs(rows, tokenizer, config["max_length"])
    training = TrainingArguments(
        output_dir=str(ROOT / config["output_dir"]), num_train_epochs=config["epochs"],
        max_steps=config["max_steps"], learning_rate=config["learning_rate"],
        per_device_train_batch_size=config["batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        logging_steps=1, save_strategy="no", report_to="none", seed=config["seed"],
        dataloader_pin_memory=device.startswith("cuda"), use_cpu=device == "cpu",
        fp16=device.startswith("cuda"), gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_torch", remove_unused_columns=False)
    trainer = Trainer(model=model, args=training, train_dataset=data, processing_class=tokenizer)
    result = trainer.train()
    adapter = ROOT / config["output_dir"]
    model.save_pretrained(adapter)
    tokenizer.save_pretrained(adapter)
    evidence = ROOT / config["evidence_dir"]
    evidence.mkdir(parents=True, exist_ok=True)
    trainable, total = model.get_nb_trainable_parameters()
    summary = {"created_at": datetime.now(timezone.utc).isoformat(), "config": config,
               "model": project["model"], "device": device, "samples": len(data),
               "truncated_samples": data.truncated, "trainable_parameters": trainable,
               "total_parameters": total, "metrics": result.metrics,
               "log_history": trainer.state.log_history,
               "elapsed_seconds": time.perf_counter() - started,
               "adapter_path": str(adapter), "adapter_reload_verified": False,
               "interpretation": "Pipeline execution only; no matching-quality claim." if config["smoke"] else "Training completed; independent evaluation still required."}
    (evidence / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    trainer.state.save_to_json(str(evidence / "trainer_state.json"))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
