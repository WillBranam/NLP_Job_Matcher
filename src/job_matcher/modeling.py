import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .prompts import messages


def select_device(requested="auto"):
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(path, device="auto"):
    device = select_device(device)
    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        path, local_files_only=True, torch_dtype=dtype, attn_implementation="eager")
    model.to(device)
    return model, tokenizer, device


def prompt_tokens(tokenizer, resume, job, budget):
    """Bound both fields, keeping the output space separate; disclose truncation."""
    original = tokenizer.apply_chat_template(messages(resume, job), tokenize=True, add_generation_prompt=True)
    if len(original) <= budget:
        return original, False
    overhead = len(tokenizer.apply_chat_template(messages("", ""), tokenize=True, add_generation_prompt=True))
    available = budget - overhead - 16
    if available < 64:
        raise ValueError("Insufficient prompt budget; increase max_length")
    resume_ids = tokenizer.encode(resume, add_special_tokens=False)
    job_ids = tokenizer.encode(job, add_special_tokens=False)
    for _ in range(30):
        resume_budget = min(len(resume_ids), int(available * 0.6))
        job_budget = min(len(job_ids), available - resume_budget)
        resume_budget = min(len(resume_ids), available - job_budget)
        r = tokenizer.decode(resume_ids[:resume_budget], skip_special_tokens=True)
        j = tokenizer.decode(job_ids[:job_budget], skip_special_tokens=True)
        encoded = tokenizer.apply_chat_template(messages(r, j), tokenize=True, add_generation_prompt=True)
        if len(encoded) <= budget:
            return encoded, True
        available -= max(16, len(encoded) - budget)
        if available < 64:
            break
    raise ValueError("Could not fit prompt in token budget")


def generate(model, tokenizer, resume, job, max_input_tokens=3584, max_new_tokens=512):
    ids, truncated = prompt_tokens(tokenizer, resume, job, max_input_tokens)
    inputs = torch.tensor([ids], dtype=torch.long, device=model.device)
    model.eval()
    with torch.inference_mode():
        output = model.generate(input_ids=inputs, attention_mask=torch.ones_like(inputs),
                                max_new_tokens=max_new_tokens, do_sample=False,
                                pad_token_id=tokenizer.pad_token_id)
    text = tokenizer.decode(output[0, len(ids):], skip_special_tokens=True)
    parsed, valid = None, False
    try:
        parsed = json.loads(text)
        valid = (isinstance(parsed, dict) and type(parsed.get("score")) in (int, float)
                 and 0 <= parsed["score"] <= 100
                 and all(isinstance(parsed.get(key), list)
                         and all(isinstance(x, str) for x in parsed[key])
                         for key in ("matching_skills", "missing_skills"))
                 and parsed.get("eligibility") in ("eligible", "ineligible", "unclear")
                 and isinstance(parsed.get("explanation"), str))
    except (ValueError, TypeError):
        pass
    return {"raw_output": text, "parsed_output": parsed, "valid_schema": valid,
            "input_tokens": len(ids), "generated_tokens": len(output[0]) - len(ids),
            "input_truncated": truncated}
