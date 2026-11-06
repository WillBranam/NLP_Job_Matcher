import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

from _bootstrap import ROOT
from job_matcher.modeling import generate, load_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair", type=Path, default=ROOT / "examples/pair.json")
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/predictions/base_example.json")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    args = parser.parse_args()
    config = json.loads((ROOT / "configs/project.json").read_text())
    model_path = ROOT / config["model"]["local_dir"]
    start = time.perf_counter()
    model, tokenizer, device = load_model(model_path, args.device)
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter, local_files_only=True)
    loaded = time.perf_counter()
    pair = json.loads(args.pair.read_text())
    result = generate(model, tokenizer, pair["resume"], pair["job"], max_new_tokens=args.max_new_tokens)
    result.update({"created_at": datetime.now(timezone.utc).isoformat(), "model": config["model"],
                   "adapter": str(args.adapter) if args.adapter else None,
                   "pair_file": str(args.pair), "example_type": pair.get("example_type", "unspecified"),
                   "device": device, "load_seconds": loaded - start,
                   "generation_seconds": time.perf_counter() - loaded})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
