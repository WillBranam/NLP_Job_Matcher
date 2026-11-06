"""Download pinned public assets; record hashes without requiring Hub credentials."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def fetch(url, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        temporary = destination.with_suffix(destination.suffix + ".part")
        request = urllib.request.Request(url, headers={"User-Agent": "NLP-Job-Matcher/0.1"})
        with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as target:
            shutil.copyfileobj(response, target, length=1024 * 1024)
        temporary.replace(destination)
    digest = hashlib.sha256()
    with destination.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    result = {"path": str(destination.relative_to(ROOT)), "url": url,
              "bytes": destination.stat().st_size, "sha256": digest.hexdigest()}
    print(f"Ready: {result['path']} ({result['bytes']:,} bytes)", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", choices=["model", "resumes", "jobs", "all"], default="all")
    args = parser.parse_args()
    config = json.loads((ROOT / "configs/project.json").read_text())
    tasks = []
    if args.only in ("model", "all"):
        item = config["model"]
        for name in ["LICENSE", "README.md", "config.json", "generation_config.json",
                     "merges.txt", "model.safetensors", "tokenizer.json", "tokenizer_config.json", "vocab.json"]:
            tasks.append((f"https://huggingface.co/{item['repo_id']}/resolve/{item['revision']}/{name}",
                          ROOT / item["local_dir"] / name))
    if args.only in ("resumes", "all"):
        item = config["resume_dataset"]
        for name in ["README.md", "job_resume_fit.csv"]:
            tasks.append((f"https://huggingface.co/datasets/{item['repo_id']}/resolve/{item['revision']}/{name}",
                          ROOT / item["local_dir"] / name))
    if args.only in ("jobs", "all"):
        item = config["job_dataset"]
        tasks.append((item["download_url"], ROOT / item["local_dir"] / "archive.zip"))
    manifest = {"created_at": datetime.now(timezone.utc).isoformat(), "assets": [], "errors": []}
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [(url, executor.submit(fetch, url, path)) for url, path in tasks]
        for url, future in futures:
            try:
                manifest["assets"].append(future.result())
            except Exception as error:
                manifest["errors"].append({"url": url, "error": str(error)})
                print(f"FAILED: {url}: {error}", flush=True)
    archive = ROOT / config["job_dataset"]["local_dir"] / "archive.zip"
    if args.only in ("jobs", "all") and archive.exists():
        try:
            with zipfile.ZipFile(archive) as source:
                for member in source.infolist():
                    path = (archive.parent / member.filename).resolve()
                    if not path.is_relative_to(archive.parent.resolve()):
                        raise ValueError("Unsafe archive path")
                source.extractall(archive.parent)
            print("Extracted LinkedIn archive", flush=True)
        except Exception as error:
            manifest["errors"].append({"archive": str(archive), "error": str(error)})
    output = ROOT / "outputs/downloads"
    output.mkdir(parents=True, exist_ok=True)
    (output / f"{args.only}_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if manifest["errors"]:
        raise SystemExit("Some downloads failed; see the manifest. Successful files are preserved.")


if __name__ == "__main__":
    main()
