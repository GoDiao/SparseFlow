from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import time
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from .common import sha256_file, write_json


DATASETS = {
    "hellaswag": {
        "dataset": "Rowan/hellaswag",
        "config": "default",
        "split": "validation",
    },
    "arc_challenge": {
        "dataset": "allenai/ai2_arc",
        "config": "ARC-Challenge",
        "split": "validation",
    },
    "mmlu": {
        "dataset": "cais/mmlu",
        "config": "all",
        "split": "test",
    },
}
LEVELS = {"smoke": 1, "pilot": 3, "development": 5, "formal": 20}


def fetch_json(url: str, attempts: int = 5) -> dict[str, Any]:
    last_error = None
    for attempt in range(attempts):
        try:
            request = Request(url, headers={"User-Agent": "SparseFlow/quality-manifest-v1"})
            with urlopen(request, timeout=60) as response:
                return json.load(response)
        except Exception as exc:  # Network failures are retried and surfaced with the URL.
            last_error = exc
            if attempt + 1 < attempts:
                time.sleep(min(5, attempt + 1))
    raise RuntimeError(f"failed to fetch {url}: {last_error}")


def dataset_revision(dataset: str) -> str:
    value = fetch_json(f"https://huggingface.co/api/datasets/{dataset}")
    revision = value.get("sha")
    if not revision:
        raise ValueError(f"dataset API did not return a revision for {dataset}")
    return str(revision)


def fetch_row(spec: dict[str, str], offset: int) -> tuple[int, dict[str, Any]]:
    query = urlencode(
        {
            "dataset": spec["dataset"],
            "config": spec["config"],
            "split": spec["split"],
            "offset": offset,
            "length": 1,
        }
    )
    value = fetch_json(f"https://datasets-server.huggingface.co/rows?{query}")
    rows = value.get("rows", [])
    if len(rows) != 1 or int(rows[0]["row_idx"]) != offset:
        raise ValueError(f"dataset row mismatch at offset {offset}: {spec}")
    return int(value["num_rows_total"]), rows[0]["row"]


def normalize_row(task: str, row_idx: int, row: dict[str, Any]) -> dict[str, Any]:
    if task == "hellaswag":
        choices = [" " + str(value).lstrip() for value in row["endings"]]
        gold = int(row["label"])
        ctx = str(row["ctx"]).strip()
        source_id = str(row.get("ind", row_idx))
    elif task == "arc_challenge":
        labels = [str(value) for value in row["choices"]["label"]]
        answer = str(row["answerKey"])
        if answer not in labels:
            raise ValueError(f"ARC answer {answer!r} is not in labels {labels!r}")
        choices = [" " + str(value).lstrip() for value in row["choices"]["text"]]
        gold = labels.index(answer)
        ctx = f"Question: {str(row['question']).strip()}\nAnswer:"
        source_id = str(row["id"])
    elif task == "mmlu":
        choices = [" " + str(value).lstrip() for value in row["choices"]]
        gold = int(row["answer"])
        ctx = f"Question: {str(row['question']).strip()}\nAnswer:"
        source_id = f"{row.get('subject', 'unknown')}:{row_idx}"
    else:
        raise ValueError(f"unsupported task: {task}")
    if not ctx or len(choices) < 2 or not 0 <= gold < len(choices):
        raise ValueError(f"invalid normalized row: task={task}, offset={row_idx}")
    return {
        "id": f"{task}:{source_id}",
        "task": task,
        "ctx": ctx,
        "choices": choices,
        "gold": gold,
        "source_row": row_idx,
        "source_id": source_id,
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    path.write_text(payload, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Freeze standard Stage 7.5 quality manifests.")
    parser.add_argument("--output-dir", default="benchmarks/manifests")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    output = Path(args.output_dir).expanduser().resolve()
    paths = {level: output / f"quality_{level}_v1.jsonl" for level in LEVELS}
    metadata_path = output / "quality_manifest_v1.meta.json"
    if not args.force and metadata_path.exists() and all(path.exists() for path in paths.values()):
        raise FileExistsError("quality manifest v1 already exists; use --force to replace it")

    per_task: dict[str, list[dict[str, Any]]] = {}
    sources = {}
    for task, spec in DATASETS.items():
        total, _ = fetch_row(spec, 0)
        indices = random.Random(f"{args.seed}:{task}").sample(
            range(total), LEVELS["formal"]
        )
        normalized = []
        for position, row_idx in enumerate(indices, 1):
            observed_total, row = fetch_row(spec, row_idx)
            if observed_total != total:
                raise ValueError(f"dataset size changed while fetching {task}")
            normalized.append(normalize_row(task, row_idx, row))
            print(f"task={task} row={position}/{len(indices)} offset={row_idx}", flush=True)
        per_task[task] = normalized
        sources[task] = {
            **spec,
            "revision": dataset_revision(spec["dataset"]),
            "total_rows": total,
            "sampled_rows": indices,
        }

    files = {}
    for level, count in LEVELS.items():
        rows = [row for task in DATASETS for row in per_task[task][:count]]
        write_jsonl(paths[level], rows)
        files[level] = {
            "path": str(paths[level]),
            "rows": len(rows),
            "sha256": sha256_file(paths[level]),
        }
    metadata = {
        "schema_version": 1,
        "kind": "sparseflow_quality_manifest_v1",
        "agent": "Main Dev",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "sampling": "independent random.Random(f'{seed}:{task}').sample without replacement",
        "levels": LEVELS,
        "sources": sources,
        "files": files,
    }
    write_json(metadata_path, metadata)
    print(json.dumps(files, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# [Main Dev]
