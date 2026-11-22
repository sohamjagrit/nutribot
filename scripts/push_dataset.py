"""Upload eval/dataset.json to a LangSmith dataset (idempotent: skips if it already exists).

Usage: uv run python scripts/push_dataset.py [--name nutribot-eval]
"""

import argparse
import json
from pathlib import Path

from dotenv import load_dotenv
from langsmith import Client

load_dotenv()
ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="nutribot-eval")
    name = ap.parse_args().name
    client = Client()
    if client.has_dataset(dataset_name=name):
        print(f"Dataset '{name}' already exists; delete it in LangSmith to re-upload.")
        return
    items = json.loads((ROOT / "eval" / "dataset.json").read_text())["items"]
    ds = client.create_dataset(name, description="NutriBot golden eval set")
    client.create_examples(
        dataset_id=ds.id,
        inputs=[{"question": i["question"]} for i in items],
        outputs=[{"expected_answer": i["expected_answer"]} for i in items],
        metadata=[{"id": i["id"], "relevant_keywords": i["relevant_keywords"]} for i in items],
    )
    print(f"Uploaded {len(items)} examples to '{name}'.")


if __name__ == "__main__":
    main()
