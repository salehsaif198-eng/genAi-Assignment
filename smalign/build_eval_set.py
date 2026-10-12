import json
import random
from pathlib import Path


dataset_repository = "tatsu-lab/alpaca_eval"
path_obj = Path("eval_sets/alpaca_eval_300_seed42.json")
N = 300
SEED = 42

from datasets import load_dataset

def load_alpaca_eval():

    data = load_dataset(dataset_repository, "alpaca_eval", split="eval")
    return list(data)


rows = load_alpaca_eval()
print("loaded", len(rows), "prompts from", dataset_repository)


rows = sorted(rows, key=lambda row: row["instruction"])
random.Random(SEED).shuffle(rows)


subset = []
for index, row in enumerate(rows[:N]):
    subset.append({
        "id": "alpaca_eval_" + str(index).zfill(4),
        "instruction": row["instruction"],
        "dataset": "alpaca_eval",
    })

path_obj.parent.mkdir(parents=True, exist_ok=True)

text = json.dumps(subset, indent=2, ensure_ascii=False)
path_obj.write_text(text, encoding="utf-8")

print("wrote", len(subset), "prompts to", path_obj)




















