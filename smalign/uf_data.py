import json
from transformers import AutoTokenizer
import random

from smalign.prompts import configure_tokenizer, format_prompt
from pathlib import Path
from datasets import load_dataset


DATASET = "HuggingFaceH4/ultrafeedback_binarized"
MODEL = "Qwen/Qwen2.5-0.5B"

SEED = 42
RAW_ROWS = 22000
MAX_PROMPT_TOKENS = 256
N_TRAIN = 15000
N_VAL = 1000

OUT_DIR = Path("data/dpo_uf")


def extract_pair(row):

    prompt = row.get("prompt", "").strip()


    chosen_messages = row.get("chosen") or [{}]
    rejected_messages = row.get("rejected") or [{}]

    return {
        "prompt": prompt,
        "chosen": chosen_messages[-1].get("content", "").strip(),
        "rejected": rejected_messages[-1].get("content", "").strip(),
    }


def check_pair(pair, tokenizer):
    
    if pair["prompt"]:

        if pair["chosen"] and pair["rejected"]:

            if pair["chosen"] != pair["rejected"]:

                ids = tokenizer.encode(
                    format_prompt(pair["prompt"]), add_special_tokens=False
                )

                if len(ids) <= MAX_PROMPT_TOKENS:
                    return None

                return "prompt too long"

            return "identical responses"

        return "empty response"

    return "empty prompt"


def divide_the_pairs(pairs, n_train, n_val, seed):
    
    random.Random(seed).shuffle(pairs)

    needed = n_train + n_val

    if len(pairs) >= needed:
        train = pairs[:n_train]
        val = pairs[n_train:needed]
        return train, val

    raise SystemExit("only " + str(len(pairs)) + " pairs, need " + str(needed))


tokenizer = AutoTokenizer.from_pretrained(MODEL)
configure_tokenizer(tokenizer)

print("loading", RAW_ROWS, "rows from", DATASET)
raw = load_dataset(DATASET, split="train_prefs[:{}]".format(RAW_ROWS))

pairs = []
dropped = {}

for row in raw:
    pair = extract_pair(row)
    reason = check_pair(pair, tokenizer)

    if reason is None:
        pairs.append(pair)
        continue

    if reason not in dropped:
        dropped[reason] = 0

    dropped[reason] = dropped[reason] + 1

print("usable pairs:", len(pairs))
print("dropped:", dropped)

train, val = divide_the_pairs(pairs, N_TRAIN, N_VAL, SEED)

OUT_DIR.mkdir(parents=True, exist_ok=True)

train_text = json.dumps(train, indent=2, ensure_ascii=False)
(OUT_DIR / "train.json").write_text(train_text, encoding="utf-8")

val_text = json.dumps(val, indent=2, ensure_ascii=False)
(OUT_DIR / "val.json").write_text(val_text, encoding="utf-8")

def showsplit(train, val):
    
    print("built", len(train), "training pairs and", len(val), "validation pairs")
    print()
    print("sample prompt   :", train[0]["prompt"][:80])
    print("sample chosen   :", train[0]["chosen"][:80])
    print("sample rejected :", train[0]["rejected"][:80])

showsplit(train, val)