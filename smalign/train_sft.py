
import random
import json

from datasets import load_dataset

from pathlib import Path


DATASET = "HuggingFaceH4/ultrachat_200k"

N_TRAIN = 30000
SEED = 42
RAW_ROWS = 40000
N_VAL = 1000

OUT_DIR = Path("data/sft")


def download_n_rows(n):
    return load_dataset(DATASET, split="train_sft[:{}]".format(n))


def is_it_toUse(messages):

    

    if not messages or len(messages) < 2:
        return False

    if messages[1].get("role") != "assistant":
        return False
    
    if messages[0].get("role") != "user":
        return False

    else:
        return True


def take_first_turn(row):

    messages = row.get("messages")

    if not is_it_toUse(messages):
        return None

    prompt = messages[0].get("content", "").strip()
    response = messages[1].get("content", "").strip()

    if not prompt or not response:
        return None

    return {"prompt": prompt, "response": response}


def get_all_pairs(raw):

    

    pairs = []
    for row in raw:
        pair = take_first_turn(row)

        if pair is not None:
            pairs.append(pair)

    return pairs


def divide_the_pairs(pairs, n_train, n_val, seed):
    
    random.Random(seed).shuffle(pairs)

    needed = n_train + n_val
    if len(pairs) < needed:
        raise SystemExit("only " + str(len(pairs)) + " examples, need " + str(needed))

    train = pairs[:n_train]
    val = pairs[n_train:needed]

    return train, val


def write_the_js(rows, path):
    
    path.parent.mkdir(parents=True, exist_ok=True)

    text = json.dumps(rows, indent=2, ensure_ascii=False)
    path.write_text(text, encoding="utf-8")

def show(pair):
    
    print("example prompt:", pair["prompt"][:80])
    print("example response:", pair["response"][:80])


def describe(train, val):
    
    print("train examples:", len(train))
    print("val examples:", len(val))

    show(train[0])


print("loading", RAW_ROWS, "rows from", DATASET)
raw = download_n_rows(RAW_ROWS)

pairs = get_all_pairs(raw)
print("usable exchanges:", len(pairs))

train, val = divide_the_pairs(pairs, N_TRAIN, N_VAL, SEED)

write_the_js(train, OUT_DIR / "train.json")
write_the_js(val, OUT_DIR / "val.json")

describe(train, val)