import json
import random
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer

from smalign.prompts import configure_tokenizer, format_prompt

DATASET = "Anthropic/hh-rlhf"
MODEL = "Qwen/Qwen2.5-0.5B"

HUMAN = "\n\nHuman:"
ASSISTANT = "\n\nAssistant:"

RAW_ROWS = 60000
N_TRAIN = 15000
N_VAL = 1000
MAX_PROMPT_TOKENS = 256
SEED = 42

OUT_DIR = Path("data/dpo_hh")


def check_pair(pair, tokenizer):
    
    if pair["prompt"]:

        
        
        if pair["prompt"] == pair["other_prompt"]:

            
            if pair["raw"].count(HUMAN) == 1 and pair["raw"].count(ASSISTANT) == 1:

                if pair["chosen"] and pair["rejected"]:

                    if pair["chosen"] != pair["rejected"]:

                        ids = tokenizer.encode(
                            format_prompt(pair["prompt"]), add_special_tokens=False
                        )

                        if len(ids) <= MAX_PROMPT_TOKENS:
                            return None

                        return "prompt longer than the budget"

                    return "both sides say the same thing"

                return "one side has no response"

            return "conversation has more than one turn"

        return "the two transcripts disagree on the prompt"

    return "nothing before the assistant turn"


tokenizer = AutoTokenizer.from_pretrained(MODEL)
configure_tokenizer(tokenizer)

print("loading", RAW_ROWS, "rows from", DATASET)
raw = load_dataset(DATASET, split="train[:{}]".format(RAW_ROWS))


candidates = []

for row in raw:
    chosen_text = row.get("chosen", "")
    rejected_text = row.get("rejected", "")

    chosen_head, _, chosen_tail = chosen_text.rpartition(ASSISTANT)
    rejected_head, _, rejected_tail = rejected_text.rpartition(ASSISTANT)

    candidates.append({
        "prompt": chosen_head.replace(HUMAN, "", 1).strip(),
        "chosen": chosen_tail.strip(),
        "rejected": rejected_tail.strip(),
        "other_prompt": rejected_head.replace(HUMAN, "", 1).strip(),
        "raw": chosen_text,
    })

print("split", len(candidates), "transcripts")



pairs = []
dropped = {}

for pair in candidates:
    reason = check_pair(pair, tokenizer)

    if reason is None:
        pairs.append(pair)
        continue

    if reason not in dropped:
        dropped[reason] = 0

    dropped[reason] = dropped[reason] + 1

print("usable pairs:", len(pairs))
print("dropped:", dropped)



for pair in pairs:
    del pair["other_prompt"]
    del pair["raw"]


random.Random(SEED).shuffle(pairs)

needed = N_TRAIN + N_VAL
if len(pairs) < needed:
    raise SystemExit("only " + str(len(pairs)) + " pairs, need " + str(needed))

train = pairs[:N_TRAIN]
val = pairs[N_TRAIN:needed]

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