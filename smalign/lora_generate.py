from transformers import AutoModelForCausalLM, AutoTokenizer

from smalign.lora_dpo import SETTINGS, grab_and_cut
import torch
from peft import PeftModel
from pathlib import Path

import json

BASE = "HuggingFaceTB/SmolLM2-360M-Instruct"
ADAPTER = Path("outputs/lora_persona/adapter")
OUT_DIR = Path("generations")

HOW_MANY = 100
BATCH = 8
NEW_TOKENS = 256


def fire_up(wearing_adapter):
    tok = AutoTokenizer.from_pretrained(BASE)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    net = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.float16)
    if wearing_adapter:
        net = PeftModel.from_pretrained(net, str(ADAPTER))

    net.to("cuda" if torch.cuda.is_available() else "cpu")
    net.eval()
    return net, tok


def chat_them_up(net, tok, chunk):
    as_chat = []
    for text in chunk:
        turn = [{"role": "user", "content": text}]
        as_chat.append(
            tok.apply_chat_template(turn, tokenize=False, add_generation_prompt=True)
        )

    packed = tok(as_chat, return_tensors="pt", padding=True, truncation=True, max_length=512)
    packed = packed.to(net.device)

    how_to_talk = {
        "max_new_tokens": NEW_TOKENS,
        "do_sample": False,
        "pad_token_id": tok.pad_token_id,
    }

    with torch.inference_mode():
        spat_out = net.generate(**packed, **how_to_talk)

    how_wide = packed["input_ids"].shape[1]
    fresh = spat_out[:, how_wide:]

    said = tok.batch_decode(fresh, skip_special_tokens=True)
    return [line.strip() for line in said]


if __name__ == "__main__":
    kept, held_back = grab_and_cut(SETTINGS)
    style_asks = [row["prompt"][0]["content"] for row in held_back]

    alpaca = json.loads(Path("eval_sets/alpaca_eval_300_seed42.json").read_text(encoding="utf-8"))
    useful_asks = [row["instruction"] for row in alpaca[:HOW_MANY]]

    hh = json.loads(Path("eval_sets/hh_rlhf_eval_300_seed42.json").read_text(encoding="utf-8"))
    safe_asks = [row["instruction"] for row in hh[:HOW_MANY]]

    jobs = [("style", style_asks), ("alpaca", useful_asks), ("safety", safe_asks)]
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for wearing_adapter in [False, True]:
        if wearing_adapter:
            label = "lora"
        else:
            label = "base"

        net, tok = fire_up(wearing_adapter)

        which_job = 0
        while which_job < len(jobs):
            name, questions = jobs[which_job]
            done = []

            for start in range(0, len(questions), BATCH):
                chunk = questions[start:start + BATCH]
                said = chat_them_up(net, tok, chunk)

                spot = 0
                while spot < len(chunk):
                    done.append({
                        "id": "item_" + str(len(done)).zfill(4),
                        "instruction": chunk[spot],
                        "output": said[spot],
                        "generator": label,
                    })
                    spot = spot + 1

                print(label, name, len(done), "of", len(questions))

            out_file = OUT_DIR / ("smollm_" + label + "_" + name + ".json")
            out_file.write_text(json.dumps(done, indent=2, ensure_ascii=False), encoding="utf-8")

            which_job = which_job + 1

        del net
        torch.cuda.empty_cache()

    print("six files written into", OUT_DIR)