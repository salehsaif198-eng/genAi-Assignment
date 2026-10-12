
import json
from pathlib import Path
import time
import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from smalign.prompts import configure_tokenizer, format_prompt

BATCH_SIZE = 16
MAX_PROMPT_TOKENS = 512
MAX_NEW_TOKENS = 256


def load_eval_set(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def does_json_exist(path):
    if not path.exists():
        return []

    else:
        return json.loads(path.read_text(encoding="utf-8"))


def load_is_done(path):
    the_rows = does_json_exist(path)
    
    done = {}
    for row in the_rows:
        if row["output"].strip():
            done[row["id"]] = row
    return done

def which_is_device():
    
    if torch.cuda.is_available():
        return "cuda"

    else:
        return "cpu"

def load_the_model(path_of_model):

    this_tokenizer = AutoTokenizer.from_pretrained(path_of_model)
    configure_tokenizer(this_tokenizer)

    this_tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(path_of_model, dtype=torch.float16)
    model.to(which_is_device())
    model.eval()

    return model, this_tokenizer


def set_the_setting(tokenizer, args):
    
    all_settings = {
        "max_new_tokens": MAX_NEW_TOKENS,
        "do_sample": False,
        "num_beams": 1,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }

    if args.repetition_penalty != 1.0:
        all_settings["repetition_penalty"] = args.repetition_penalty

    if args.no_repeat_ngram_size > 0:
        all_settings["no_repeat_ngram_size"] = args.no_repeat_ngram_size

    return all_settings

def to_decode_completions(tokenizer, generated, prompt_width):
    
    texts = []
    for row in generated:
        new_tokens = row[prompt_width:]
        texts.append(tokenizer.decode(new_tokens, skip_special_tokens=True).strip())

    return texts

def generate_batch(model, tokenizer, instructions, args):
    prompts = [format_prompt(text) for text in instructions]

    encoded = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_PROMPT_TOKENS,
    ).to(model.device)


    settings = set_the_setting(tokenizer, args)
    with torch.no_grad():
        generated = model.generate(**encoded, **settings)

    the_prompt_width = encoded["input_ids"].shape[1]
    return to_decode_completions(tokenizer, generated, the_prompt_width)

def which_are_left(rows, done):
    
    todo = []
    for row in rows:
        if row["id"] not in done:
            todo.append(row)

    return todo


def add_required_args(parser):
    
    parser.add_argument("--model", required=True)
    parser.add_argument("--eval-set", required=True)
    parser.add_argument("--generator", required=True)
    parser.add_argument("--out", required=True, type=Path)


def add_decoding_args(parser):
    
    parser.add_argument("--dataset-name", default="alpaca_eval")
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=0)


def read_args():
    parser = argparse.ArgumentParser()
    add_required_args(parser)
    add_decoding_args(parser)
    return parser.parse_args()


args = read_args()

rows = load_eval_set(args.eval_set)
args.out.parent.mkdir(parents=True, exist_ok=True)

done = load_is_done(args.out)
todo = which_are_left(rows, done)

print("the prompts:", len(rows))
print("are done already:", len(done))
print("remaining to generate are:", len(todo))

if todo:
    model, tokenizer = load_the_model(args.model)
    results = dict(done)
    started = time.time()

    for start in range(0, len(todo), BATCH_SIZE):
        chunk = todo[start : start + BATCH_SIZE]
        instructions = [row["instruction"] for row in chunk]
        outputs = generate_batch(model, tokenizer, instructions, args)

        for row, output in zip(chunk, outputs):
            results[row["id"]] = {
                "id": row["id"],
                "instruction": row["instruction"],
                "output": output,
                "generator": args.generator,
                "dataset": args.dataset_name,
            }

        ordered = [results[row["id"]] for row in rows if row["id"] in results]
        text = json.dumps(ordered, indent=2, ensure_ascii=False)
        args.out.write_text(text, encoding="utf-8")

        elapsed = round(time.time() - started)
        print("  done", min(start + BATCH_SIZE, len(todo)), "of", len(todo),
              " (" + str(elapsed) + "s)", flush=True)

    empty = 0
    for row in results.values():
        if not row["output"].strip():
            empty = empty + 1

    print("wrote", len(results), "records to", args.out)
    if empty:
        print("warning:", empty, "generations were empty")