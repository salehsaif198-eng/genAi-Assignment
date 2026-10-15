import json
import random
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer

SETTINGS = {
    "model": "HuggingFaceTB/SmolLM2-360M-Instruct",
    "data": Path("data/persona_lora/ramsay_style_preferences.json"),
    "out_dir": Path("outputs/lora_persona"),
    "n_val": 20,
    "seed": 42,
    "rank": 16,
    "alpha": 32,
    "dropout": 0.05,
    "targets": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "lr": 5e-5,
    "batch": 4,
    "epochs": 3,
    "beta": 0.1,
    "max_length": 512,
}


def make_it_chatty(pair):
    prompt_turn = [{"role": "user", "content": pair["prompt"]}]
    good_turn = [{"role": "assistant", "content": pair["chosen"]}]
    bad_turn = [{"role": "assistant", "content": pair["rejected"]}]

    return {"prompt": prompt_turn, "chosen": good_turn, "rejected": bad_turn}


def grab_and_cut(cfg):
    on_disk = Path(cfg["data"]).read_text(encoding="utf-8")
    rows = json.loads(on_disk)

    shuffler = random.Random(cfg["seed"])
    shuffler.shuffle(rows)

    cut = cfg["n_val"]
    held_back = []
    kept = []

    for spot, pair in enumerate(rows):
        if spot < cut:
            held_back.append(make_it_chatty(pair))
        else:
            kept.append(make_it_chatty(pair))

    return Dataset.from_list(kept), Dataset.from_list(held_back)


def grab_tokenizer(cfg):
    tok = AutoTokenizer.from_pretrained(cfg["model"])
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return tok


def the_adapter_bits(cfg):
    return LoraConfig(
        r=cfg["rank"],
        lora_alpha=cfg["alpha"],
        lora_dropout=cfg["dropout"],
        target_modules=cfg["targets"],
        bias="none",
        task_type="CAUSAL_LM",
    )


def the_trainer_bits(cfg):
    return DPOConfig(
        output_dir=str(cfg["out_dir"]),
        beta=cfg["beta"],
        learning_rate=cfg["lr"],
        per_device_train_batch_size=cfg["batch"],
        per_device_eval_batch_size=cfg["batch"],
        num_train_epochs=cfg["epochs"],
        max_length=cfg["max_length"],
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        fp16=torch.cuda.is_available(),
        report_to=[],
        seed=cfg["seed"],
    )


def say_where(folder):
    print("all finished, the adapter is sitting in", folder)


train_set, val_set = grab_and_cut(SETTINGS)
print("training on", len(train_set), "pairs, keeping", len(val_set), "back to check")

tokenizer = grab_tokenizer(SETTINGS)
base = AutoModelForCausalLM.from_pretrained(SETTINGS["model"])

SETTINGS["out_dir"].mkdir(parents=True, exist_ok=True)
saved_to = SETTINGS["out_dir"] / "adapter"

trainer = DPOTrainer(
    model=base,
    args=the_trainer_bits(SETTINGS),
    train_dataset=train_set,
    eval_dataset=val_set,
    processing_class=tokenizer,
    peft_config=the_adapter_bits(SETTINGS),
)

print("off we go")
trainer.train()

trainer.save_model(str(saved_to))
tokenizer.save_pretrained(str(saved_to))
say_where(saved_to)