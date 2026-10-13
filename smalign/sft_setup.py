import json
import random
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import get_cosine_schedule_with_warmup

from smalign.prompts import configure_tokenizer, format_prompt
from smalign.sft_main import compute_sft_loss

MODEL = "Qwen/Qwen2.5-0.5B"
DATA_DIR = Path("data/sft")
OUT_DIR = Path("outputs/sft")

MAX_LENGTH = 512
MICRO_BATCH = 2
ACCUM_STEPS = 16
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.1
WARMUP_FRACTION = 0.03
CLIP_NORM = 1.0
EPOCHS = 1
VALIDATE_EVERY = 50
SEED = 42



def load_pairs(path):
    
    return json.loads(Path(path).read_text(encoding="utf-8"))


def shuffled_order(n, shuffle, seed):
    
    order = list(range(n))

    if shuffle:
        random.Random(seed).shuffle(order)

    return order


def form_the_batch(pairs, size, shuffle, seed):
    
    order = shuffled_order(len(pairs), shuffle, seed)

    for start in range(0, len(order), size):
        chunk = order[start : start + size]

        rows = [pairs[index] for index in chunk]

        prompts = [format_prompt(row["prompt"]) for row in rows]
        responses = [row["response"].strip() for row in rows]

        yield prompts, responses


def save_all(model, tokenizer, folder, optimizer=None, scheduler=None,
             scaler=None, step=None, best_val=None):
    
    folder.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(folder)
    tokenizer.save_pretrained(folder)

    if optimizer is not None:
        torch.save(
            {
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "step": step,
                "best_val": best_val,
            },
            folder / "resume.pt",
        )


def load_all(folder, optimizer, scheduler, scaler):
    
    path = folder / "resume.pt"
    if not path.exists():
        return 0, float("inf")

    state = torch.load(path, map_location="cuda" if torch.cuda.is_available() else "cpu")
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    scaler.load_state_dict(state["scaler"])

    return state["step"], state["best_val"]