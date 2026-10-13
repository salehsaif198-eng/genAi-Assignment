from transformers import get_cosine_schedule_with_warmup
from transformers import AutoModelForCausalLM, AutoTokenizer

from smalign.prompts import configure_tokenizer
import smalign.sft_setup as cfg

import torch

def where_from():
    
    resume_dir = cfg.OUT_DIR / "resume"

    if (resume_dir / "resume.pt").exists():
        print("resuming from", resume_dir)
        return resume_dir

    return cfg.MODEL


def get_tokenizer(source):
    
    tokenizer = AutoTokenizer.from_pretrained(source)
    configure_tokenizer(tokenizer)

    return tokenizer


def get_model(source):
    
    model = AutoModelForCausalLM.from_pretrained(source, dtype=torch.float32)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.train()

    return model


def setup_optim(model, total_steps):
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.LEARNING_RATE,
        weight_decay=cfg.WEIGHT_DECAY,
    )

    warmup_steps = int(total_steps * cfg.WARMUP_FRACTION)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps,)

    return optimizer, scheduler


def get_ready(total_steps):
    source = where_from()

    tokenizer = get_tokenizer(source)
    model = get_model(source)
    optimizer, scheduler = setup_optim(model, total_steps)
    scaler = torch.amp.GradScaler()

    return model, tokenizer, optimizer, scheduler, scaler