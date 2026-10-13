import json

import torch

import smalign.sft_setup as cfg
from smalign.sft_build import get_ready
from smalign.sft_main import compute_sft_loss


def validate(model, tokenizer, val_pairs):
    model.eval()

    total = 0.0
    count = 0

    with torch.no_grad():
        for prompts, responses in cfg.form_the_batch(val_pairs, cfg.MICRO_BATCH, False, cfg.SEED):
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                loss = compute_sft_loss(model, tokenizer, prompts, responses, cfg.MAX_LENGTH)

            total = total + loss.item()
            count = count + 1
    model.train()

    return total / count


def apply_update(model, optimizer, scheduler, scaler):

    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.CLIP_NORM)

    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad()
    scheduler.step()


train_pairs = cfg.load_pairs(cfg.DATA_DIR / "train.json")
val_pairs = cfg.load_pairs(cfg.DATA_DIR / "val.json")

micro_per_epoch = len(train_pairs) // cfg.MICRO_BATCH
total_steps = (micro_per_epoch // cfg.ACCUM_STEPS) * cfg.EPOCHS

print("train examples:", len(train_pairs))
print("optimizer steps:", total_steps)

model, tokenizer, optimizer, scheduler, scaler = get_ready(total_steps)

resume_dir = cfg.OUT_DIR / "resume"
start_step, best_val = cfg.load_all(resume_dir, optimizer, scheduler, scaler)

step = start_step
micro_count = 0
running_loss = 0.0

optimizer.zero_grad()

for epoch in range(cfg.EPOCHS):
    for prompts, responses in cfg.form_the_batch(train_pairs, cfg.MICRO_BATCH, True, cfg.SEED + epoch):

        
        if micro_count >= start_step * cfg.ACCUM_STEPS:

            with torch.autocast(device_type="cuda", dtype=torch.float16):
                loss = compute_sft_loss(model, tokenizer, prompts, responses, cfg.MAX_LENGTH)

            #Divide so the accumulated gradient averages over the effective batch.
            scaler.scale(loss / cfg.ACCUM_STEPS).backward()
            running_loss = running_loss + loss.item()

            if (micro_count + 1) % cfg.ACCUM_STEPS == 0:
                apply_update(model, optimizer, scheduler, scaler)
                step = step + 1

                if step % cfg.VALIDATE_EVERY == 0:
                    train_loss = running_loss / (cfg.VALIDATE_EVERY * cfg.ACCUM_STEPS)
                    val_loss = validate(model, tokenizer, val_pairs)
                    running_loss = 0.0

                    record = {
                        "step": step,
                        "train_loss": round(train_loss, 4),
                        "val_loss": round(val_loss, 4),
                        "lr": scheduler.get_last_lr()[0],
                    }

                    cfg.OUT_DIR.mkdir(parents=True, exist_ok=True)
                    with open(cfg.OUT_DIR / "metrics.jsonl", "a", encoding="utf-8") as handle:
                        handle.write(json.dumps(record) + "\n")

                    print(record, flush=True)

                    if val_loss < best_val:
                        best_val = val_loss
                        cfg.save_all(model, tokenizer, cfg.OUT_DIR / "best")
                        print("  new best", flush=True)

                    cfg.save_all(
                        model, tokenizer, resume_dir,
                        optimizer=optimizer, scheduler=scheduler, scaler=scaler,
                        step=step, best_val=best_val,
                    )
        micro_count = micro_count + 1

cfg.save_all(model, tokenizer, cfg.OUT_DIR / "final")
print("done. best val loss:", round(best_val, 4))