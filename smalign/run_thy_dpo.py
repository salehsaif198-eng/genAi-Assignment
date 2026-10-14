import argparse
import json
from pathlib import Path

import torch

import smalign.dpo_setup as cfg
from smalign.dpo_run import get_ready
from smalign.dpo_core import compute_dpo_loss_logps, score_both_sides, the_dpo_batch
from smalign.dpo_valid import validate


def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, choices=["hh", "uf"])
    parser.add_argument("--sft", default=cfg.SFT_CHECKPOINT)

    return parser.parse_args()


def one_micro_step(policy, reference, tokenizer, prompts, chosen, rejected):
    
    batches = the_dpo_batch(tokenizer, prompts, chosen, rejected, cfg.MAX_LENGTH)

    with torch.autocast(device_type="cuda", dtype=torch.float16):
        policy_chosen, policy_rejected = score_both_sides(policy, batches, False)

    
    ref_chosen, ref_rejected = score_both_sides(reference, batches, True)

    device = policy_chosen.device

    return compute_dpo_loss_logps(
        policy_chosen,
        policy_rejected,
        ref_chosen.to(device),
        ref_rejected.to(device),
        cfg.BETA,
    )


def apply_update(policy, optimizer, scaler):
    
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg.CLIP_NORM)

    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad()


def main():
    args = read_args()

    data_dir = Path("data/dpo_" + args.data)
    out_dir = Path("outputs/dpo_" + args.data)
    resume_dir = out_dir / "resume"
    resume_file = resume_dir / "resume.pt"

    def load_the_pairs():
        
        train = json.loads((data_dir / "train.json").read_text(encoding="utf-8"))
        val = json.loads((data_dir / "val.json").read_text(encoding="utf-8"))

        return train, val

    def pick_up_where_we_left_off(optimizer, scaler):
        
        if not resume_file.exists():
            return 0, -1.0

        state = torch.load(resume_file, map_location="cpu")

        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])

        return state["step"], state["best_accuracy"]

    def write_down_progress(record):
        
        out_dir.mkdir(parents=True, exist_ok=True)

        with open(out_dir / "metrics.jsonl", "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")

        print(record, flush=True)

    train_pairs, val_pairs = load_the_pairs()

    total_steps = (len(train_pairs) // cfg.MICRO_BATCH // cfg.ACCUM_STEPS) * cfg.EPOCHS

    print("dataset:", args.data)
    print("train pairs:", len(train_pairs))
    print("optimizer steps:", total_steps)

    start_from = args.sft
    if resume_file.exists():
        start_from = resume_dir
        print("resuming from", resume_dir)

    policy, reference, tokenizer, optimizer, scaler = get_ready(start_from)

    step, best_accuracy = pick_up_where_we_left_off(optimizer, scaler)

    start_step = step
    micro_count = 0
    running_loss = 0.0

    optimizer.zero_grad()

    for epoch in range(cfg.EPOCHS):

        batch_source = cfg.give_me_the_next_batch(
            train_pairs, cfg.MICRO_BATCH, True, cfg.SEED + epoch
        )

        for prompts, chosen, rejected in batch_source:

            
            if micro_count >= start_step * cfg.ACCUM_STEPS:

                loss = one_micro_step(
                    policy, reference, tokenizer, prompts, chosen, rejected
                )

                scaler.scale(loss / cfg.ACCUM_STEPS).backward()
                running_loss = running_loss + loss.item()

                if (micro_count + 1) % cfg.ACCUM_STEPS == 0:
                    apply_update(policy, optimizer, scaler)
                    step = step + 1

                    if step % cfg.VALIDATE_EVERY == 0:

                        train_loss = running_loss / (cfg.VALIDATE_EVERY * cfg.ACCUM_STEPS)
                        val_loss, accuracy, margin = validate(
                            policy, reference, tokenizer, val_pairs
                        )
                        running_loss = 0.0

                        write_down_progress({
                            "step": step,
                            "train_loss": round(train_loss, 4),
                            "val_loss": round(val_loss, 4),
                            "reward_accuracy": round(accuracy, 4),
                            "reward_margin": round(margin, 4),
                        })

                        if accuracy > best_accuracy:
                            best_accuracy = accuracy
                            cfg.put_all_on_disk(policy, tokenizer, out_dir / "best")
                            print("  new best", flush=True)

                        cfg.put_all_on_disk(
                            policy, tokenizer, resume_dir,
                            the_optimizer=optimizer, the_scaler=scaler,
                            which_step=step, best_so_far=best_accuracy,
                        )

            micro_count = micro_count + 1

    cfg.put_all_on_disk(policy, tokenizer, out_dir / "final")
    print("done. best reward accuracy:", round(best_accuracy, 4))


if __name__ == "__main__":
    main()