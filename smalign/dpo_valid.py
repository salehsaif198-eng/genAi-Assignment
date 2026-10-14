import smalign.dpo_setup as cfg
from smalign.dpo_core import (
    compute_dpo_loss_logps,
    rewards_being_implicit,
    score_both_sides,
    the_dpo_batch,
)


def score_everything(policy, reference, tokenizer, prompts, chosen, rejected):
    
    batches = the_dpo_batch(tokenizer, prompts, chosen, rejected, cfg.MAX_LENGTH)

    policy_chosen, policy_rejected = score_both_sides(policy, batches, True)
    ref_chosen, ref_rejected = score_both_sides(reference, batches, True)

    device = policy_chosen.device

    return policy_chosen, policy_rejected, ref_chosen.to(device), ref_rejected.to(device)


def both_rewards(policy_chosen, policy_rejected, ref_chosen, ref_rejected):
    
    reward_chosen = rewards_being_implicit(policy_chosen, ref_chosen, cfg.BETA)
    reward_rejected = rewards_being_implicit(policy_rejected, ref_rejected, cfg.BETA)

    return reward_chosen, reward_rejected


def how_many_right(reward_chosen, reward_rejected):
    
    return (reward_chosen > reward_rejected).sum().item()


def how_far_apart(reward_chosen, reward_rejected):
    
    return (reward_chosen - reward_rejected).sum().item()


def empty_totals():
    
    return {"loss": 0.0, "correct": 0, "margin": 0.0, "seen": 0, "batches": 0}


def add_up(totals, loss, reward_chosen, reward_rejected, how_many):
    
    totals["loss"] = totals["loss"] + loss.item()
    totals["correct"] = totals["correct"] + how_many_right(reward_chosen, reward_rejected)
    totals["margin"] = totals["margin"] + how_far_apart(reward_chosen, reward_rejected)
    totals["seen"] = totals["seen"] + how_many
    totals["batches"] = totals["batches"] + 1


def validate(policy, reference, tokenizer, val_pairs):
    
    policy.eval()

    totals = empty_totals()

    batch_source = cfg.give_me_the_next_batch(val_pairs, cfg.MICRO_BATCH, False, cfg.SEED)

    for prompts, chosen, rejected in batch_source:

        policy_chosen, policy_rejected, ref_chosen, ref_rejected = score_everything(
            policy, reference, tokenizer, prompts, chosen, rejected
        )

        loss = compute_dpo_loss_logps(
            policy_chosen,
            policy_rejected,
            ref_chosen,
            ref_rejected,
            cfg.BETA,
        )

        reward_chosen, reward_rejected = both_rewards(
            policy_chosen,
            policy_rejected,
            ref_chosen,
            ref_rejected,
        )

        add_up(totals, loss, reward_chosen, reward_rejected, len(prompts))

    policy.train()

    mean_loss = totals["loss"] / totals["batches"]
    accuracy = totals["correct"] / totals["seen"]
    margin = totals["margin"] / totals["seen"]

    return mean_loss, accuracy, margin