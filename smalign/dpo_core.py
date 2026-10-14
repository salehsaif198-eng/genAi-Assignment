from smalign.sft_main import get_logits
import torch.nn.functional as F

from smalign.batching import build_sft_batch

import torch


def the_dpo_batch(tokenizer, prompts, chosen_responses, rejected_responses, max_length):
    
    to_be_selected = build_sft_batch(tokenizer, prompts, chosen_responses, max_length)
    to_be_rejected = build_sft_batch(tokenizer, prompts, rejected_responses, max_length)

    return {"chosen": to_be_selected, "rejected": to_be_rejected}


def compute_the_gaps(policy_chosen, policy_rejected, reference_chosen, reference_rejected):
    
    reference_gap = reference_chosen - reference_rejected
    policy_gap = policy_chosen - policy_rejected

    return policy_gap, reference_gap


def compute_dpo_loss_logps(policy_chosen_logps, policy_rejected_logps, reference_chosen_logps, reference_rejected_logps, beta):
    
    policy_gap, ref_gap = compute_the_gaps(policy_chosen_logps, policy_rejected_logps, reference_chosen_logps, reference_rejected_logps,)

    inside = beta * (policy_gap - ref_gap)

    return -F.logsigmoid(inside).mean()


def onto_device(batch, model):
    
    device = next(model.parameters()).device

    moved = {}
    for key in batch:
        moved[key] = batch[key].to(device)

    return moved


def grad_score(model, batch):
    on_device = onto_device(batch, model)

    output = model( input_ids=on_device["input_ids"], attention_mask=on_device["attention_mask"],)

    return get_logits(output.logits, on_device["labels"])


def score_frozen(model, batch):
    
    on_device = onto_device(batch, model)

    with torch.no_grad():
        output = model(
            input_ids=on_device["input_ids"],
            attention_mask=on_device["attention_mask"],
        )

        return get_logits(output.logits, on_device["labels"])


def join_side(batches):
    
    chosen = batches["chosen"]
    rejected = batches["rejected"]

    joined = {}
    for key in chosen:
        joined[key] = torch.cat([chosen[key], rejected[key]], dim=0)

    return joined


def score_both_sides(model, batches, frozen):
    
    joined = join_side(batches)

    if frozen:
        logps = score_frozen(model, joined)
    else:
        logps = grad_score(model, joined)

    return logps.chunk(2, dim=0)



def rewards_being_implicit(policy_logps, reference_logps, beta):
    
    reference_logps = reference_logps.to(policy_logps.device)

    return beta * (policy_logps - reference_logps).detach()



def compute_dpo_loss(policy_model, reference_model, tokenizer, prompts,
                     chosen_responses, rejected_responses, beta, max_length):
    
    batches = the_dpo_batch( tokenizer, prompts, chosen_responses, rejected_responses, max_length)

    policy_chosen, policy_rejected = score_both_sides(policy_model, batches, False)
    reference_chosen, reference_rejected = score_both_sides(reference_model, batches, True)

    
    device = policy_chosen.device

    return compute_dpo_loss_logps( policy_chosen, policy_rejected, reference_chosen.to(device), reference_rejected.to(device), beta, )