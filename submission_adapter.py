"""Stable grading interface for the SFT and DPO implementation.

Keep the function signatures below unchanged. Each function should be a thin
wrapper around your own implementation; you do not need to place the training
code itself in this file.
"""

from __future__ import annotations


from smalign.batching import build_sft_batch as building_the_sft_batch
def build_sft_batch(tokenizer, prompts, responses, max_length):
    """Return input_ids, attention_mask, and response-only labels."""

    return building_the_sft_batch(tokenizer, prompts, responses, max_length)

from smalign.sft_main import compute_sft_loss as computing_the_sft_loss
from smalign.sft_main import get_logits


def to_model_device(batch, model):
    
    device = next(model.parameters()).device

    on_the_device = {}
    for key in batch:
        on_the_device[key] = batch[key].to(device)

    return on_the_device


def forward_pss(model, batch):
    
    output = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
    )

    return output.logits


def compute_response_logprobs(model, tokenizer, prompts, responses, max_length):
    """Return one summed response-token log-probability per input row."""
    batch = building_the_sft_batch(tokenizer, prompts, responses, max_length)
    batch = to_model_device(batch, model)

    logits = forward_pss(model, batch)

    return get_logits(logits, batch["labels"])


def compute_sft_loss(model, tokenizer, prompts, responses, max_length):
    """Return a scalar response-only SFT loss using either documented reduction."""

    return computing_the_sft_loss(model, tokenizer, prompts, responses, max_length)


from smalign.dpo_core import the_dpo_batch

def build_dpo_batch(tokenizer, prompts, chosen_responses, rejected_responses, max_length):
    

    return the_dpo_batch(tokenizer, prompts, chosen_responses, rejected_responses, max_length)

from smalign.dpo_core import compute_dpo_loss_logps
def compute_dpo_loss_from_logps(
    policy_chosen_logps,
    policy_rejected_logps,
    reference_chosen_logps,
    reference_rejected_logps,
    beta,
):
    """Return the scalar mean DPO objective from per-example log probabilities."""

    return compute_dpo_loss_logps(
    policy_chosen_logps,
    policy_rejected_logps,
    reference_chosen_logps,
    reference_rejected_logps,
    beta,
)

from smalign.dpo_core import compute_dpo_loss as computing_the_dpo_loss

def compute_dpo_loss(
    policy_model,
    reference_model,
    tokenizer,
    prompts,
    chosen_responses,
    rejected_responses,
    beta,
    max_length,
):
    """Return a scalar mean DPO loss with a frozen reference computation."""

    return computing_the_dpo_loss(
    policy_model,
    reference_model,
    tokenizer,
    prompts,
    chosen_responses,
    rejected_responses,
    beta,
    max_length,
)
