"""Shared configuration and helpers for both DPO runs."""

import random
from pathlib import Path

import torch

from smalign.prompts import format_prompt

SFT_CHECKPOINT = "outputs/sft/best"

MAX_LENGTH = 512
MICRO_BATCH = 1
ACCUM_STEPS = 32
LEARNING_RATE = 1e-6
BETA = 0.1
CLIP_NORM = 1.0
EPOCHS = 1
VALIDATE_EVERY = 25
SEED = 42


def give_me_the_next_batch(all_the_pairs, how_many_at_once, should_i_shuffle, the_seed):
    
    walking_order = list(range(len(all_the_pairs)))

    if should_i_shuffle:
        random.Random(the_seed).shuffle(walking_order)

    where_we_are = 0

    while where_we_are < len(walking_order):
        this_chunk = walking_order[where_we_are : where_we_are + how_many_at_once]

        the_prompts = []
        the_good_ones = []
        the_bad_ones = []

        for index in this_chunk:
            this_row = all_the_pairs[index]

            the_prompts.append(format_prompt(this_row["prompt"]))
            the_good_ones.append(this_row["chosen"].strip())
            the_bad_ones.append(this_row["rejected"].strip())

        yield the_prompts, the_good_ones, the_bad_ones

        where_we_are = where_we_are + how_many_at_once

def put_all_on_disk(the_model, the_tokenizer, where_to_put_it, the_optimizer=None,
                           the_scaler=None, which_step=None, best_so_far=None):
    
    where_to_put_it.mkdir(parents=True, exist_ok=True)

    the_model.save_pretrained(where_to_put_it)
    the_tokenizer.save_pretrained(where_to_put_it)

    if the_optimizer is not None:
        the_training_state = {
            "optimizer": the_optimizer.state_dict(),
            "scaler": the_scaler.state_dict(),
            "step": which_step,
            "best_accuracy": best_so_far,
        }

        torch.save(the_training_state, where_to_put_it / "resume.pt")