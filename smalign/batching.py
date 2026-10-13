import torch

IGNORE_INDEX = -100


def padding(ids, mask, labels, pad_id, max_length):
    
    while len(ids) < max_length:
        ids.append(pad_id)
        mask.append(0)
        labels.append(IGNORE_INDEX)

    return ids, mask, labels


def build_sft_batch(tokenizer, prompts, responses, max_length):
    
    all_ids = []
    all_masks = []
    all_labels = []

    for prompt, response in zip(prompts, responses):
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        response_ids = tokenizer.encode(response, add_special_tokens=False)

        
        if tokenizer.eos_token_id is not None:
            response_ids = response_ids + [tokenizer.eos_token_id]

        ids = (prompt_ids + response_ids)[:max_length]

        
        labels = list(ids)
        for position in range(min(len(prompt_ids), len(ids))):
            labels[position] = IGNORE_INDEX

        mask = [1] * len(ids)

        pad_id = tokenizer.pad_token_id

        padded = padding(ids, mask, labels, pad_id, max_length)

        the_ids = padded[0]
        the_mask = padded[1]
        the_labels = padded[2]

        all_ids.append(the_ids)
        all_masks.append(the_mask)
        all_labels.append(the_labels)


    return {
        "input_ids": torch.tensor(all_ids, dtype=torch.long),
        "attention_mask": torch.tensor(all_masks, dtype=torch.long),
        "labels": torch.tensor(all_labels, dtype=torch.long),
    }