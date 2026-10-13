import torch

from smalign.batching import IGNORE_INDEX, build_sft_batch

def shift(the_logits, labels):

    shifted_logits = the_logits[:, :-1, :]
    shifted_labels = labels[:, 1:]

    return shifted_logits, shifted_labels

def thy_safe_labels(labels, loss_mask):
    
    the_safe_labels = labels.clone()
    the_safe_labels[~loss_mask] = 0
    return the_safe_labels


def to_log(logits):
    return torch.log_softmax(logits.float(), dim=-1)


def get_logits(logits, labels):

    logits, labels = shift(logits, labels)

    loss_mask = labels != IGNORE_INDEX
    the_safe_labels = thy_safe_labels(labels, loss_mask)


    the_log_probs = to_log(logits)
    index = the_safe_labels.unsqueeze(2)
    gathered = torch.gather(the_log_probs, 2, index)
    token_logps = gathered.squeeze(2)


    return (token_logps * loss_mask).sum(-1)



def compute_sft_loss(model, tokenizer, prompts, responses, max_length):
    

    get_batch = build_sft_batch(tokenizer, prompts, responses, max_length)
    
    device = next(model.parameters()).device

    input_ids = get_batch["input_ids"]
    attention_mask = get_batch["attention_mask"]
    labels = get_batch["labels"]

    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    labels = labels.to(device)

    output = model(input_ids=input_ids, attention_mask=attention_mask)
    logps = get_logits(output.logits, labels)


    shifted = labels[:, 1:]
    is_response = shifted != IGNORE_INDEX
    n_tokens = is_response.sum()
    n_tokens = n_tokens.clamp(min=1)

    nll = -logps.sum()
    the_loss = nll / n_tokens

    return the_loss