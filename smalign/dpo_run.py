import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import smalign.dpo_setup as cfg
from smalign.prompts import configure_tokenizer


def which_devices():
    
    count = torch.cuda.device_count()

    if count >= 2:
        return "cuda:0", "cuda:1"

    if count == 1:
        return "cuda:0", "cuda:0"

    return "cpu", "cpu"


def wake_the_tokenizer(checkpoint):
    
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    configure_tokenizer(tokenizer)

    return tokenizer


def wake_the_policy(checkpoint, device):
    
    model = AutoModelForCausalLM.from_pretrained(checkpoint, dtype=torch.float32)
    model.to(device)
    model.train()

    return model


def freeze_the_reference(checkpoint, device):
    
    model = AutoModelForCausalLM.from_pretrained(checkpoint, dtype=torch.float16)
    model.to(device)
    model.eval()

    for parameter in model.parameters():
        parameter.requires_grad_(False)

    return model


def set_up_the_optimizer(policy):
    
    optimizer = torch.optim.AdamW(policy.parameters(), lr=cfg.LEARNING_RATE)
    scaler = torch.amp.GradScaler()

    return optimizer, scaler


def get_ready(checkpoint):
    
    policy_device, reference_device = which_devices()
    print("policy on", policy_device, " reference on", reference_device)

    tokenizer = wake_the_tokenizer(checkpoint)

    policy = wake_the_policy(checkpoint, policy_device)
    reference = freeze_the_reference(checkpoint, reference_device)

    optimizer, scaler = set_up_the_optimizer(policy)

    return policy, reference, tokenizer, optimizer, scaler