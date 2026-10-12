
from __future__ import annotations

the_end = "<|im_end|>"
the_start = "<|im_start|>"

fixed_systemPrompt = ( "You are an assistant that answers the query, and gives a very short explanation as well.")


def format_prompt(instruction: str, system: str = fixed_systemPrompt) -> str:
    
    system_says = the_start + "system\n" + system + the_end + "\n"
    user_says = the_start + "user\n" + instruction.strip() + the_end + "\n"
    model_says = the_start + "assistant\n"

    return system_says + user_says + model_says

def retarget_eos(tokenizer):

    end_token_id = tokenizer.convert_tokens_to_ids(the_end)
    if end_token_id is not None and end_token_id >= 0:
        tokenizer.eos_token = the_end
    return tokenizer

def fix_token_pad(tokenizer):
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer

def configure_tokenizer(tokenizer):
    
    retarget_eos(tokenizer)
    fix_token_pad(tokenizer)
    return tokenizer
