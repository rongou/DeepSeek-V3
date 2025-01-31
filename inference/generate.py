import json
import os
from argparse import ArgumentParser

import torch
import torch.distributed as dist
from safetensors.torch import load_model
from transformers import AutoTokenizer

from model import Transformer, ModelArgs


def sample(logits, temperature: float = 1.0):
    """
    Samples a token from logits using temperature scaling and multinomial sampling.
    """
    logits = logits / max(temperature, 1e-5)
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, 1).squeeze(-1)


@torch.inference_mode()
def generate(
    model: Transformer,
    prompt_tokens: torch.Tensor,
    max_new_tokens: int,
    eos_id: int,
    temperature: float = 1.0
):
    """
    Generates tokens using the given model with proper positional embedding updates.
    """
    batch_size, prompt_len = prompt_tokens.shape
    total_len = min(model.max_seq_len, max_new_tokens + prompt_len)
    tokens = torch.full((batch_size, total_len), -1, dtype=torch.long, device="cuda")
    tokens[:, :prompt_len] = prompt_tokens

    finished = torch.zeros(batch_size, dtype=torch.bool, device="cuda")

    for cur_pos in range(prompt_len, total_len):
        logits = model.forward(tokens[:, :cur_pos], 0)

        # Ensure logits have the correct shape
        if logits.dim() == 2:
            logits = logits.unsqueeze(1)  # Add missing sequence length dimension

        next_token = sample(logits[:, -1], temperature) if temperature > 0 else logits[:, -1].argmax(dim=-1)

        tokens[:, cur_pos] = next_token
        finished |= next_token == eos_id

        if finished.all():
            break

    completion_tokens = []
    for i in range(batch_size):
        toks = tokens[i, prompt_len:cur_pos + 1].tolist()
        if eos_id in toks:
            toks = toks[:toks.index(eos_id)]
        completion_tokens.append(toks)

    return completion_tokens


def main(
    ckpt_path: str,
    config: str,
    input_file: str = "",
    interactive: bool = True,
    max_new_tokens: int = 100,
    temperature: float = 1.0,
) -> None:
    """
    Load the model and perform interactive or batch inference.
    """
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    rank = int(os.getenv("RANK", "0"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    if world_size > 1:
        dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank)
    torch.set_default_dtype(torch.bfloat16)
    torch.manual_seed(965)

    with open(config) as f:
        args = ModelArgs(**json.load(f))

    model = Transformer(args).to("cuda")
    tokenizer = AutoTokenizer.from_pretrained(ckpt_path)
    load_model(model, os.path.join(ckpt_path, f"model{rank}-mp{world_size}.safetensors"))

    if interactive:
        messages = []
        while True:
            if world_size > 1:
                if rank == 0:
                    prompt = input(">>> ")
                    objects = [prompt]
                    dist.broadcast_object_list(objects, src=0)
                else:
                    objects = [None]
                    dist.broadcast_object_list(objects, src=0)
                    prompt = objects[0]
            else:
                prompt = input(">>> ")

            if prompt == "/exit":
                break
            if prompt == "/clear":
                messages.clear()
                continue

            messages.append({"role": "user", "content": prompt})
            prompt_tokens = tokenizer([prompt], return_tensors="pt", padding=True)["input_ids"].to("cuda")
            completion_tokens = generate(model, prompt_tokens, max_new_tokens, tokenizer.eos_token_id, temperature)
            completion = tokenizer.batch_decode(completion_tokens, skip_special_tokens=True)[0]

            if rank == 0:  # Only print on rank 0 to avoid duplicated output
                print(completion)

            messages.append({"role": "assistant", "content": completion})
    else:
        with open(input_file) as f:
            prompts = [line.strip() for line in f.readlines()]
        prompt_tokens = tokenizer(prompts, return_tensors="pt", padding=True)["input_ids"].to("cuda")
        completion_tokens = generate(model, prompt_tokens, max_new_tokens, tokenizer.eos_token_id, temperature)
        completions = tokenizer.batch_decode(completion_tokens, skip_special_tokens=True)
        if rank == 0:
            for prompt, completion in zip(prompts, completions):
                print(f"Prompt: {prompt}\nCompletion: {completion}\n")

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--ckpt-path", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--input-file", type=str, default="")
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.2)
    args = parser.parse_args()
    main(args.ckpt_path, args.config, args.input_file, args.interactive, args.max_new_tokens, args.temperature)
