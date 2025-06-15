# Licensed under the MIT license.

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import numpy as np
import math


def load_vLLM_model(model_ckpt, seed, tensor_parallel_size=1, half_precision=False, max_num_seqs=256):
    tokenizer = AutoTokenizer.from_pretrained(model_ckpt)

    if half_precision:
        llm = LLM(
            model=model_ckpt,
            dtype="float",
            tensor_parallel_size=tensor_parallel_size,
            seed=seed,
            trust_remote_code=True,
            max_num_seqs=max_num_seqs,
            swap_space=16,
            max_model_len=16000,
        )
    else:
        llm = LLM(
            model=model_ckpt,
            tensor_parallel_size=tensor_parallel_size,
            seed=seed,
            trust_remote_code=True,
            max_num_seqs=max_num_seqs,
            swap_space=16,
            max_model_len=10000,
            gpu_memory_utilization=0.95
        )

    return tokenizer, llm


def generate_with_vLLM_model(
    model,
    input,
    temperature=0.8,
    top_p=0.95,
    top_k=40,
    repetition_penalty=1.1,
    n=1,
    max_tokens=256,
    logprobs=1,
    stop=[],
    tokenizer=None,
    enable_chat_template=False,
):
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
        n=n,
        logprobs=logprobs,
        max_tokens=max_tokens,
        stop=stop,
    )
    if enable_chat_template and "### Instruction:" in input and "### Response:" in input:
        segments = input.split("### Instruction:")
        messages = [
            {"role": "system",
             "content": segments[0].strip()},
        ]
        for idx in range(1, len(segments)):
            examples = segments[idx].split("### Response:")
            messages.append({
                "role": "user",
                "content": examples[0].strip()
            })
            if len(examples) > 0 and len(examples[1].strip()) > 0:
                messages.append({
                    "role": "assistant",
                    "content": examples[1].strip()
                })
        chat_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        # import ipdb; ipdb.set_trace()
        output = model.generate(chat_prompt, sampling_params, use_tqdm=False)
    else:
        output = model.generate(input, sampling_params, use_tqdm=False)
    return output


if __name__ == "__main__":
    model_ckpt = "mistralai/Mistral-7B-v0.1"
    tokenizer, model = load_vLLM_model(model_ckpt, seed=42, tensor_parallel_size=1, half_precision=False)
    input = "What is the meaning of life?"
    output = generate_with_vLLM_model(model, input)
    breakpoint()
    print(output[0].outputs[0].text)
