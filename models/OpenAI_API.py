# Licensed under the MIT license.

import os
import os
import time
from tqdm import tqdm
import concurrent.futures
# from openai import AzureOpenAI
import httpx
from openai import OpenAI
# import anthropic


client = OpenAI(
    api_key=""
)







max_threads = 32


def load_OpenAI_model(model):
    return None, model


def generate_with_OpenAI_model(
    prompt,
    n=1,
    model_ckpt="gpt-35-turbo",
    max_tokens=256,
    temperature=0.8,
    top_k=40,
    top_p=0.95,
    stop=["\n"],
    enable_chat_template=False,
):
    old_messages = [{"role": "system", "content": "You are helpful AI assistant"}, {"role": "user", "content": prompt}]
    parameters = {
        "model": model_ckpt,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "top_p": top_p,
        "stop": stop,
        "seed": 1,
        "n": n
    }
    if enable_chat_template and "### Instruction:" in prompt and "### Response:" in prompt:
        segments = prompt.split("### Instruction:")
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
        # import ipdb; ipdb.set_trace()
    else:
        messages = old_messages
    ans_lst, timeout = "", 1
    while not ans_lst:
        try:
            time.sleep(timeout)
            completion = client.chat.completions.create(messages=messages, **parameters)
            ans_lst = []
            for idx in range(len(completion.choices)):
                ans_lst.append(completion.choices[idx].message.content)
            # import ipdb; ipdb.set_trace()
            return ans_lst

        except Exception as e:
            print(e)
        if not ans_lst:
            timeout = timeout * 2
            if timeout > 120:
                timeout = 1
            try:
                print(f"Will retry after {timeout} seconds ...")
            except:
                pass
    return ans_lst


def generate_n_with_OpenAI_model(
    prompt,
    n=1,
    model_ckpt="gpt-35-turbo",
    max_tokens=256,
    temperature=0.8,
    top_k=40,
    top_p=0.95,
    stop=["\n"],
    max_threads=3,
    disable_tqdm=True,
    enable_chat_template=False,
):
    preds = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_threads) as executor:
        futures = [
            executor.submit(generate_with_OpenAI_model, prompt, model_ckpt, max_tokens, temperature, top_k, top_p, stop, enable_chat_template)
            for _ in range(n)
        ]
        for i, future in tqdm(
            enumerate(concurrent.futures.as_completed(futures)),
            total=len(futures),
            desc="running evaluate",
            disable=disable_tqdm,
        ):
            ans = future.result()
            preds.append(ans)
    return preds
