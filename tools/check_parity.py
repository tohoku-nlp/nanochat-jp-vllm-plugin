#!/usr/bin/env python3
# Copyright 2026 The nanochat-jp authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Parity check: HF transformers (trust_remote_code) vs. vLLM + this plugin.

Runs the same prompts through both backends with greedy decoding and compares
(a) the generated token ids and (b) the first decoding step's top-1 logprob.
Run in an environment with a CUDA-capable GPU, CUDA-enabled PyTorch,
vLLM, transformers, and this plugin installed.

Example:
    python tools/check_parity.py \\
        --model_path /path/to/nanochat-jp-checkpoint \\
        --max_new_tokens 32
"""

import argparse
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from vllm import LLM, SamplingParams

DEFAULT_PROMPTS = [
    "The quick brown fox",
    "こんにちは、元気ですか?",
    "def fibonacci(n):",
    "Once upon a time in a distant galaxy,",
]


def load_prompts(args: argparse.Namespace) -> list[str]:
    if args.prompts_file is not None:
        with open(args.prompts_file, encoding="utf-8") as f:
            prompts = [line.rstrip("\n") for line in f if line.strip()]
        if not prompts:
            raise ValueError(f"No prompts found in {args.prompts_file!r}")
        return prompts
    return DEFAULT_PROMPTS


def run_hf(
    model_path: str, prompts: list[str], max_new_tokens: int, dtype: torch.dtype, device: str
) -> tuple[list[list[int]], list[float]]:
    """Greedy-decode each prompt with HF transformers, trust_remote_code=True.

    Returns (generated_token_ids_per_prompt, first_step_top1_logprob_per_prompt).
    """
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    # transformers v5.x deprecates/removes `torch_dtype` in favor of `dtype`.
    model = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True, dtype=dtype
    )
    model.to(device)
    model.eval()

    all_new_ids: list[list[int]] = []
    all_first_logprobs: list[float] = []
    with torch.no_grad():
        for prompt in prompts:
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            prompt_len = inputs["input_ids"].shape[1]
            output = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=1,
                output_scores=True,
                return_dict_in_generate=True,
            )
            new_ids = output.sequences[0, prompt_len:].tolist()
            all_new_ids.append(new_ids)

            first_step_logits = output.scores[0][0].float()
            first_step_logprobs = torch.log_softmax(first_step_logits, dim=-1)
            all_first_logprobs.append(first_step_logprobs.max().item())

    del model
    return all_new_ids, all_first_logprobs


def run_vllm(
    model_path: str,
    prompts: list[str],
    max_new_tokens: int,
    dtype: str,
    tensor_parallel_size: int,
    enforce_eager: bool,
) -> tuple[list[list[int]], list[float]]:
    """Greedy-decode each prompt with vLLM (this plugin registers the arch).

    Returns (generated_token_ids_per_prompt, first_step_top1_logprob_per_prompt).
    """
    llm = LLM(
        model=model_path,
        trust_remote_code=True,
        dtype=dtype,
        tensor_parallel_size=tensor_parallel_size,
        enforce_eager=enforce_eager,
    )
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=max_new_tokens,
        logprobs=1,
    )
    outputs = llm.generate(prompts, sampling_params)

    all_new_ids: list[list[int]] = []
    all_first_logprobs: list[float] = []
    for output in outputs:
        completion = output.outputs[0]
        all_new_ids.append(list(completion.token_ids))
        first_step_logprob_dict = completion.logprobs[0]
        top_logprob = max(lp.logprob for lp in first_step_logprob_dict.values())
        all_first_logprobs.append(top_logprob)

    return all_new_ids, all_first_logprobs


def main(args: argparse.Namespace) -> int:
    prompts = load_prompts(args)
    torch_dtype = getattr(torch, args.dtype)

    print(f"Running HF reference ({args.model_path}) on {len(prompts)} prompt(s)...")
    hf_ids, hf_logprobs = run_hf(
        args.model_path, prompts, args.max_new_tokens, torch_dtype, args.hf_device
    )

    print(f"Running vLLM + nanochat-jp-vllm plugin on {len(prompts)} prompt(s)...")
    vllm_ids, vllm_logprobs = run_vllm(
        args.model_path,
        prompts,
        args.max_new_tokens,
        args.dtype,
        args.tensor_parallel_size,
        args.enforce_eager,
    )

    all_passed = True
    print("\n" + "=" * 72)
    for i, prompt in enumerate(prompts):
        ids_match = hf_ids[i] == vllm_ids[i]
        logprob_diff = abs(hf_logprobs[i] - vllm_logprobs[i])
        logprob_match = logprob_diff <= args.logprob_atol

        status = "PASS" if (ids_match and logprob_match) else "FAIL"
        all_passed &= ids_match and logprob_match

        print(f"[{status}] prompt {i}: {prompt!r}")
        if not ids_match:
            print(f"    token id mismatch:")
            print(f"      HF:   {hf_ids[i]}")
            print(f"      vLLM: {vllm_ids[i]}")
        if not logprob_match:
            print(
                f"    first-step top-1 logprob mismatch: "
                f"HF={hf_logprobs[i]:.6f} vLLM={vllm_logprobs[i]:.6f} "
                f"diff={logprob_diff:.6f} (atol={args.logprob_atol})"
            )
    print("=" * 72)
    print("OVERALL: " + ("PASS" if all_passed else "FAIL"))

    return 0 if all_passed else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Compare HF transformers (trust_remote_code) and vLLM "
            "(nanochat-jp-vllm plugin) greedy generations for parity."
        )
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to (or HF hub id of) the nanochat-jp checkpoint.",
    )
    parser.add_argument(
        "--prompts_file",
        type=str,
        default=None,
        help="Optional path to a newline-separated prompts file. Defaults to a "
        "small built-in prompt set (English/Japanese/code/prose).",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=32,
        help="Number of tokens to greedily decode per prompt.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float16", "bfloat16", "float32"],
        help="Activation dtype for both backends (must match for a fair comparison).",
    )
    parser.add_argument(
        "--hf_device",
        type=str,
        default="cuda",
        help="Device for the HF reference model (e.g. 'cuda', 'cuda:0').",
    )
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=1,
        help="vLLM tensor parallel size. This plugin only supports 1.",
    )
    parser.add_argument(
        "--enforce_eager",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Disable CUDA graphs in vLLM. Default True: eager execution is "
        "the numerical reference baseline for parity checking. Pass "
        "--no-enforce_eager to also exercise the compiled/CUDA-graph path.",
    )
    parser.add_argument(
        "--logprob_atol",
        type=float,
        default=0.05,
        help="Absolute tolerance for the first-step top-1 logprob comparison. "
        "A nonzero tolerance is expected: HF upcasts logits to fp32 before "
        "the softcap, while vLLM's LogitsProcessor may not.",
    )
    parsed_args = parser.parse_args()
    sys.exit(main(parsed_args))
