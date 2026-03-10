"""Diagnostic script: check model's generation behavior from checkpoint.

Usage:
    python 05_optimized_dllm/debug_generate.py --ckpt /path/to/ckpt_000500.pt

Checks:
1. Logit distribution at first denoise step (is it peaked?)
2. Generation with text prompt (does context help?)
3. Training data sanity (what tokens does the model see?)
"""

import sys
import os

_real_argv = list(sys.argv)  # save real args before config import
sys.argv = ['debug', '--train']  # fake args for config import

# Adjust path for module imports
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

import argparse
import torch
import torch.nn.functional as F

from phase5 import config
from phase5.model import Model
from phase5.tokenizer import encode, decode, load_tokenizer
from phase5.generate import generate


def check_logit_distribution(model):
    """Check what logits look like when model processes all-mask input."""
    print('\n' + '=' * 60)
    print('TEST 1: Logit distribution on all-mask input (first denoise step)')
    print('=' * 60)

    block = torch.full((1, config.block_size), config.mask_token_id,
                       dtype=torch.long, device=config.device)
    logits, _ = model(block, pos_offset=0)  # (1, 32, vocab)

    # Suppress mask/pad tokens (same as generate.py)
    logits[:, :, config.mask_token_id] = -float('inf')
    logits[:, :, config.pad_token_id] = -float('inf')

    # Check softmax distribution at position 0
    probs = F.softmax(logits[0, 0].float(), dim=-1)
    top_probs, top_ids = torch.topk(probs, 20)

    tok = load_tokenizer()
    print(f'\nPosition 0 (all-mask context):')
    print(f'  Entropy: {-(probs * probs.log().nan_to_num()).sum():.2f} nats')
    print(f'  Max prob: {top_probs[0]:.4f}')
    print(f'  Top-20 tokens:')
    for p, tid in zip(top_probs, top_ids):
        decoded = tok.decode([tid.item()])
        print(f'    ID {tid.item():5d} ({decoded!r:20s}): {p:.4f}')

    # Check across multiple positions
    entropies = []
    top1_tokens = []
    for pos in range(config.block_size):
        p = F.softmax(logits[0, pos].float(), dim=-1)
        ent = -(p * p.log().nan_to_num()).sum().item()
        entropies.append(ent)
        top1_tokens.append(p.argmax().item())

    print(f'\nAcross all {config.block_size} positions:')
    print(f'  Mean entropy: {sum(entropies)/len(entropies):.2f} nats')
    print(f'  Min entropy:  {min(entropies):.2f} nats')
    print(f'  Unique top-1 tokens: {len(set(top1_tokens))} out of {config.block_size}')
    top1_decoded = [tok.decode([t]) for t in top1_tokens]
    print(f'  Top-1 tokens: {top1_decoded[:16]}...')


def check_prompted_generation(model):
    """Check if the model generates language tokens when given a prompt."""
    print('\n' + '=' * 60)
    print('TEST 2: Generation WITH text prompt')
    print('=' * 60)

    prompts = [
        'The quick brown fox',
        'In the year 2024',
        'Machine learning is',
    ]
    for prompt in prompts:
        print(f'\nPrompt: {prompt!r}')
        result = generate(model, encode, decode,
                         prompt=prompt, max_new_tokens=64,
                         temperature=0.8, top_k=5)
        print(f'Output: {result[:200]!r}')


def check_unprompted_temperatures(model):
    """Check unprompted generation at different temperatures."""
    print('\n' + '=' * 60)
    print('TEST 3: Unprompted generation at different temperatures')
    print('=' * 60)

    for temp in [0.0, 0.5, 0.8, 1.0, 1.5, 2.0]:
        result = generate(model, encode, decode,
                         max_new_tokens=64,
                         temperature=temp, top_k=50)
        print(f'\n  temp={temp}: {result[:100]!r}')


def check_training_data():
    """Check what the training data looks like after encoding."""
    print('\n' + '=' * 60)
    print('TEST 4: Training data samples')
    print('=' * 60)

    from datasets import load_dataset
    ds = load_dataset(
        'HuggingFaceFW/finepdfs_50BT-dclm_30BT-fineweb_edu_20BT-shuffled',
        split='train', streaming=True
    )

    for i, doc in enumerate(ds):
        if i >= 3:
            break
        text = doc['text'][:300]
        ids = encode(text)
        decoded = decode(ids)
        print(f'\n--- Doc {i} ---')
        print(f'  Text:    {text[:100]!r}')
        print(f'  IDs:     {ids[:20]}...')
        print(f'  Decoded: {decoded[:100]!r}')
        print(f'  Match:   {text == decoded}')


if __name__ == '__main__':
    real_parser = argparse.ArgumentParser()
    real_parser.add_argument('--ckpt', type=str, required=True)
    real_parser.add_argument('--skip-data', action='store_true',
                            help='skip slow data loading test')
    real_args, _ = real_parser.parse_known_args(_real_argv[1:])

    # Load model + checkpoint
    model = Model().to(config.device)
    print(f'Loading checkpoint: {real_args.ckpt}')
    ckpt = torch.load(real_args.ckpt, map_location=config.device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    step = ckpt.get('step', '?')
    loss = ckpt.get('loss', '?')
    print(f'Checkpoint step={step}, loss={loss}')

    model_was_eval = not model.training
    model_obj = model

    with torch.no_grad():
        model_obj.eval()
        check_logit_distribution(model_obj)
        check_prompted_generation(model_obj)
        check_unprompted_temperatures(model_obj)

    if not real_args.skip_data:
        check_training_data()

    print('\n' + '=' * 60)
    print('DIAGNOSTIC COMPLETE')
    print('=' * 60)
