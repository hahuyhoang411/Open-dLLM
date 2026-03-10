"""
core_eval.py — DCLM CORE evaluation: prompt rendering, batching, forward, scoring.

This module handles the full "inner loop" of DCLM CORE benchmark evaluation:
  1. Rendering prompts from task data using jinja2 templates
  2. Tokenizing prompts and finding common prefix/suffix boundaries
  3. Batching token sequences for model forward passes
  4. AR and dLLM (masked diffusion) forward passes with per-token loss
  5. Scoring: multiple choice / schema via loss comparison, LM via exact match

The DCLM CORE benchmark (https://arxiv.org/abs/2406.11794) evaluates language
models across 22 tasks in three categories:
  - Multiple choice (e.g., HellaSwag, ARC): same context, different continuations
  - Schema / Winograd-style (e.g., WinoGrande): different contexts, same continuation
  - Language modeling (LAMBADA): predict the final word given context

Ported from nanochat (github.com/karpathy/nanochat), adapted for our simpler
tokenizer interface: tokenize_fn(text) -> list[int], no BOS token handling.

Architecture
============

    Task data (dict)
         |
         v
    render_prompts_*()          # jinja2 templates -> list of prompt strings
         |
         v
    batch_sequences_*()         # tokenize + find answer boundaries
         |                        tokenize_fn(text) -> list[int]
         v
    (tokens, start_indices,     # ready for model forward pass
     end_indices)
         |
         v
    forward_model_ar()          # standard AR: shift-by-1 CE
    forward_model_dllm()        # masked diffusion: MC likelihood estimation
         |
         v
    evaluate_example()          # score one item -> bool
    evaluate_task()             # score all items -> accuracy
"""

import random

from jinja2 import Template
import torch
import torch.nn.functional as F


# =============================================================================
# Prompt Rendering — jinja2 templates for each task type
# =============================================================================
#
# Each render function takes a single evaluation item (dict), a continuation
# delimiter (string separating context from answer), and optional few-shot
# examples. Returns a list of prompt strings ready for tokenization.
#
# The templates follow DCLM conventions: few-shot examples are prepended as
# (query + correct answer) pairs separated by newlines.

def render_prompts_mc(item, continuation_delimiter, fewshot_examples=None):
    """Render prompts for a multiple choice task (e.g., HellaSwag, ARC, PIQA).

    Each choice gets its own prompt: shared context + one candidate answer.
    The model scores each prompt and picks the lowest-loss option.

    Returns: list of prompt strings, one per choice.
    """
    template_str = """\
{%- for example in fewshot_examples -%}
{{ example.query }}{{ continuation_delimiter }}{{ example.choices[example.gold] }}

{% endfor -%}
{{ item.query }}{{ continuation_delimiter }}{{ choice }}"""
    template = Template(template_str)
    fewshot_examples = fewshot_examples or []
    context = {
        "fewshot_examples": fewshot_examples,
        "continuation_delimiter": continuation_delimiter,
        "item": item,
    }
    return [template.render(choice=choice, **context) for choice in item["choices"]]


def render_prompts_schema(item, continuation_delimiter, fewshot_examples=None):
    """Render prompts for a schema/Winograd-style task (e.g., WinoGrande).

    Each context option gets its own prompt: different context + shared continuation.
    The model scores each prompt and picks the lowest-loss option.

    Returns: list of prompt strings, one per context option.
    """
    template_str = """\
{%- for example in fewshot_examples -%}
{{ example.context_options[example.gold] }}{{ continuation_delimiter }}{{ example.continuation }}

{% endfor -%}
{{ context }}{{ continuation_delimiter }}{{ item.continuation }}"""
    template = Template(template_str)
    fewshot_examples = fewshot_examples or []
    context = {
        "fewshot_examples": fewshot_examples,
        "continuation_delimiter": continuation_delimiter,
        "item": item,
    }
    return [
        template.render(context=context_option, **context)
        for context_option in item["context_options"]
    ]


def render_prompts_lm(item, continuation_delimiter, fewshot_examples=None):
    """Render prompts for a language modeling task (LAMBADA).

    Returns TWO prompts: [context_only, context+continuation].
    The model checks if it predicts the continuation tokens correctly.

    Context whitespace is trimmed to prevent tokenizer artifacts — trailing
    spaces in context can merge with the first continuation token, making it
    impossible to cleanly detect the continuation boundary in token space.

    Returns: [prompt_without_continuation, prompt_with_continuation]
    """
    template_str = """\
{%- for example in fewshot_examples -%}
{{ example.context | trim }}{{ continuation_delimiter }}{{ example.continuation }}

{% endfor -%}
{{ item.context | trim }}{{ continuation_delimiter }}{% if include_continuation %}{{ item.continuation }}{% endif %}"""
    template = Template(template_str)
    fewshot_examples = fewshot_examples or []
    context = {
        "fewshot_examples": fewshot_examples,
        "continuation_delimiter": continuation_delimiter,
        "item": item,
    }
    prompt_without = template.render(include_continuation=False, **context)
    prompt_with = template.render(include_continuation=True, **context)
    # Strip trailing whitespace from the "without" prompt — otherwise tokenizers
    # may absorb whitespace into the first continuation token, breaking the
    # clean prefix alignment needed to detect continuation boundaries.
    prompt_without = prompt_without.strip()
    return [prompt_without, prompt_with]


# =============================================================================
# Sequence Utilities
# =============================================================================

def find_common_length(token_sequences, direction="left"):
    """Find the length of the common prefix (left) or suffix (right) across
    multiple token sequences.

    Used to locate answer boundaries:
      - MC tasks: common prefix = shared context, divergence = answer start
      - Schema tasks: common suffix = shared continuation

    Args:
        token_sequences: list of list[int], tokenized prompts
        direction: 'left' for common prefix, 'right' for common suffix

    Returns: int, number of matching positions from the given direction
    """
    min_len = min(len(seq) for seq in token_sequences)
    if direction == "left":
        indices = range(min_len)
    elif direction == "right":
        indices = range(-1, -min_len - 1, -1)
    else:
        raise ValueError(f"direction must be 'left' or 'right', got '{direction}'")

    for i, idx in enumerate(indices):
        token = token_sequences[0][idx]
        if not all(seq[idx] == token for seq in token_sequences):
            return i
    return min_len


def stack_sequences(tokens, pad_token_id):
    """Pad a list of variable-length token sequences and stack into a tensor.

    Right-pads shorter sequences with pad_token_id so all sequences have
    the same length, then stacks into a (batch_size, max_seq_len) tensor.

    Args:
        tokens: list of list[int], variable-length token sequences
        pad_token_id: int, token ID used for padding

    Returns: torch.LongTensor of shape (batch_size, max_seq_len)
    """
    bsz = len(tokens)
    seq_len = max(len(x) for x in tokens)
    input_ids = torch.full((bsz, seq_len), pad_token_id, dtype=torch.long)
    for i, x in enumerate(tokens):
        input_ids[i, : len(x)] = torch.tensor(x, dtype=torch.long)
    return input_ids


# =============================================================================
# Batch Preparation — tokenize prompts and locate answer boundaries
# =============================================================================
#
# Each batch_sequences_* function:
#   1. Tokenizes all prompt strings using the provided tokenize_fn
#   2. Finds where the "answer" region starts and ends in token space
#   3. Returns (tokens, start_indices, end_indices)
#
# The tokenize_fn interface is simple: tokenize_fn(text) -> list[int].
# No BOS token prepending — our dLLM doesn't use BOS. For HF models (GPT-2),
# the caller wraps the tokenizer to prepend BOS if needed.

def batch_sequences_mc(tokenize_fn, prompts):
    """Batch tokenized prompts for a multiple choice task.

    In MC tasks, all prompts share the same context prefix and differ only in
    the answer continuation. We find the common prefix length to locate where
    the answer tokens begin.

    Returns:
        tokens: list of list[int], tokenized prompts
        start_indices: list of int, answer start position (same for all)
        end_indices: list of int, sequence end position (varies per choice)
    """
    tokens = [tokenize_fn(p) for p in prompts]
    # All prompts share a common prefix (the context); answers diverge after it
    answer_start_idx = find_common_length(tokens, direction="left")
    start_indices = [answer_start_idx] * len(prompts)
    end_indices = [len(t) for t in tokens]
    return tokens, start_indices, end_indices


def batch_sequences_schema(tokenize_fn, prompts):
    """Batch tokenized prompts for a schema/Winograd-style task.

    In schema tasks, prompts have different context prefixes but share the same
    continuation suffix. We find the common suffix length to locate where the
    scored continuation begins.

    Returns:
        tokens: list of list[int], tokenized prompts
        start_indices: list of int, continuation start per prompt (varies)
        end_indices: list of int, sequence end per prompt (varies)
    """
    tokens = [tokenize_fn(p) for p in prompts]
    # All prompts share a common suffix (the continuation); contexts differ
    suffix_length = find_common_length(tokens, direction="right")
    end_indices = [len(t) for t in tokens]
    start_indices = [ei - suffix_length for ei in end_indices]
    return tokens, start_indices, end_indices


def batch_sequences_lm(tokenize_fn, prompts):
    """Batch tokenized prompts for a language modeling task (LAMBADA).

    Takes exactly two prompts: [without_continuation, with_continuation].
    The continuation region is everything after the "without" prompt ends.

    Returns:
        tokens: list containing one sequence (the with-continuation prompt)
        start_indices: [len(tokens_without)] — continuation starts here
        end_indices: [len(tokens_with)] — continuation ends here
    """
    assert len(prompts) == 2, "LM task expects [prompt_without, prompt_with]"
    tokens_without = tokenize_fn(prompts[0])
    tokens_with = tokenize_fn(prompts[1])
    start_idx = len(tokens_without)
    end_idx = len(tokens_with)
    assert start_idx < end_idx, (
        f"prompt_without ({start_idx} tokens) should be shorter than "
        f"prompt_with ({end_idx} tokens)"
    )
    assert tokens_without == tokens_with[:start_idx], (
        "prompt_without must be a token-level prefix of prompt_with"
    )
    # Only the with-continuation prompt is needed for the model forward pass
    return [tokens_with], [start_idx], [end_idx]


# =============================================================================
# Model Forward Passes
# =============================================================================
#
# Two forward strategies:
#   - AR: standard autoregressive, logits[:, t] predicts token at t+1
#   - dLLM: Monte Carlo likelihood estimation via random masking of answer tokens
#
# Both return (losses, predictions) of shape (B, T), so downstream scoring
# code doesn't need to know which model type produced them.

@torch.no_grad()
def forward_model_ar(model_fn, input_ids):
    """Autoregressive forward pass with per-token cross-entropy.

    Args:
        model_fn: callable, input_ids (B,T) -> logits (B,T,V)
        input_ids: LongTensor (B, T)

    Returns:
        losses: FloatTensor (B, T) — CE at each position (last col = nan)
        predictions: LongTensor (B, T) — argmax prediction at each position
    """
    B, T = input_ids.shape
    logits = model_fn(input_ids)  # (B, T, V)
    V = logits.shape[-1]

    # AR shift-by-1: logits[:, t, :] predicts input_ids[:, t+1]
    target_ids = torch.roll(input_ids, shifts=-1, dims=1)
    losses = F.cross_entropy(
        logits.reshape(-1, V), target_ids.reshape(-1), reduction="none"
    ).reshape(B, T)
    # Last position has no valid next token
    losses[:, -1] = float("nan")

    predictions = logits.argmax(dim=-1)  # (B, T)
    return losses, predictions


@torch.no_grad()
def forward_model_dllm(model_fn, input_ids, start_idxs, end_idxs,
                        mask_token_id, mc_num=128, mc_batch_size=64):
    """Batched Monte Carlo likelihood estimation for masked diffusion LMs.

    Uses replica batching with antithetic masking (Dream/BD3-LMs pattern):
    replicate each sequence mc_batch_size times, apply different masks per
    replica in one batched forward pass. Linear schedule (mask_prob=t) with
    ELBO importance weight 1/t.

    Args:
        model_fn: callable, input_ids (B,T) -> logits (B,T,V)
        input_ids: LongTensor (B, T) — B choices, padded to same length
        start_idxs: list[int], answer region start per choice
        end_idxs: list[int], answer region end per choice
        mask_token_id: int
        mc_num: int, total MC samples per choice (default 128)
        mc_batch_size: int, replicas per forward pass (default 64, fits A10G)

    Returns:
        losses: FloatTensor (B, T) — ELBO loss spread over answer region, nan elsewhere
        predictions: LongTensor (B, T) — argmax with fully-masked answer region
    """
    B, T = input_ids.shape
    device = input_ids.device
    losses = torch.full((B, T), float("nan"), device=device)
    predictions = torch.zeros(B, T, dtype=torch.long, device=device)

    for i in range(B):
        si = int(start_idxs[i])
        ei = int(end_idxs[i])
        answer_len = ei - si

        # --- Batched MC likelihood estimation ---
        u0 = torch.rand(1, device=device)
        weighted_loss_sum = 0.0
        total_samples = 0

        for batch_start in range(0, mc_num, mc_batch_size):
            cur_batch = min(mc_batch_size, mc_num - batch_start)

            # Antithetic t: uniform coverage of (0, 1]
            indices = torch.arange(batch_start, batch_start + cur_batch,
                                   device=device, dtype=torch.float32)
            t = (u0 + indices / mc_num) % 1  # (cur_batch,)
            t = t.clamp(min=0.01)  # avoid 1/t explosion

            # Linear schedule: mask_prob = t
            # Mask answer tokens per replica
            answer_rand = torch.rand(cur_batch, answer_len, device=device)
            mask_flags = answer_rand < t.unsqueeze(1)  # (cur_batch, answer_len)

            # Force at least 1 masked token per replica
            no_mask = ~mask_flags.any(dim=1)
            if no_mask.any():
                forced = torch.randint(0, answer_len, (int(no_mask.sum()),),
                                       device=device)
                mask_flags[no_mask, forced] = True

            # Build masked inputs: replicate and apply masks
            seq = input_ids[i : i + 1].expand(cur_batch, -1).clone()
            seq[:, si:ei] = torch.where(mask_flags, mask_token_id,
                                        seq[:, si:ei])

            # Batched forward pass — the critical speedup
            logits = model_fn(seq)  # (cur_batch, T, V)

            # CE on answer region, zero out unmasked positions
            V = logits.shape[-1]
            logits_answer = logits[:, si:ei, :].reshape(-1, V)
            targets = input_ids[i, si:ei].unsqueeze(0).expand(cur_batch, -1)
            ce = F.cross_entropy(
                logits_answer, targets.reshape(-1), reduction="none"
            ).reshape(cur_batch, answer_len)
            del logits, logits_answer  # free VRAM

            # Per-replica: mean CE over masked positions, weighted by 1/t
            ce_masked = ce * mask_flags.float()
            num_masked = mask_flags.sum(dim=1).clamp(min=1)
            per_replica_mean = ce_masked.sum(dim=1) / num_masked  # (cur_batch,)
            weighted = per_replica_mean / t  # importance weight 1/t
            weighted_loss_sum += weighted.sum().item()
            total_samples += cur_batch

        mean_loss = weighted_loss_sum / max(total_samples, 1)
        losses[i, si:ei] = mean_loss / answer_len

        # --- Prediction: fully-masked answer region, single forward ---
        pred_input = input_ids[i : i + 1].clone()
        pred_input[0, si:ei] = mask_token_id
        pred_logits = model_fn(pred_input)
        predictions[i] = pred_logits[0].argmax(dim=-1)
        del pred_logits

    return losses, predictions


# =============================================================================
# Example-Level and Task-Level Evaluation
# =============================================================================

def evaluate_example(idx, model_fn, tokenize_fn, data, device, task_meta,
                     mode="ar", mask_token_id=None, mc_num=128, max_seq_len=None,
                     pad_token_id=None, mc_batch_size=64):
    """Evaluate a single DCLM CORE example and return whether prediction is correct.

    Handles the full pipeline: fewshot sampling -> prompt rendering -> tokenization
    -> batching -> forward pass -> scoring.

    Args:
        idx: int, index into data
        model_fn: callable, see forward_model_ar / forward_model_dllm
        tokenize_fn: callable, text -> list[int]
        data: list of dicts, the task dataset
        device: torch device
        task_meta: dict with keys 'task_type', 'continuation_delimiter', 'num_fewshot'
        mode: 'ar' or 'dllm'
        mask_token_id: int, required when mode='dllm'
        mc_num: int, MC samples for dllm mode
        max_seq_len: int or None, truncate from the left if sequences exceed this

    Returns: bool, True if the example was answered correctly
    """
    item = data[idx]
    task_type = task_meta["task_type"]
    continuation_delimiter = task_meta["continuation_delimiter"]
    num_fewshot = task_meta.get("num_fewshot", 0)

    # --- Fewshot examples ---
    fewshot_examples = None
    if num_fewshot > 0:
        rng = random.Random(1234 + idx)
        candidates = [i for i in range(len(data)) if i != idx]
        fewshot_indices = rng.sample(candidates, min(num_fewshot, len(candidates)))
        fewshot_examples = [data[i] for i in fewshot_indices]

    # --- Render prompts and batch ---
    if task_type == "multiple_choice":
        prompts = render_prompts_mc(item, continuation_delimiter, fewshot_examples)
        tokens, start_idxs, end_idxs = batch_sequences_mc(tokenize_fn, prompts)
    elif task_type == "schema":
        prompts = render_prompts_schema(item, continuation_delimiter, fewshot_examples)
        tokens, start_idxs, end_idxs = batch_sequences_schema(tokenize_fn, prompts)
    elif task_type == "language_modeling":
        prompts = render_prompts_lm(item, continuation_delimiter, fewshot_examples)
        tokens, start_idxs, end_idxs = batch_sequences_lm(tokenize_fn, prompts)
    else:
        raise ValueError(f"Unknown task_type: {task_type}")

    # --- Left-crop if max_seq_len exceeded ---
    if max_seq_len is not None:
        for j in range(len(tokens)):
            seq_len = len(tokens[j])
            if seq_len > max_seq_len:
                crop = seq_len - max_seq_len
                tokens[j] = tokens[j][crop:]
                start_idxs[j] = max(start_idxs[j] - crop, 0)
                end_idxs[j] = max(end_idxs[j] - crop, start_idxs[j] + 1)

    # --- Stack and move to device ---
    if pad_token_id is None:
        pad_token_id = mask_token_id if mask_token_id is not None else 0
    input_ids = stack_sequences(tokens, pad_token_id).to(device)

    # --- Forward pass ---
    if mode == "ar":
        losses, predictions = forward_model_ar(model_fn, input_ids)
    elif mode == "dllm":
        assert mask_token_id is not None, "mask_token_id required for dllm mode"
        losses, predictions = forward_model_dllm(
            model_fn, input_ids, start_idxs, end_idxs, mask_token_id,
            mc_num, mc_batch_size,
        )
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # --- Scoring ---
    if task_type in ("multiple_choice", "schema"):
        # Pick the choice with the lowest mean loss in its answer region
        mean_losses = []
        for j in range(len(tokens)):
            si, ei = start_idxs[j], end_idxs[j]
            if mode == "ar":
                # AR: logits[:, t] predicts t+1, so loss at position t covers
                # the prediction of token t+1. For the answer region [si, ei),
                # we need losses at positions [si-1, ei-1) to score tokens si..ei-1.
                # But si-1 is the last context token predicting the first answer token.
                region_losses = losses[j, max(si - 1, 0) : ei - 1]
            else:
                region_losses = losses[j, si:ei]
            # Filter out nans (padding or boundary)
            valid = region_losses[~torch.isnan(region_losses)]
            mean_losses.append(valid.mean().item() if len(valid) > 0 else float("inf"))
        pred_idx = int(torch.tensor(mean_losses).argmin().item())
        return pred_idx == item["gold"]

    else:  # language_modeling
        # Check if argmax predictions match actual continuation tokens
        si, ei = start_idxs[0], end_idxs[0]
        if mode == "ar":
            # AR predictions[0, t] = argmax of logits[0, t] which predicts token t+1
            # So predictions[0, si-1:ei-1] should match input_ids[0, si:ei]
            pred_tokens = predictions[0, max(si - 1, 0) : ei - 1]
            actual_tokens = input_ids[0, si:ei]
        else:
            # dLLM predictions at masked positions directly predict those positions
            pred_tokens = predictions[0, si:ei]
            actual_tokens = input_ids[0, si:ei]
        return bool((pred_tokens == actual_tokens).all().item())


def evaluate_task(model_fn, tokenize_fn, data, device, task_meta,
                  mode="ar", mask_token_id=None, mc_num=128, max_seq_len=None,
                  pad_token_id=None, mc_batch_size=64):
    """Evaluate all examples in a DCLM CORE task and return mean accuracy."""
    if not data:
        return 0.0
    correct = 0
    for idx in range(len(data)):
        if evaluate_example(
            idx, model_fn, tokenize_fn, data, device, task_meta,
            mode, mask_token_id, mc_num, max_seq_len, pad_token_id,
            mc_batch_size,
        ):
            correct += 1
    return correct / len(data)
