"""
core_eval.py — DCLM CORE evaluation: prompt rendering and sequence batching.

This module handles the "inner loop" of DCLM CORE benchmark evaluation:
  1. Rendering prompts from task data using jinja2 templates
  2. Tokenizing prompts and finding common prefix/suffix boundaries
  3. Batching token sequences for model forward passes

The DCLM CORE benchmark (https://arxiv.org/abs/2406.11794) evaluates language
models across 22 tasks in three categories:
  - Multiple choice (e.g., HellaSwag, ARC): same context, different continuations
  - Schema / Winograd-style (e.g., WinoGrande): different contexts, same continuation
  - Language modeling (LAMBADA): predict the final word given context

Ported from nanochat (github.com/karpathy/nanochat), adapted for our simpler
tokenizer interface: tokenize_fn(text) -> list[int], no BOS token handling.
Model forward and scoring logic lives in Task 2 (not here).

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
"""

from jinja2 import Template
import torch


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
