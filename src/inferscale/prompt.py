"""Chat templates for backends that take raw text (Triton's generate endpoint).

vLLM's OpenAI server applies the model's own chat template from the tokenizer.
Triton receives a pre-rendered prompt, so the gateway renders it here. The default
template must match the deployed model, otherwise the two backends are not
generating from the same input and the benchmark comparison is invalid.
"""

from __future__ import annotations

from collections.abc import Sequence

from inferscale.schemas import ChatMessage


def _chatml(messages: Sequence[ChatMessage]) -> str:
    parts = [f"<|im_start|>{m.role}\n{m.content}<|im_end|>\n" for m in messages]
    parts.append("<|im_start|>assistant\n")
    return "".join(parts)


def _llama3(messages: Sequence[ChatMessage]) -> str:
    parts = ["<|begin_of_text|>"]
    for m in messages:
        parts.append(f"<|start_header_id|>{m.role}<|end_header_id|>\n\n{m.content}<|eot_id|>")
    parts.append("<|start_header_id|>assistant<|end_header_id|>\n\n")
    return "".join(parts)


def _plain(messages: Sequence[ChatMessage]) -> str:
    lines = [f"{m.role.capitalize()}: {m.content}" for m in messages]
    lines.append("Assistant:")
    return "\n".join(lines)


TEMPLATES = {"chatml": _chatml, "llama3": _llama3, "plain": _plain}

# End-of-turn markers that should stop generation for each template, in case the
# engine does not treat them as EOS on its own.
TEMPLATE_STOPS = {
    "chatml": ("<|im_end|>",),
    "llama3": ("<|eot_id|>",),
    "plain": ("\nUser:",),
}


def render(messages: Sequence[ChatMessage], template: str) -> str:
    try:
        return TEMPLATES[template](messages)
    except KeyError:
        raise ValueError(
            f"unknown chat template {template!r}; expected one of {sorted(TEMPLATES)}"
        ) from None


def prefix_key(messages: Sequence[ChatMessage], chars: int = 1024) -> str:
    """The cacheable prefix of a conversation: everything before the newest
    message (system prompt, instructions, earlier turns), cut to ``chars``.

    Engines with prefix caching (vLLM, TensorRT-LLM) reuse KV blocks for a shared
    leading token sequence. Two requests with the same key share that sequence,
    so sending them to the same replica turns the second prefill into a cache hit.
    A single-message request has no stable prefix, so its own text is the key.
    """
    stable = messages[:-1] if len(messages) > 1 else messages
    return "".join(f"{m.role}\x1f{m.content}\x1e" for m in stable)[:chars]
