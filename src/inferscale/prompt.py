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
