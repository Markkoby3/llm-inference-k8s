from __future__ import annotations

import pytest

from inferscale.prompt import render
from inferscale.schemas import ChatMessage
from inferscale.stops import StopSequenceFilter


def stream_through(stops: list[str], chunks: list[str]) -> tuple[str, bool]:
    f = StopSequenceFilter(stops)
    out = "".join(f.feed(c) for c in chunks) + f.flush()
    return out, f.stopped


def test_no_stops_is_passthrough():
    assert stream_through([], ["a", "b"]) == ("ab", False)


def test_stop_inside_a_single_chunk():
    assert stream_through(["END"], ["hello END world"]) == ("hello ", True)


def test_stop_split_across_chunks_is_caught():
    assert stream_through(["<|im_end|>"], ["answer<|im", "_e", "nd|> trailing"]) == ("answer", True)


def test_partial_match_that_never_completes_is_released():
    assert stream_through(["<|im_end|>"], ["a <|im", " b"]) == ("a <|im b", False)


def test_held_text_is_not_emitted_early():
    f = StopSequenceFilter(["STOP"])
    assert f.feed("go ST") == "go "
    assert f.feed("ART") == "START"


def test_earliest_of_multiple_stops_wins():
    assert stream_through(["zzz", "b"], ["a b c zzz"]) == ("a ", True)


def test_nothing_after_stop():
    f = StopSequenceFilter(["X"])
    f.feed("aXb")
    assert f.feed("more") == "" and f.flush() == ""


def test_chatml_render_matches_qwen_format():
    prompt = render(
        [ChatMessage(role="system", content="Be brief."), ChatMessage(role="user", content="Hi")],
        "chatml",
    )
    assert prompt == (
        "<|im_start|>system\nBe brief.<|im_end|>\n"
        "<|im_start|>user\nHi<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def test_unknown_template_fails_fast():
    with pytest.raises(ValueError, match="unknown chat template"):
        render([ChatMessage(role="user", content="x")], "mistral")
