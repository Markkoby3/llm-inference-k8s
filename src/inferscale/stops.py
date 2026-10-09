"""Streaming-safe stop-sequence handling for backends that cannot apply it themselves.

Triton's generate endpoint only accepts scalar request parameters, so a list of
stop strings cannot be forwarded. The gateway applies them instead. The hard part
is streaming: a stop string can be split across chunks ("<|im_" + "end|>"), so the
filter holds back any trailing text that could still become a stop match and
releases it once it is disambiguated.
"""

from __future__ import annotations

from collections.abc import Iterable


class StopSequenceFilter:
    def __init__(self, stops: Iterable[str]):
        self._stops = tuple(s for s in stops if s)
        self._buffer = ""
        self.stopped = False

    def feed(self, text: str) -> str:
        """Add streamed text; return the portion that is safe to emit now."""
        if self.stopped:
            return ""
        if not self._stops:
            return text

        self._buffer += text
        cut = self._earliest_match(self._buffer)
        if cut is not None:
            emitted, self._buffer = self._buffer[:cut], ""
            self.stopped = True
            return emitted

        hold = self._longest_partial_suffix(self._buffer)
        split = len(self._buffer) - hold
        emitted, self._buffer = self._buffer[:split], self._buffer[split:]
        return emitted

    def flush(self) -> str:
        """Release held-back text once the stream has ended without a match."""
        if self.stopped:
            return ""
        emitted, self._buffer = self._buffer, ""
        return emitted

    def apply(self, text: str) -> tuple[str, bool]:
        """Non-streaming helper: truncate at the first stop. Returns (text, matched)."""
        emitted = self.feed(text) + self.flush()
        return emitted, self.stopped

    def _earliest_match(self, text: str) -> int | None:
        positions = [i for i in (text.find(s) for s in self._stops) if i != -1]
        return min(positions) if positions else None

    def _longest_partial_suffix(self, text: str) -> int:
        longest = 0
        for stop in self._stops:
            for k in range(min(len(stop) - 1, len(text)), longest, -1):
                if text.endswith(stop[:k]):
                    longest = k
                    break
        return longest
